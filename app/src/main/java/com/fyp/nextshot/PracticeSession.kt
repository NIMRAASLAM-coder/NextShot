// PracticeSession.kt — MediaPipe-only (upload + live)
// Roboflow / OkHttp removed. Both video-upload and live-camera modes now
// run fully on-device using MediaPipe PoseLandmarker + ObjectDetector.

package com.fyp.nextshot

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.*
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.io.ByteArrayOutputStream
import androidx.activity.viewModels
import com.fyp.nextshot.data.local.database.AppDatabase
import com.fyp.nextshot.data.local.models.SessionEntity
import com.fyp.nextshot.data.repository.SessionRepository
import com.fyp.nextshot.ui.viewmodel.SessionViewModel
import com.fyp.nextshot.ui.viewmodel.SessionViewModelFactory
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

// ═════════════════════════════════════════════════════════════════════════════
// Kalman Filter for single keypoint
// ═════════════════════════════════════════════════════════════════════════════
class KeypointKalmanFilter(
    private val processNoise: Float = 0.00005f,
    private val measurementNoise: Float = 0.001f,
    private val velocityDeadband: Float = 0.0f
) {
    private var x = 0f; private var y = 0f
    private var vx = 0f; private var vy = 0f
    private var px = 1f; private var py = 1f
    private var pvx = 0.001f; private var pvy = 0.001f

    fun update(measuredX: Float, measuredY: Float, confidence: Float): Pair<Float, Float> {
        if (!measuredX.isFinite() || !measuredY.isFinite()) return x to y
        val mNoise = measurementNoise / confidence.coerceIn(0.1f, 1.0f)
        x += vx; y += vy
        px += pvx + processNoise; py += pvy + processNoise
        pvx += processNoise; pvy += processNoise
        val kx = px / (px + mNoise); val ky = py / (py + mNoise)
        val dx = measuredX - x; val dy = measuredY - y
        x += kx * dx; y += ky * dy
        vx = vx * 0.8f + (kx * dx) * 0.2f
        vy = vy * 0.8f + (ky * dy) * 0.2f
        if (velocityDeadband > 0f) {
            if (abs(vx) < velocityDeadband) vx = 0f
            if (abs(vy) < velocityDeadband) vy = 0f
        }
        vx = vx.coerceIn(-0.1f, 0.1f); vy = vy.coerceIn(-0.1f, 0.1f)
        px *= (1f - kx); py *= (1f - ky)
        if (!x.isFinite() || !y.isFinite()) reset()
        return x to y
    }

    fun reset() {
        x = 0f; y = 0f; vx = 0f; vy = 0f
        px = 1f; py = 1f; pvx = 0.001f; pvy = 0.001f
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SmoothedDetection wrapper
// ═════════════════════════════════════════════════════════════════════════════
data class SmoothedDetection(
    val raw: Detection,
    val kalmanFilters: List<KeypointKalmanFilter>,
    val smoothedKeypoints: List<Keypoint>
) {
    fun getSmoothedKeypoint(idx: Int): Keypoint? =
        if (idx < smoothedKeypoints.size) smoothedKeypoints[idx] else null
}

class PracticeSession : AppCompatActivity() {

    private val TAG = "NEXTSHOT_DEBUG"

    private lateinit var previewView: PreviewView
    private lateinit var videoView: VideoView
    private lateinit var cameraOverlay: BoundingBoxOverlay
    private lateinit var videoOverlay: BoundingBoxOverlay
    private lateinit var videoContainer: FrameLayout
    private lateinit var cameraContainer: FrameLayout
    private lateinit var playPauseBtn: ImageView
    private lateinit var headTv: TextView
    private lateinit var shouldersTv: TextView
    private lateinit var weightTv: TextView
    private lateinit var feetTv: TextView

    // Video sync
    private val processedDetections = TreeMap<Long, Detection>()
    private val syncHandler = Handler(Looper.getMainLooper())
    private var isVideoPlaying = false
    private var videoCompleted = false

    private var videoFrameW = 640
    private var videoFrameH = 480

    private var isLiveMode = true
    private lateinit var progressDialog: android.app.AlertDialog
    private var mediaPlayer: MediaPlayer? = null

    private var currentShotType = ""
    private var lastSeenShotEventCount = 0

    // Live mode
    private val liveHandler = Handler(Looper.getMainLooper())
    private var lastGoodLiveDetection: Detection? = null
    private val liveKalmanFilters = mutableListOf<KeypointKalmanFilter>()
    private var liveFrameW = 256
    private var liveFrameH = 240

    private val liveShotDetector = ShotEventDetector()
    private val finalizedVideoShots = TreeMap<Long, String>()

    private lateinit var cameraExecutor: ExecutorService

    // MediaPipe
    private val COCO_TO_MP = intArrayOf(0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    @Volatile private var mediaPipeLandmarker: PoseLandmarker? = null
    @Volatile private var mediaPipeDetector: ObjectDetector? = null
    @Volatile private var livePoseLandmarker: PoseLandmarker? = null

    // Analysis state
    private var headStabilityScore = 100f
    private var shoulderScore = 100f
    private var weightScore = 100f
    private var footworkScore = 100f
    private var weightShiftText = "100%"
    private var headStatus = "100%"
    private var shoulderStatus = "100%"
    private var footworkStatus = "100%"

    private val headHistory = java.util.ArrayDeque<Pair<Float, Float>>()
    private val HISTORY_SIZE = 10
    private var lastFootPosition: Pair<Float, Float>? = null

    private var headBadCount = 0
    private var shoulderBadCount = 0
    private var weightBadCount = 0
    private var footworkBadCount = 0
    private val PERSISTENCE_FRAMES = 15
    private val MAX_PERSISTENCE = 30

    // Session
    private val auth by lazy { FirebaseAuth.getInstance() }
    private val db by lazy { FirebaseFirestore.getInstance() }
    private val userId = auth.currentUser?.uid ?: "FALLBACK_UID"
    private val database by lazy { AppDatabase.getDatabase(applicationContext) }
    private val repository by lazy { SessionRepository(database.sessionDao(), userId, db) }
    private val sessionViewModel: SessionViewModel by viewModels {
        SessionViewModelFactory(repository)
    }
    private var sessionStartTime = 0L

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_practice_session)

        previewView    = findViewById(R.id.preview_view)
        videoView      = findViewById(R.id.video_view)
        cameraOverlay  = findViewById(R.id.bbox_overlay)
        videoOverlay   = findViewById(R.id.video_overlay)
        videoContainer = findViewById(R.id.video_container)
        cameraContainer= findViewById(R.id.camera_container)
        playPauseBtn   = findViewById(R.id.btn_play_pause)
        headTv         = findViewById(R.id.tv_head_stability)
        shouldersTv    = findViewById(R.id.tv_shoulders)
        weightTv       = findViewById(R.id.tv_weight_balance)
        feetTv         = findViewById(R.id.tv_footwork)

        cameraExecutor = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors().coerceAtLeast(4)
        )

        findViewById<View>(R.id.btn_upload_video).setOnClickListener { pickVideo() }
        findViewById<View>(R.id.btn_live_record).setOnClickListener { enterLiveMode() }
        playPauseBtn.visibility = View.GONE

        videoContainer.setOnClickListener {
            if (videoView.isPlaying) {
                videoView.pause()
                isVideoPlaying = false
                syncHandler.removeCallbacks(syncRunnable)
                Toast.makeText(this, "Paused", Toast.LENGTH_SHORT).show()
            } else {
                if (videoCompleted) {
                    videoCompleted = false
                    videoView.seekTo(0)
                    processedDetections.lastEntry()?.let { videoOverlay.update(listOf(it.value), "") }
                }
                videoView.start()
                isVideoPlaying = true
                syncHandler.removeCallbacks(syncRunnable)
                syncHandler.post(syncRunnable)
                Toast.makeText(this, "Resuming", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<View>(R.id.btn_save_session).setOnClickListener { saveSession() }

        setupProgressDialog()

        if (allPermissionsGranted()) {
            cameraOverlay.setImageSize(1, 1)
            startCamera()
        } else {
            requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) { cameraOverlay.setImageSize(1, 1); startCamera() }
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, android.Manifest.permission.CAMERA
    ) == android.content.pm.PackageManager.PERMISSION_GRANTED

    private fun setupProgressDialog() {
        val builder = android.app.AlertDialog.Builder(this)
        builder.setCancelable(false)
        builder.setView(ProgressBar(this).apply { setPadding(50, 50, 50, 50) })
        builder.setMessage("Processing Video… Please Wait")
        progressDialog = builder.create()
    }

    // ── Video upload ──────────────────────────────────────────────────────────

    private fun pickVideo() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "video/*" }
        videoPicker.launch(intent)
    }

    private val videoPicker = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data?.data != null)
            startVideoAnalysis(result.data!!.data!!)
    }

    private fun startVideoAnalysis(uri: Uri) {
        isLiveMode = false
        liveHandler.removeCallbacks(liveSyncRunnable)
        resetAnalysisState()

        cameraContainer.visibility = View.GONE
        videoContainer.visibility  = View.VISIBLE
        videoOverlay.clear()

        videoView.stopPlayback()
        syncHandler.removeCallbacks(syncRunnable)
        videoView.setVideoURI(uri)
        videoView.setOnPreparedListener { mp -> mediaPlayer = mp; mp.isLooping = false }

        runOnUiThread { progressDialog.show() }
        cameraExecutor.execute { preProcessVideo(uri) }
    }

    /** Two-pass video analysis — 100 % on-device MediaPipe. */
    private fun preProcessVideo(uri: Uri) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(this, uri)
            val durationMs = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L

            val rawW = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 640
            val rawH = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 480
            val rotation = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val needsSwap = rotation == 90 || rotation == 270
            val effectiveW = if (needsSwap) rawH else rawW
            val effectiveH = if (needsSwap) rawW else rawH

            val arScale = minOf(640f / effectiveW, 640f / effectiveH, 1.0f)
            videoFrameW = (effectiveW * arScale).toInt().coerceAtLeast(1)
            videoFrameH = (effectiveH * arScale).toInt().coerceAtLeast(1)
            Log.d(TAG, "Video: ${rawW}x${rawH} rot=${rotation}° → send as ${videoFrameW}x${videoFrameH}")

            processedDetections.clear()

            // ── Pass 1: fast local motion scan ─────────────────────────────────
            val DENSE_INTERVAL_MS = 200L
            val coarseInterval    = 200L
            val THUMB_W = 32; val THUMB_H = 18
            val MOTION_THRESHOLD = 8.0

            fun bitmapToGray(bmp: Bitmap): IntArray {
                val scaled = Bitmap.createScaledBitmap(bmp, THUMB_W, THUMB_H, false)
                val pixels = IntArray(THUMB_W * THUMB_H)
                scaled.getPixels(pixels, 0, THUMB_W, 0, 0, THUMB_W, THUMB_H)
                if (scaled !== bmp) scaled.recycle()
                return pixels.map { p ->
                    val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                    (r * 299 + g * 587 + b * 114) / 1000
                }.toIntArray()
            }

            fun grayDiff(a: IntArray, b: IntArray) =
                if (a.size != b.size) 0.0
                else a.zip(b.toList()).sumOf { (pa, pb) -> Math.abs(pa - pb) }.toDouble() / a.size

            val allCoarseTimes = mutableListOf<Long>().apply {
                var t = 0L; while (t < durationMs) { add(t); t += coarseInterval }
            }
            val NUM_SCAN_WORKERS = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
            val chunkSize = ceil(allCoarseTimes.size.toFloat() / NUM_SCAN_WORKERS).toInt().coerceAtLeast(1)
            val chunks = allCoarseTimes.chunked(chunkSize)
            val activeTimestamps = java.util.Collections.synchronizedSet(mutableSetOf<Long>())

            val pass1Futures = chunks.map { chunk ->
                cameraExecutor.submit {
                    val wr = MediaMetadataRetriever()
                    try {
                        wr.setDataSource(this, uri)
                        var lastGray: IntArray? = null
                        for (t in chunk) {
                            val bmp = wr.getFrameAtTime(t * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                            if (bmp != null) {
                                val gray = bitmapToGray(bmp); bmp.recycle()
                                if (lastGray != null && grayDiff(lastGray!!, gray) > MOTION_THRESHOLD) {
                                    val ws = (t - 400L).coerceAtLeast(0L)
                                    val we = (t + 400L).coerceAtMost(durationMs)
                                    var ts = ws
                                    while (ts <= we) {
                                        activeTimestamps.add(ts - (ts % DENSE_INTERVAL_MS)); ts += DENSE_INTERVAL_MS
                                    }
                                }
                                lastGray = gray
                            }
                        }
                    } finally { wr.release() }
                }
            }
            pass1Futures.forEach { it.get() }
            Log.d(TAG, "Pass 1 done. Active timestamps: ${activeTimestamps.size}")

            val timestampsToProcess: Set<Long> = if (activeTimestamps.isEmpty()) {
                Log.d(TAG, "No motion — full even scan")
                mutableSetOf<Long>().apply { var t = 0L; while (t < durationMs) { add(t); t += DENSE_INTERVAL_MS } }
            } else activeTimestamps

            // ── Pass 2: MediaPipe pose per frame ───────────────────────────────
            val MAX_PARALLEL = Runtime.getRuntime().availableProcessors().coerceAtLeast(4)
            Log.d(TAG, "Pass 2: ${timestampsToProcess.size} frames, workers=$MAX_PARALLEL")

            initMediaPipe()
            initObjectDetector()

            val semaphore = java.util.concurrent.Semaphore(MAX_PARALLEL)
            val futures = timestampsToProcess.sorted().map { timeMs ->
                semaphore.acquire()
                cameraExecutor.submit {
                    try {
                        val wr = MediaMetadataRetriever()
                        wr.setDataSource(this, uri)
                        val bitmap = wr.getFrameAtTime(timeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                        wr.release()
                        if (bitmap != null) {
                            val detection = runMediaPipePose(bitmap)
                            bitmap.recycle()
                            if (detection != null) {
                                synchronized(processedDetections) { processedDetections[timeMs] = detection }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Worker failed at $timeMs", e)
                    } finally { semaphore.release() }
                }
            }
            futures.forEach { try { it.get() } catch (e: Exception) { Log.e(TAG, "Frame error", e) } }

            val count = processedDetections.size
            Log.d(TAG, "Analysis complete. Frames processed: $count")

            if (count > 0) {
                val shotLabel = classifyVideoShot(processedDetections)
                if (shotLabel.isNotEmpty()) {
                    val midKey = processedDetections.keys.toList()
                        .getOrElse(processedDetections.size / 2) { processedDetections.firstKey() }
                    finalizedVideoShots[midKey] = shotLabel
                    finalizedVideoShots[0L]     = shotLabel
                    currentShotType             = shotLabel
                }
            }

            runOnUiThread {
                progressDialog.dismiss()
                if (count > 0) {
                    Toast.makeText(this, "Analyzed $count frames!", Toast.LENGTH_SHORT).show()
                    startSyncedPlayback()
                } else {
                    Toast.makeText(this, "No body detected in video.", Toast.LENGTH_LONG).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Pre-processing error", e)
            runOnUiThread {
                progressDialog.dismiss()
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        } finally { retriever.release() }
    }

    // ── Playback sync ─────────────────────────────────────────────────────────

    private fun startSyncedPlayback() {
        videoCompleted = false
        videoView.start()
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M)
                mediaPlayer?.playbackParams = mediaPlayer?.playbackParams?.setSpeed(1f)
                    ?: android.media.PlaybackParams().setSpeed(1f)
        } catch (e: Exception) { Log.e(TAG, "Playback speed error", e) }

        isVideoPlaying  = true
        sessionStartTime = System.currentTimeMillis()
        videoOverlay.setImageSize(videoFrameW, videoFrameH)

        videoView.setOnCompletionListener {
            videoCompleted  = true
            isVideoPlaying  = false
            syncHandler.removeCallbacks(syncRunnable)
            processedDetections.lastEntry()?.let { entry ->
                val label = finalizedVideoShots.values.lastOrNull()?.takeIf { it.isNotEmpty() } ?: currentShotType
                videoOverlay.update(listOf(entry.value), label)
            }
        }
        syncHandler.post(syncRunnable)
    }

    private val syncRunnable = object : Runnable {
        override fun run() {
            if (videoCompleted || (!isVideoPlaying && !videoView.isPlaying)) return
            try {
                val pos      = videoView.currentPosition.toLong()
                val entry    = processedDetections.floorEntry(pos)
                val ceilEntry= processedDetections.ceilingEntry(pos)
                val best: Detection? = when {
                    entry == null && ceilEntry == null -> null
                    entry == null  -> ceilEntry!!.value
                    ceilEntry == null -> entry.value
                    else -> {
                        val span  = (ceilEntry.key - entry.key).toFloat()
                        if (span < 1f) entry.value
                        else interpolateDetections(entry.value, ceilEntry.value,
                            ((pos - entry.key) / span).toFloat().coerceIn(0f, 1f))
                    }
                }
                if (best != null) {
                    videoOverlay.update(listOf(best), "")
                    analyzePose(best)
                    headTv.text      = headStatus
                    shouldersTv.text = shoulderStatus
                    weightTv.text    = weightShiftText
                    feetTv.text      = footworkStatus
                }
            } catch (e: Exception) { Log.e(TAG, "Sync error", e) }
            syncHandler.postDelayed(this, 33)
        }
    }

    // ── Live mode ─────────────────────────────────────────────────────────────

    private val liveSyncRunnable = object : Runnable {
        override fun run() {
            if (!isLiveMode) return
            val det = lastGoodLiveDetection
            if (det != null) {
                cameraOverlay.update(listOf(det), currentShotType)
                analyzePose(det)
                headTv.text      = headStatus
                shouldersTv.text = shoulderStatus
                weightTv.text    = weightShiftText
                feetTv.text      = footworkStatus
            } else {
                cameraOverlay.clear()
            }
            liveHandler.postDelayed(this, 16)
        }
    }

    private fun enterLiveMode() {
        if (videoView.isPlaying) videoView.stopPlayback()
        isVideoPlaying = false
        isLiveMode     = true
        resetAnalysisState()
        syncHandler.removeCallbacks(syncRunnable)

        videoContainer.visibility  = View.GONE
        cameraContainer.visibility = View.VISIBLE

        cameraExecutor.execute { initLiveMediaPipe() }

        liveHandler.removeCallbacks(liveSyncRunnable)
        liveHandler.post(liveSyncRunnable)
        Toast.makeText(this, "Live Mode Active", Toast.LENGTH_SHORT).show()
    }

    private fun initLiveMediaPipe() {
        livePoseLandmarker?.close()
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setMinPosePresenceConfidence(0.3f)
                .setResultListener { result, _ ->
                    if (result.landmarks().isEmpty()) { lastGoodLiveDetection = null; return@setResultListener }
                    val landmarks  = result.landmarks()[0]
                    val keypoints  = COCO_TO_MP.map { mpIdx ->
                        val lm = landmarks.getOrNull(mpIdx)
                        if (lm != null) Keypoint(lm.x(), lm.y(), lm.visibility().orElse(0f))
                        else            Keypoint(0f, 0f, 0f)
                    }
                    val highConf = keypoints.count { it.confidence > 0.5f }
                    if (highConf < 5) { lastGoodLiveDetection = null; return@setResultListener }
                    val visKps   = keypoints.filter { it.confidence > 0.3f }
                    if (visKps.map { it.confidence }.average() < 0.5) {
                        lastGoodLiveDetection = null; return@setResultListener
                    }
                    val bbox = RectF(visKps.minOf { it.x }, visKps.minOf { it.y },
                        visKps.maxOf { it.x }, visKps.maxOf { it.y })
                    val avgConf   = keypoints.map { it.confidence }.average().toFloat()
                    val detection = Detection("batsman", avgConf, bbox, keypoints)

                    if (liveKalmanFilters.size != keypoints.size) {
                        liveKalmanFilters.clear()
                        repeat(keypoints.size) {
                            liveKalmanFilters.add(KeypointKalmanFilter(0.00001f, 0.005f, 0.001f))
                        }
                    }
                    val smoothed = keypoints.mapIndexed { i, kp ->
                        val (sx, sy) = liveKalmanFilters[i].update(kp.x, kp.y, kp.confidence)
                        Keypoint(sx, sy, kp.confidence)
                    }
                    lastGoodLiveDetection = detection.copy(keypoints = smoothed)

                    if (sessionStartTime == 0L) sessionStartTime = System.currentTimeMillis()
                    val shot = liveShotDetector.feed(smoothed)
                    if (shot.isNotEmpty()) currentShotType = shot
                    val newCount = liveShotDetector.shotEventCount
                    if (newCount > lastSeenShotEventCount && shot.isNotEmpty()) {
                        lastSeenShotEventCount = newCount
                        runOnUiThread { Toast.makeText(this, "Shot: $shot", Toast.LENGTH_SHORT).show() }
                    }
                }
                .setErrorListener { e -> Log.e(TAG, "Live MediaPipe error: ${e.message}") }
                .build()
            livePoseLandmarker = PoseLandmarker.createFromOptions(this, options)
            Log.d(TAG, "Live LIVE_STREAM PoseLandmarker initialised")
        } catch (e: Exception) { Log.e(TAG, "initLiveMediaPipe failed", e) }
    }

    @OptIn(ExperimentalGetImage::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy -> processFrame(proxy) } }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.camera.core.ExperimentalGetImage
    private fun processFrame(proxy: ImageProxy) {
        if (!isLiveMode) { proxy.close(); return }
        val landmarker = livePoseLandmarker
        if (landmarker == null) { proxy.close(); return }

        val tsMs = SystemClock.uptimeMillis()
        val bitmap = proxy.toBitmap()
        var rotated = bitmap
        if (proxy.imageInfo.rotationDegrees != 0)
            rotated = rotateBitmap(bitmap, proxy.imageInfo.rotationDegrees.toFloat()).also {
                if (it !== bitmap) bitmap.recycle()
            }
        proxy.close()

        val fw = rotated.width; val fh = rotated.height
        if (fw != liveFrameW || fh != liveFrameH) {
            liveFrameW = fw; liveFrameH = fh
            runOnUiThread { cameraOverlay.setImageSize(fw, fh) }
        }
        try {
            val argbBmp = if (rotated.config == Bitmap.Config.ARGB_8888) rotated
            else rotated.copy(Bitmap.Config.ARGB_8888, false)
            landmarker.detectAsync(BitmapImageBuilder(argbBmp).build(), tsMs)
            if (argbBmp !== rotated) argbBmp.recycle()
        } catch (e: Exception) { Log.e(TAG, "detectAsync error", e) }
        finally { if (rotated !== bitmap) rotated.recycle() else bitmap.recycle() }
    }

    // ── MediaPipe helpers ─────────────────────────────────────────────────────

    private fun initMediaPipe() {
        if (mediaPipeLandmarker != null) return
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build()
            mediaPipeLandmarker = PoseLandmarker.createFromOptions(this,
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumPoses(1)
                    .setMinPoseDetectionConfidence(0.3f)
                    .setMinTrackingConfidence(0.3f)
                    .build())
            Log.d(TAG, "MediaPipe PoseLandmarker initialised")
        } catch (e: Exception) { Log.e(TAG, "MediaPipe init failed", e) }
    }

    private fun initObjectDetector() {
        if (mediaPipeDetector != null) return
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("efficientdet_lite0.tflite").build()
            mediaPipeDetector = ObjectDetector.createFromOptions(this,
                ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMaxResults(5)
                    .setScoreThreshold(0.4f)
                    .build())
            Log.d(TAG, "MediaPipe ObjectDetector initialised")
        } catch (e: Exception) { Log.e(TAG, "ObjectDetector init failed", e) }
    }

    /**
     * Run on-device pose estimation on a single bitmap.
     * Step 1: EfficientDet person detection → bounding box.
     * Step 2: BlazePose-Lite keypoints → COCO-17 remapped.
     */
    private fun runMediaPipePose(bitmap: Bitmap): Detection? {
        val landmarker = mediaPipeLandmarker ?: return null
        var argbBmp: Bitmap? = null
        return try {
            val imgW = bitmap.width.toFloat().coerceAtLeast(1f)
            val imgH = bitmap.height.toFloat().coerceAtLeast(1f)
            argbBmp = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
            val mpImage = BitmapImageBuilder(argbBmp).build()

            // Step 1 – person detection
            var batsmanBbox: RectF? = null
            mediaPipeDetector?.detect(mpImage)?.detections()
                ?.filter { d -> d.categories().any { it.categoryName() == "person" } }
                ?.maxByOrNull { d -> d.categories().firstOrNull { it.categoryName() == "person" }?.score() ?: 0f }
                ?.let { det ->
                    val r = det.boundingBox()
                    batsmanBbox = RectF(r.left/imgW, r.top/imgH, r.right/imgW, r.bottom/imgH)
                }

            // Step 2 – pose keypoints
            val poseResult = landmarker.detect(mpImage)
            if (poseResult.landmarks().isEmpty()) return null
            val landmarks = poseResult.landmarks()[0]
            val keypoints = COCO_TO_MP.map { mpIdx ->
                val lm = landmarks.getOrNull(mpIdx)
                if (lm != null) Keypoint(lm.x(), lm.y(), lm.visibility().orElse(0f))
                else            Keypoint(0f, 0f, 0f)
            }
            val bbox = batsmanBbox ?: run {
                val vis = keypoints.filter { it.confidence > 0.3f }
                if (vis.isNotEmpty()) RectF(vis.minOf { it.x }, vis.minOf { it.y },
                    vis.maxOf { it.x }, vis.maxOf { it.y })
                else RectF(0f, 0f, 1f, 1f)
            }
            Detection("batsman", keypoints.map { it.confidence }.average().toFloat(), bbox, keypoints)
        } catch (e: Exception) { Log.e(TAG, "MediaPipe inference error", e); null }
        finally { if (argbBmp !== null && argbBmp !== bitmap) argbBmp!!.recycle() }
    }

    // ── Shot classification (video) ───────────────────────────────────────────

    private fun classifyVideoShot(detections: TreeMap<Long, Detection>): String {
        if (detections.isEmpty()) return ""
        val keys  = detections.keys.toList(); val total = keys.size
        val (startIdx, endIdx) = if (total <= 4) 0 to total
        else { val skip = (total * 0.20).toInt().coerceAtLeast(1); skip to (total - skip) }

        fun wristPos(det: Detection): Pair<Float, Float>? {
            val kps = det.keypoints
            val lw = kps.getOrNull(9)?.takeIf  { it.confidence > 0.25f }
            val rw = kps.getOrNull(10)?.takeIf { it.confidence > 0.25f }
            return when { lw != null && rw != null -> (lw.x+rw.x)/2f to (lw.y+rw.y)/2f
                lw != null -> lw.x to lw.y; rw != null -> rw.x to rw.y; else -> null }
        }

        val labels = mutableListOf<String>()
        var prevWrist: Pair<Float, Float>? = null
        for (i in startIdx until endIdx) {
            val det = detections[keys[i]] ?: continue
            val cur = wristPos(det)
            val vel = if (prevWrist != null && cur != null) {
                val dx = cur.first - prevWrist!!.first; val dy = cur.second - prevWrist!!.second
                sqrt(dx*dx + dy*dy)
            } else 0f
            if (cur != null) prevWrist = cur
            val label = ShotClassifier.classify(det.keypoints, vel)
            if (label.isNotEmpty()) labels += label
        }
        if (labels.isEmpty()) {
            for (i in startIdx until endIdx) {
                val det = detections[keys[i]] ?: continue
                val label = ShotClassifier.classify(det.keypoints, 1f)
                if (label.isNotEmpty()) labels += label
            }
        }
        if (labels.isEmpty()) return ""
        val nonDef = labels.filter { it != "Defensive" }
        return if (nonDef.isNotEmpty()) nonDef.groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
        else "Defensive"
    }

    // ── Pose analysis ─────────────────────────────────────────────────────────

    private fun analyzePose(detection: Detection) {
        val kp = detection.keypoints
        fun getPt(idx: Int): Pair<Float,Float>? {
            if (idx >= kp.size) return null
            val k = kp[idx]; return if (k.confidence < 0.3f) null else k.x to k.y
        }

        // Head stability
        val nose = getPt(0)
        if (nose != null) {
            headHistory.addLast(nose)
            if (headHistory.size > HISTORY_SIZE) headHistory.removeFirst()
            if (headHistory.size > 2) {
                val avgX = headHistory.map { it.first }.average()
                val avgY = headHistory.map { it.second }.average()
                val variance = headHistory.sumOf { (it.first - avgX).pow(2) + (it.second - avgY).pow(2) } / headHistory.size
                val score = if (variance.toFloat() <= 0.0002f) 100f
                else (100f - ((variance.toFloat() - 0.0002f) / (0.0015f - 0.0002f)) * 100f).coerceIn(0f, 100f)
                headBadCount = if (score < 100f) minOf(headBadCount + 1, MAX_PERSISTENCE) else maxOf(headBadCount - 2, 0)
                val final = if (headBadCount >= PERSISTENCE_FRAMES) score else 100f
                headStabilityScore = headStabilityScore * 0.8f + final * 0.2f
                headStatus = "${headStabilityScore.toInt()}%"
            }
        }

        // Shoulders
        val ls = getPt(5); val rs = getPt(6)
        if (ls != null && rs != null) {
            val angle = abs(Math.toDegrees(atan2((rs.second - ls.second).toDouble(), (rs.first - ls.first).toDouble())).toFloat())
            val score = if (angle <= 4f) 100f else (100f - ((angle - 4f) / (20f - 4f)) * 100f).coerceIn(0f, 100f)
            shoulderBadCount = if (score < 100f) minOf(shoulderBadCount + 1, MAX_PERSISTENCE) else maxOf(shoulderBadCount - 2, 0)
            val final = if (shoulderBadCount >= PERSISTENCE_FRAMES) score else 100f
            shoulderScore  = shoulderScore * 0.8f + final * 0.2f
            shoulderStatus = "${shoulderScore.toInt()}%"
        }

        // Weight balance
        val lHip = getPt(11); val rHip = getPt(12)
        val lAnk = getPt(15); val rAnk = getPt(16)
        if (lHip != null && rHip != null && lAnk != null && rAnk != null) {
            val diff = abs((lHip.first + rHip.first) / 2 - (lAnk.first + rAnk.first) / 2)
            val score = if (diff <= 0.02f) 100f else (100f - ((diff - 0.02f) / (0.1f - 0.02f)) * 100f).coerceIn(0f, 100f)
            weightBadCount = if (score < 100f) minOf(weightBadCount + 1, MAX_PERSISTENCE) else maxOf(weightBadCount - 2, 0)
            val final = if (weightBadCount >= PERSISTENCE_FRAMES) score else 100f
            weightScore    = weightScore * 0.8f + final * 0.2f
            weightShiftText= "${weightScore.toInt()}%"
        }

        // Footwork
        if (lAnk != null && rAnk != null) {
            val cur = (lAnk.first + rAnk.first) / 2 to (lAnk.second + rAnk.second) / 2
            if (lastFootPosition != null) {
                val dist = sqrt((cur.first - lastFootPosition!!.first).pow(2) + (cur.second - lastFootPosition!!.second).pow(2))
                val score = if (dist <= 0.005f) 100f else (100f - ((dist - 0.005f) / (0.05f - 0.005f)) * 100f).coerceIn(0f, 100f)
                footworkBadCount = if (score < 100f) minOf(footworkBadCount + 1, MAX_PERSISTENCE) else maxOf(footworkBadCount - 2, 0)
                val final = if (footworkBadCount >= PERSISTENCE_FRAMES) score else 100f
                footworkScore  = footworkScore * 0.7f + final * 0.3f
                footworkStatus = "${footworkScore.toInt()}%"
            }
            lastFootPosition = cur
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun interpolateDetections(a: Detection, b: Detection, alpha: Float): Detection {
        fun lerp(x: Float, y: Float) = x + (y - x) * alpha
        val bbox = RectF(lerp(a.bbox.left,a.bbox.left), lerp(a.bbox.top,b.bbox.top),
            lerp(a.bbox.right,b.bbox.right), lerp(a.bbox.bottom,b.bbox.bottom))
        val kps = if (a.keypoints.size == b.keypoints.size)
            a.keypoints.zip(b.keypoints).map { (k1,k2) ->
                Keypoint(lerp(k1.x,k2.x), lerp(k1.y,k2.y), lerp(k1.confidence,k2.confidence)) }
        else a.keypoints
        return Detection(a.label, lerp(a.confidence,b.confidence), bbox, kps)
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap =
        if (degrees != 0f) Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height,
            Matrix().apply { postRotate(degrees) }, true) else bitmap

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer; val uBuffer = planes[1].buffer; val vBuffer = planes[2].buffer
        val nv21 = ByteArray(yBuffer.remaining() + uBuffer.remaining() + vBuffer.remaining())
        yBuffer.get(nv21, 0, yBuffer.remaining())
        vBuffer.get(nv21, yBuffer.limit(), vBuffer.remaining())
        uBuffer.get(nv21, yBuffer.limit() + vBuffer.limit(), uBuffer.remaining())
        val yuv = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(android.graphics.Rect(0, 0, width, height), 85, out)
        val bytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)!!
    }

    private fun resetAnalysisState() {
        liveShotDetector.reset(); finalizedVideoShots.clear(); currentShotType = ""
        headHistory.clear()
        headStabilityScore = 100f; shoulderScore = 100f; weightScore = 100f; footworkScore = 100f
        weightShiftText = "100%"; shoulderStatus = "100%"; headStatus = "100%"; footworkStatus = "100%"
        lastFootPosition = null
        headBadCount = 0; shoulderBadCount = 0; weightBadCount = 0; footworkBadCount = 0
        lastGoodLiveDetection = null; lastSeenShotEventCount = 0
        runOnUiThread { headTv.text = "--"; shouldersTv.text = "--"; weightTv.text = "--"; feetTv.text = "--" }
    }

    // ── Save session ──────────────────────────────────────────────────────────

    private fun saveSession() {
        if (sessionStartTime == 0L) sessionStartTime = System.currentTimeMillis()
        if (isLiveMode) liveShotDetector.flush()

        val duration = ((System.currentTimeMillis() - sessionStartTime) / 1000).toInt().coerceAtLeast(30)
        val summary  = "Head: $headStatus | Shoulders: $shoulderStatus | Weight: $weightShiftText | Feet: $footworkStatus"

        val totalShots: Int; val shotSummaryText: String
        if (isLiveMode) {
            totalShots      = liveShotDetector.shotEventCount
            val shotSummary = liveShotDetector.getShotSummary()
            shotSummaryText = if (shotSummary.isEmpty()) "No shots detected."
            else shotSummary.entries.sortedByDescending { it.value }.joinToString("\n") { (l, c) -> "  • $l  ×$c" }
        } else {
            val label = finalizedVideoShots.values.lastOrNull()?.takeIf { it.isNotEmpty() } ?: currentShotType
            totalShots      = if (label.isNotEmpty()) 1 else 0
            shotSummaryText = if (label.isNotEmpty()) "  • $label" else "No shot detected."
        }

        val message = buildString {
            append("Shots detected: $totalShots\n\n$shotSummaryText\n\n")
            append("─────────────────────\nPose Quality\n")
            append("  Head:      $headStatus\n  Shoulders: $shoulderStatus\n")
            append("  Weight:    $weightShiftText\n  Feet:      $footworkStatus")
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("Session Summary")
            .setMessage(message)
            .setPositiveButton("Save") { _, _ ->
                sessionViewModel.insert(SessionEntity(
                    userId = userId, drillType = "Pose Analysis",
                    durationSeconds = duration, successRate = 1.0,
                    flawDetails = summary, dateMillis = System.currentTimeMillis()
                ))
                Toast.makeText(this, "Session Saved!", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Discard", null)
            .show()
    }

    // ── Destroy ───────────────────────────────────────────────────────────────

    override fun onDestroy() {
        super.onDestroy()
        mediaPipeLandmarker?.close(); mediaPipeLandmarker = null
        mediaPipeDetector?.close();   mediaPipeDetector   = null
        livePoseLandmarker?.close();  livePoseLandmarker  = null
        cameraExecutor.shutdown()
        syncHandler.removeCallbacksAndMessages(null)
        liveHandler.removeCallbacksAndMessages(null)
    }
}