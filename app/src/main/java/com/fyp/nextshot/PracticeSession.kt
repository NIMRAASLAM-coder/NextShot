// PracticeSession.kt — ENHANCED KEYPOINT SMOOTHING v3
// Improvements over v2:
//   - Kalman Filter per keypoint for adaptive smoothing that reduces model noise
//     without introducing lag. Learns from detection confidence + velocity history.
//   - Confidence-weighted interpolation: low-confidence keypoints snap to nearest
//     detection; high-confidence keypoints smoothly interpolate.
//   - Velocity clamping: detects sudden jumps (likely model errors) and constrains
//     movement to physically plausible speeds.
//   - Adaptive alpha in EMA based on confidence: high confidence = faster response,
//     low confidence = more smoothing.
//   - Temporal smoothing via velocity history: prevents stutter by blending current
//     velocity with recent history.

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
import android.util.Base64
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
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
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
import java.util.concurrent.TimeUnit
import kotlin.math.*

// ═════════════════════════════════════════════════════════════════════════════
// Kalman Filter for single keypoint — reduces jitter while tracking motion
// ═════════════════════════════════════════════════════════════════════════════
class KeypointKalmanFilter(
    private val processNoise: Float = 0.00005f,   // Model error (lower = trust model more)
    private val measurementNoise: Float = 0.001f, // Sensor noise (lower = trust detections more)
    private val velocityDeadband: Float = 0.0f    // Zero out micro-velocities to stop drift when still
) {
    private var x = 0f  // Estimated position X
    private var y = 0f  // Estimated position Y
    private var vx = 0f // Estimated velocity X (px/frame)
    private var vy = 0f // Estimated velocity Y (px/frame)

    private var px = 1f  // Position uncertainty X
    private var py = 1f  // Position uncertainty Y
    private var pvx = 0.001f // Velocity uncertainty X
    private var pvy = 0.001f // Velocity uncertainty Y

    fun update(measuredX: Float, measuredY: Float, confidence: Float): Pair<Float, Float> {
        // Guard against NaN/Inf inputs (can happen if Roboflow returns garbage coords)
        if (!measuredX.isFinite() || !measuredY.isFinite()) return x to y

        val measurementTrust = confidence.coerceIn(0.1f, 1.0f)
        val mNoise = measurementNoise / measurementTrust

        // Predict: advance position by estimated velocity
        x += vx
        y += vy

        // Update uncertainties (they grow because of process noise)
        px += pvx + processNoise
        py += pvy + processNoise
        pvx += processNoise
        pvy += processNoise

        // Update step: blend prediction with measurement
        val kx = px / (px + mNoise)  // Kalman gain for X
        val ky = py / (py + mNoise)  // Kalman gain for Y

        val dx = measuredX - x
        val dy = measuredY - y

        x += kx * dx
        y += ky * dy

        // FIX: Velocity = simple EMA of the position correction (kx*dx).
        // The old formula divided dx by mNoise (~0.001) which amplified small
        // deltas by 1000x every frame, causing exponential blowup to Infinity.
        vx = vx * 0.8f + (kx * dx) * 0.2f
        vy = vy * 0.8f + (ky * dy) * 0.2f

        // Deadband: suppress micro-velocities so skeleton stays put when person is still
        if (velocityDeadband > 0f) {
            if (kotlin.math.abs(vx) < velocityDeadband) vx = 0f
            if (kotlin.math.abs(vy) < velocityDeadband) vy = 0f
        }

        // Safety clamp: velocity should never exceed ~10% of frame per update
        vx = vx.coerceIn(-0.1f, 0.1f)
        vy = vy.coerceIn(-0.1f, 0.1f)

        // Uncertainty shrinks after update
        px *= (1f - kx)
        py *= (1f - ky)

        // Guard against NaN/Inf state (reset if corrupted)
        if (!x.isFinite() || !y.isFinite()) reset()

        return x to y
    }

    fun predictAhead(framesDelta: Int): Pair<Float, Float> {
        // Project where the keypoint will be in N frames (for interpolation lookahead)
        val px = x + vx * framesDelta
        val py = y + vy * framesDelta
        return px to py
    }

    fun reset() {
        x = 0f
        y = 0f
        vx = 0f
        vy = 0f
        px = 1f
        py = 1f
        pvx = 0.001f
        pvy = 0.001f
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Smooth Detection wrapper: stores Kalman state per keypoint
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
    private var lastDetection: Detection? = null
    private var lastUpdateTime = 0L

    // For Video Sync
    private val processedDetections = TreeMap<Long, Detection>()
    private val syncHandler = Handler(Looper.getMainLooper())
    private var isVideoPlaying = false
    private var videoCompleted = false

    // Actual frame dimensions used when sending frames to Roboflow (aspect-ratio-aware)
    private var videoFrameW = 640
    private var videoFrameH = 480

    // Mode tracking to prevent interference
    private var isLiveMode = true
    private lateinit var progressDialog: android.app.AlertDialog
    private var mediaPlayer: MediaPlayer? = null

    // Live Mode — rolling detection buffer with Kalman smoothing
    private val pendingRequests = java.util.concurrent.atomic.AtomicInteger(0)
    private val MAX_CONCURRENT_REQUESTS = 6
    private var currentShotType = ""

    // Tracks the last shotEventCount we have already announced as a Toast.
    // When liveShotDetector.shotEventCount exceeds this, a new shot has fired.
    private var lastSeenShotEventCount = 0

    // Live detections: now stores SmoothedDetection (with Kalman state)
    private val liveDetections = java.util.concurrent.ConcurrentSkipListMap<Long, SmoothedDetection>()

    private val LIVE_BUFFER_DELAY_MS = 200L
    private val MAX_INTERP_GAP_MS = 500L

    /** Pre-computed per-timestamp finalized shot labels (video mode only). */
    private val finalizedVideoShots = TreeMap<Long, String>()

    /** Live-mode shot detector — processes frames as they arrive from the camera. */
    private val liveShotDetector = ShotEventDetector()

    private lateinit var cameraExecutor: ExecutorService

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    // ── Inference server config ──────────────────────────────────────────────────
    // LOCAL: change this IP to your laptop's WiFi IP when running the local server.
    // CLOUD: swap back to https://serverless.roboflow.com/hello-7pqr3/workflows/custom-workflow-3
    private val INFERENCE_URL = "http://10.141.124.10:9001/hello-7pqr3/workflows/custom-workflow-3"
    private val ROBOFLOW_API_KEY = "7VCjsMFfykWO22m0bCXb"

    // ANALYSIS STATE
    private var prevHeadCenterGlobal: Pair<Float, Float>? = null
    private var headStabilityScore = 100f
    private var shoulderScore = 100f
    private var weightScore = 100f
    private var footworkScore = 100f
    private var weightShiftText = "100%"
    private var isBalanced = true
    private var isProcessing = false

    // History Buffers for Analysis
    private val headHistory = java.util.ArrayDeque<Pair<Float, Float>>()
    private val HISTORY_SIZE = 10
    private var lastFootPosition: Pair<Float, Float>? = null
    private var footworkStatus = "100%"
    private var shoulderStatus = "100%"
    private var headStatus = "100%"

    // Persistence Counters (Damping)
    private var headBadCount = 0
    private var shoulderBadCount = 0
    private var weightBadCount = 0
    private var footworkBadCount = 0
    private val PERSISTENCE_FRAMES = 15
    private val MAX_PERSISTENCE = 30

    // Session Management
    private val auth by lazy { FirebaseAuth.getInstance() }
    private val db by lazy { FirebaseFirestore.getInstance() }
    private val userId = auth.currentUser?.uid ?: "FALLBACK_UID"
    private val database by lazy { AppDatabase.getDatabase(applicationContext) }
    private val repository by lazy { SessionRepository(database.sessionDao(), userId, db) }
    private val sessionViewModel: SessionViewModel by viewModels {
        SessionViewModelFactory(repository)
    }
    private var sessionStartTime = 0L

    // Tracks the frame size sent to Roboflow for live mode
    private var liveFrameW = 256
    private var liveFrameH = 240

    // Replace the liveDetections map usage with a single latest detection
    private var latestLiveDetection: Detection? = null
    private var lastGoodLiveDetection: Detection? = null
    private val liveKalmanFilters = mutableListOf<KeypointKalmanFilter>()

    private val liveHandler = Handler(Looper.getMainLooper())

    // 25002500 MediaPipe on-device pose landmarker (used by Pass 2) 25002500250025002500250025002500250025002500250025002500250025002500
    private val COCO_TO_MP = intArrayOf(0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    @Volatile private var mediaPipeLandmarker: PoseLandmarker? = null
    @Volatile private var mediaPipeDetector: ObjectDetector? = null
    // Separate landmarker for live mode — uses LIVE_STREAM (async, ultra-low latency)
    @Volatile private var livePoseLandmarker: PoseLandmarker? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_practice_session)

        previewView = findViewById(R.id.preview_view)
        videoView = findViewById(R.id.video_view)
        cameraOverlay = findViewById(R.id.bbox_overlay)
        videoOverlay = findViewById(R.id.video_overlay)
        videoContainer = findViewById(R.id.video_container)
        cameraContainer = findViewById(R.id.camera_container)
        playPauseBtn = findViewById(R.id.btn_play_pause)
        headTv = findViewById(R.id.tv_head_stability)
        shouldersTv = findViewById(R.id.tv_shoulders)
        weightTv = findViewById(R.id.tv_weight_balance)
        feetTv = findViewById(R.id.tv_footwork)

        cameraExecutor = Executors.newFixedThreadPool(24)

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
                // If the video finished, rewind so keypoints replay from the start
                if (videoCompleted) {
                    videoCompleted = false
                    videoView.seekTo(0)
                    // Clear the shot badge — it will reappear when the video ends again
                    val lastEntry = processedDetections.lastEntry()
                    if (lastEntry != null) videoOverlay.update(listOf(lastEntry.value), "")
                }
                videoView.start()
                isVideoPlaying = true
                syncHandler.removeCallbacks(syncRunnable) // avoid duplicate runnables
                syncHandler.post(syncRunnable)
                Toast.makeText(this, "Resuming", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<View>(R.id.btn_save_session).setOnClickListener {
            saveSession()
        }

        setupProgressDialog()

        if (allPermissionsGranted()) {
            // Set overlay size for live mode immediately (normalized 0–1 coords)
            cameraOverlay.setImageSize(1, 1)
            startCamera()
        } else {
            requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            cameraOverlay.setImageSize(1, 1)
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, android.Manifest.permission.CAMERA
    ) == android.content.pm.PackageManager.PERMISSION_GRANTED

    private fun setupProgressDialog() {
        val builder = android.app.AlertDialog.Builder(this)
        builder.setCancelable(false)
        builder.setView(ProgressBar(this).apply {
            setPadding(50, 50, 50, 50)
        })
        builder.setMessage("Processing Video... Please Wait")
        progressDialog = builder.create()
    }

    private fun pickVideo() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "video/*" }
        videoPicker.launch(intent)
    }

    private val videoPicker = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK && result.data?.data != null) {
            startVideoAnalysis(result.data!!.data!!)
        }
    }

    private fun startVideoAnalysis(uri: Uri) {
        isLiveMode = false
        liveHandler.removeCallbacks(liveSyncRunnable)
        resetAnalysisState()
        cameraContainer.visibility = View.GONE
        videoContainer.visibility = View.VISIBLE

        videoOverlay.clear()

        videoView.stopPlayback()
        syncHandler.removeCallbacks(syncRunnable)

        videoView.setVideoURI(uri)
        videoView.setOnPreparedListener { mp ->
            mediaPlayer = mp
            mp.isLooping = false
        }

        runOnUiThread {
            progressDialog.show()
        }
        cameraExecutor.execute {
            preProcessVideo(uri)
        }
    }

    private fun preProcessVideo(uri: Uri) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(this, uri)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLongOrNull() ?: 0L

            val rawW = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 640
            val rawH = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 480

            // METADATA_KEY_VIDEO_WIDTH/HEIGHT returns stored (pre-rotation) dimensions.
            // getFrameAtTime() auto-applies the rotation tag, so landmarks are normalized
            // to the *post-rotation* frame size. We must match that here.
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull() ?: 0
            val needsSwap = rotation == 90 || rotation == 270
            val effectiveW = if (needsSwap) rawH else rawW
            val effectiveH = if (needsSwap) rawW else rawH

            val arScale = minOf(640f / effectiveW, 640f / effectiveH, 1.0f)
            videoFrameW = (effectiveW * arScale).toInt().coerceAtLeast(1)
            videoFrameH = (effectiveH * arScale).toInt().coerceAtLeast(1)
            Log.d(TAG, "Video native: ${rawW}x${rawH}, rotation: ${rotation}°, " +
                    "effective: ${effectiveW}x${effectiveH}, sending as: ${videoFrameW}x${videoFrameH}")
            Log.d(TAG, "Starting pre-processing. Duration: $durationMs ms")

            processedDetections.clear()

            // ── PASS 1: Fast LOCAL motion scan (no API calls) ─────────────────────────
            // Compares tiny grayscale thumbnails (32×18 px) between consecutive coarse
            // frames — runs in milliseconds on-device.
            Log.d(TAG, "Pass 1: Local motion scan starting (no API calls)...")

            // ── Fixed interval math that guarantees ≤ half video duration ─────────────
            // Proof: numFrames = durationMs / 200
            //        numRounds  = numFrames / 16  (16 parallel workers)
            //        totalTime  = numRounds × 1500ms  (avg Roboflow latency)
            //                   = (durationMs/200)/16 × 1500 = durationMs × 0.47
            //        → always finishes in ~47% of video duration ✓
            // No frame-count cap needed — the interval itself is the budget guarantee.
            // On-device MediaPipe inference: saturate all CPU cores
            val MAX_PARALLEL_CALLS = Runtime.getRuntime().availableProcessors().coerceAtLeast(4)
            val DENSE_INTERVAL_MS  = 200L   // 5 fps keypoint data; syncRunnable lerps to 30fps
            val coarseInterval     = 200L   // local motion scan step

            val THUMB_W = 32
            val THUMB_H = 18
            val MOTION_THRESHOLD = 8.0

            fun bitmapToGrayThumb(bmp: Bitmap): IntArray {
                val scaled = Bitmap.createScaledBitmap(bmp, THUMB_W, THUMB_H, false)
                val pixels = IntArray(THUMB_W * THUMB_H)
                scaled.getPixels(pixels, 0, THUMB_W, 0, 0, THUMB_W, THUMB_H)
                if (scaled !== bmp) scaled.recycle()
                return pixels.map { p ->
                    val r = (p shr 16) and 0xFF
                    val g = (p shr 8)  and 0xFF
                    val b =  p         and 0xFF
                    (r * 299 + g * 587 + b * 114) / 1000
                }.toIntArray()
            }

            fun grayDiff(a: IntArray, b: IntArray): Double {
                if (a.size != b.size) return 0.0
                return a.zip(b.toList()).sumOf { (pa, pb) -> Math.abs(pa - pb) }.toDouble() / a.size
            }

            // Build the complete ordered list of coarse timestamps
            val allCoarseTimes = mutableListOf<Long>()
            var ct = 0L
            while (ct < durationMs) { allCoarseTimes.add(ct); ct += coarseInterval }

            // Split into one chunk per logical CPU core (max 8) so all segments
            // decode frames in parallel — cuts Pass 1 time by ~8x on 8-core devices.
            // Each worker owns its own MediaMetadataRetriever (NOT thread-safe).
            val NUM_SCAN_WORKERS = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
            val chunkSize = kotlin.math.ceil(allCoarseTimes.size.toFloat() / NUM_SCAN_WORKERS)
                .toInt().coerceAtLeast(1)
            val chunks = allCoarseTimes.chunked(chunkSize)

            // Thread-safe set: multiple workers write their motion windows concurrently
            val activeTimestamps = java.util.Collections.synchronizedSet(mutableSetOf<Long>())

            val pass1Futures = mutableListOf<java.util.concurrent.Future<*>>()
            for (chunk in chunks) {
                val future = cameraExecutor.submit {
                    val workerRetriever = MediaMetadataRetriever()
                    try {
                        workerRetriever.setDataSource(this@PracticeSession, uri)
                        var lastGray: IntArray? = null
                        for (coarseTime in chunk) {
                            val bmp = workerRetriever.getFrameAtTime(
                                coarseTime * 1000L, MediaMetadataRetriever.OPTION_CLOSEST
                            )
                            if (bmp != null) {
                                val gray = bitmapToGrayThumb(bmp)
                                bmp.recycle()
                                if (lastGray != null && grayDiff(lastGray!!, gray) > MOTION_THRESHOLD) {
                                    val windowStart = (coarseTime - 400L).coerceAtLeast(0L)
                                    val windowEnd   = (coarseTime + 400L).coerceAtMost(durationMs)
                                    var ts = windowStart
                                    while (ts <= windowEnd) {
                                        activeTimestamps.add(ts - (ts % DENSE_INTERVAL_MS))
                                        ts += DENSE_INTERVAL_MS
                                    }
                                }
                                lastGray = gray
                            }
                        }
                    } finally {
                        workerRetriever.release()
                    }
                }
                pass1Futures.add(future)
            }
            // Wait for all segments to finish before proceeding to Pass 2
            for (f in pass1Futures) f.get()

            Log.d(TAG, "Pass 1 complete ($NUM_SCAN_WORKERS workers). Active timestamps: ${activeTimestamps.size} @ ${DENSE_INTERVAL_MS}ms interval")

            // Fallback: if no motion detected, sample evenly across the whole video
            val timestampsToProcess: Set<Long> = if (activeTimestamps.isEmpty()) {
                Log.d(TAG, "No motion detected — full even scan at ${DENSE_INTERVAL_MS}ms")
                val fallback = mutableSetOf<Long>()
                var t = 0L
                while (t < durationMs) { fallback.add(t); t += DENSE_INTERVAL_MS }
                fallback
            } else {
                activeTimestamps   // No hard-cap — interval already guarantees budget
            }

            // ── PASS 2: Fully parallel — each worker decodes its own frame + calls API ──
            // KEY FIX: Previously getFrameAtTime() was called sequentially in the outer
            // loop BEFORE submitting, so frame decoding serialized the whole pass.
            // Now each task owns its own MediaMetadataRetriever so frame decode + API
            // call both happen in parallel across all workers simultaneously.
            Log.d(TAG, "Pass 2: Processing ${timestampsToProcess.size} frames — on-device MediaPipe fully parallel (workers=$MAX_PARALLEL_CALLS)...")

            // Initialise once on the calling thread before spawning workers
            initMediaPipe()
            initObjectDetector()

            val semaphore = java.util.concurrent.Semaphore(MAX_PARALLEL_CALLS)
            val futures = mutableListOf<java.util.concurrent.Future<*>>()

            for (timeForFrame in timestampsToProcess.sorted()) {
                semaphore.acquire()   // acquire BEFORE submitting to bound concurrency
                val future = cameraExecutor.submit {
                    try {
                        // Each worker creates its own retriever — MediaMetadataRetriever
                        // is NOT thread-safe so sharing the outer one would corrupt frames.
                        val workerRetriever = MediaMetadataRetriever()
                        workerRetriever.setDataSource(this@PracticeSession, uri)
                        val bitmap = workerRetriever.getFrameAtTime(
                            timeForFrame * 1000L, MediaMetadataRetriever.OPTION_CLOSEST
                        )
                        workerRetriever.release()

                        if (bitmap != null) {
                            val detection = runMediaPipePose(bitmap)
                            bitmap.recycle()
                            if (detection != null) {
                                synchronized(processedDetections) {
                                    processedDetections[timeForFrame] = detection
                                }
                            } else {
                                Log.w(TAG, "No detection for frame at $timeForFrame")
                            }
                        } else {
                            Log.w(TAG, "Could not retrieve frame at $timeForFrame")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Worker failed at $timeForFrame", e)
                    } finally {
                        semaphore.release()
                    }
                }
                futures.add(future)
            }

            for (f in futures) {
                try {
                    f.get()
                } catch (e: Exception) {
                    Log.e(TAG, "Frame processing failed", e)
                }
            }

            val count = processedDetections.size
            Log.d(TAG, "Analysis Complete. Processed frames: $count")

            if (count > 0) {
                val shotLabel = classifyVideoShot(processedDetections)
                Log.d("SHOT_DETECTOR", "Video shot classified as: $shotLabel")

                if (shotLabel.isNotEmpty()) {
                    // Store against the midpoint timestamp so the overlay
                    // shows the label from the very first frame of playback.
                    val midKey = processedDetections.keys.toList()
                        .getOrElse(processedDetections.size / 2) { processedDetections.firstKey() }
                    finalizedVideoShots[midKey] = shotLabel
                    // Also store at timestamp 0 so it shows immediately on play
                    finalizedVideoShots[0L] = shotLabel
                    currentShotType = shotLabel
                }
                Log.d(TAG, "Shot classification complete: $shotLabel")
            }

            runOnUiThread {
                progressDialog.dismiss()
                if (count > 0) {
                    Toast.makeText(this@PracticeSession, "Analyzed $count frames!", Toast.LENGTH_SHORT).show()
                    startSyncedPlayback()
                } else {
                    Toast.makeText(this@PracticeSession, "No body detected in video.", Toast.LENGTH_LONG).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Pre-processing error", e)
            runOnUiThread {
                progressDialog.dismiss()
                Toast.makeText(this@PracticeSession, "Error processing: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        } finally {
            retriever.release()
        }
    }

    private fun startSyncedPlayback() {
        videoCompleted = false
        videoView.start()

        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                mediaPlayer?.playbackParams = mediaPlayer?.playbackParams?.setSpeed(1f) ?: android.media.PlaybackParams().setSpeed(1f)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Could not set playback speed", e)
        }

        isVideoPlaying = true
        sessionStartTime = System.currentTimeMillis()

        videoOverlay.setImageSize(videoFrameW, videoFrameH)

        // Stop the sync runnable when the video finishes naturally.
        // videoCompleted flag is set FIRST so the next syncRunnable tick
        // sees it and returns before reading position=0 (VideoView quirk).
        videoView.setOnCompletionListener {
            videoCompleted = true
            isVideoPlaying = false
            syncHandler.removeCallbacks(syncRunnable)

            // Freeze the overlay on the very last detection AND show the shot badge
            val lastEntry = processedDetections.lastEntry()
            if (lastEntry != null) {
                val shotLabel = finalizedVideoShots.values.lastOrNull()
                    ?.takeIf { it.isNotEmpty() } ?: currentShotType
                videoOverlay.update(listOf(lastEntry.value), shotLabel)
                Log.d(TAG, "Video complete — overlay frozen, shot badge: $shotLabel")
            }
        }

        syncHandler.post(syncRunnable)
    }

    private val syncRunnable = object : Runnable {
        override fun run() {
            // Stop if video has completed — videoCompleted is set before the
            // completion listener fires, so this catches the position=0 flash too.
            if (videoCompleted) return

            // Stop if neither the flag nor the VideoView says we are playing.
            if (!isVideoPlaying && !videoView.isPlaying) return

            try {
                val currentPos = videoView.currentPosition.toLong()

                val entry     = processedDetections.floorEntry(currentPos)
                val ceilEntry = processedDetections.ceilingEntry(currentPos)

                val bestDetection: Detection? = when {
                    entry == null && ceilEntry == null -> null
                    entry == null -> ceilEntry!!.value
                    ceilEntry == null -> entry.value
                    else -> {
                        val t1    = entry.key
                        val t2    = ceilEntry.key
                        val span  = (t2 - t1).toFloat()
                        if (span < 1f) entry.value
                        else {
                            val alpha = ((currentPos - t1) / span).toFloat().coerceIn(0f, 1f)
                            interpolateDetections(entry.value, ceilEntry.value, alpha)
                        }
                    }
                }

                if (bestDetection != null) {
                    // Show skeleton during playback, but withhold the shot badge
                    // until the video finishes — displayed in onCompletionListener.
                    videoOverlay.update(listOf(bestDetection), "")

                    analyzePose(bestDetection)
                    headTv.text = headStatus
                    shouldersTv.text = shoulderStatus
                    weightTv.text = weightShiftText
                    feetTv.text = footworkStatus
                }

            } catch (e: Exception) {
                Log.e(TAG, "Sync error", e)
            }

            syncHandler.postDelayed(this, 33)
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // liveSyncRunnable — Render loop at 60fps
    //
    // FIX: The original code set latestLiveDetection = null after one frame,
    // then showed lastGoodLiveDetection on the next tick. This alternation
    // between fresh/stale data every 16ms was the direct cause of blinking.
    // Now: parseRoboflowResponse writes directly to lastGoodLiveDetection,
    // and this loop just renders it every tick without consuming it.
    // ─────────────────────────────────────────────────────────────────
    private val liveSyncRunnable = object : Runnable {
        override fun run() {
            if (!isLiveMode) return

            val toDisplay = lastGoodLiveDetection

            if (toDisplay != null) {
                cameraOverlay.update(listOf(toDisplay), currentShotType)
                analyzePose(toDisplay)
                headTv.text      = headStatus
                shouldersTv.text = shoulderStatus
                weightTv.text    = weightShiftText
                feetTv.text      = footworkStatus
            } else {
                // No person detected — clear any stale skeleton from the overlay
                cameraOverlay.clear()
            }

            liveHandler.postDelayed(this, 16)
        }
    }

    /**
     * Confidence-weighted interpolation:
     * - High conf (>0.7): smooth lerp across time
     * - Medium conf (0.3–0.7): partial lerp
     * - Low conf (<0.3): snap to nearest (no interpolation)
     */
    private fun confidenceWeightedInterpolation(
        kps1: List<Keypoint>,
        kps2: List<Keypoint>,
        alpha: Float
    ): Detection {
        val interpolated = kps1.zip(kps2).map { (k1, k2) ->
            val avgConf = (k1.confidence + k2.confidence) / 2f

            when {
                // High confidence: smooth interpolation
                avgConf >= 0.7f -> {
                    Keypoint(
                        x = k1.x + (k2.x - k1.x) * alpha,
                        y = k1.y + (k2.y - k1.y) * alpha,
                        confidence = (k1.confidence + k2.confidence) / 2f
                    )
                }
                // Medium confidence: dampened interpolation
                avgConf >= 0.3f -> {
                    val dampedAlpha = alpha * 0.6f  // Only 60% of movement
                    Keypoint(
                        x = k1.x + (k2.x - k1.x) * dampedAlpha,
                        y = k1.y + (k2.y - k1.y) * dampedAlpha,
                        confidence = avgConf
                    )
                }
                // Low confidence: snap to nearest detection
                else -> {
                    if (alpha < 0.5f) k1 else k2
                }
            }
        }

        return Detection(
            label = "batsman",
            confidence = (kps1[0].confidence + kps2[0].confidence) / 2f,
            bbox = RectF(0f, 0f, 1f, 1f),  // Placeholder
            keypoints = interpolated
        )
    }

    @OptIn(ExperimentalGetImage::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { proxy ->
                        if (!isProcessing) processFrame(proxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Initialise a LIVE_STREAM PoseLandmarker — async, non-blocking, pipelined. */
    private fun initLiveMediaPipe() {
        livePoseLandmarker?.close()
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setMinPosePresenceConfidence(0.3f)
                .setResultListener { result, _ ->
                    // No landmarks → clear overlay immediately (no ghost when person leaves)
                    if (result.landmarks().isEmpty()) {
                        lastGoodLiveDetection = null
                        return@setResultListener
                    }

                    val landmarks = result.landmarks()[0]
                    val keypoints = COCO_TO_MP.map { mpIdx ->
                        val lm = landmarks.getOrNull(mpIdx)
                        if (lm != null) Keypoint(lm.x(), lm.y(), lm.visibility().orElse(0f))
                        else            Keypoint(0f, 0f, 0f)
                    }

                    // Gate 1: require at least 5 high-confidence keypoints (kills phantom detections)
                    val highConfCount = keypoints.count { it.confidence > 0.5f }
                    if (highConfCount < 5) {
                        lastGoodLiveDetection = null
                        return@setResultListener
                    }

                    // Gate 2: average confidence of visible keypoints must be solid
                    val visibleKps = keypoints.filter { it.confidence > 0.3f }
                    val avgVisibleConf = visibleKps.map { it.confidence }.average().toFloat()
                    if (avgVisibleConf < 0.5f) {
                        lastGoodLiveDetection = null
                        return@setResultListener
                    }

                    // Bounding box from visible keypoints only
                    val bbox = RectF(
                        visibleKps.minOf { it.x }, visibleKps.minOf { it.y },
                        visibleKps.maxOf { it.x }, visibleKps.maxOf { it.y }
                    )

                    val avgConf = keypoints.map { it.confidence }.average().toFloat()
                    val detection = Detection("batsman", avgConf, bbox, keypoints)

                    // Live Kalman filters: higher measurementNoise + deadband → smooth & still when not moving
                    if (liveKalmanFilters.size != keypoints.size) {
                        liveKalmanFilters.clear()
                        repeat(keypoints.size) {
                            liveKalmanFilters.add(
                                KeypointKalmanFilter(
                                    processNoise      = 0.00001f,  // slow to build velocity
                                    measurementNoise  = 0.005f,    // 5× more smoothing vs default
                                    velocityDeadband  = 0.001f     // zero drift < 0.1% frame width
                                )
                            )
                        }
                    }
                    val smoothedKps = keypoints.mapIndexed { idx, kp ->
                        val (sx, sy) = liveKalmanFilters[idx].update(kp.x, kp.y, kp.confidence)
                        Keypoint(sx, sy, kp.confidence)
                    }
                    lastGoodLiveDetection = detection.copy(keypoints = smoothedKps)

                    if (sessionStartTime == 0L) sessionStartTime = System.currentTimeMillis()
                    val shot = liveShotDetector.feed(smoothedKps)
                    if (shot.isNotEmpty()) currentShotType = shot
                    // Show a Toast the moment a new shot is finalized
                    val newCount = liveShotDetector.shotEventCount
                    if (newCount > lastSeenShotEventCount && shot.isNotEmpty()) {
                        lastSeenShotEventCount = newCount
                        val shotLabel = shot
                        runOnUiThread {
                            Toast.makeText(this@PracticeSession, "Shot: $shotLabel", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
                .setErrorListener { e ->
                    Log.e(TAG, "Live MediaPipe error: ${e.message}")
                }
                .build()
            livePoseLandmarker = PoseLandmarker.createFromOptions(this, options)
            Log.d(TAG, "Live LIVE_STREAM PoseLandmarker initialised")
        } catch (e: Exception) {
            Log.e(TAG, "initLiveMediaPipe failed", e)
        }
    }

    private fun enterLiveMode() {
        if (videoView.isPlaying) videoView.stopPlayback()
        isVideoPlaying = false
        isLiveMode = true
        resetAnalysisState()
        syncHandler.removeCallbacks(syncRunnable)
        liveDetections.clear()

        videoContainer.visibility = View.GONE
        cameraContainer.visibility = View.VISIBLE

        // Overlay image size is set per-frame in processFrame() once we
        // know the actual rotated bitmap dimensions — that's the only way
        // to match the exact aspect ratio MediaPipe sees.

        // Spin up the LIVE_STREAM landmarker on a background thread
        cameraExecutor.execute { initLiveMediaPipe() }

        liveHandler.removeCallbacks(liveSyncRunnable)
        liveHandler.post(liveSyncRunnable)
        Toast.makeText(this, "Live Mode Active", Toast.LENGTH_SHORT).show()
    }

    @androidx.camera.core.ExperimentalGetImage
    private fun processFrame(proxy: ImageProxy) {
        if (!isLiveMode) { proxy.close(); return }

        val landmarker = livePoseLandmarker
        if (landmarker == null) { proxy.close(); return }  // still warming up

        // Strictly-increasing timestamp required by LIVE_STREAM mode
        val timestampMs = SystemClock.uptimeMillis()

        val bitmap = proxy.toBitmap()
        var rotated = bitmap
        if (proxy.imageInfo.rotationDegrees != 0) {
            rotated = rotateBitmap(bitmap, proxy.imageInfo.rotationDegrees.toFloat())
            if (rotated !== bitmap) bitmap.recycle()
        }
        proxy.close()

        // Tell the overlay the EXACT dimensions MediaPipe will see so its
        // scale+offset math matches the actual frame aspect ratio, not a square.
        val fw = rotated.width
        val fh = rotated.height
        if (fw != liveFrameW || fh != liveFrameH) {
            liveFrameW = fw
            liveFrameH = fh
            runOnUiThread { cameraOverlay.setImageSize(fw, fh) }
        }

        try {
            // Convert to ARGB_8888 if needed (MediaPipe requirement)
            val argbBmp = if (rotated.config == Bitmap.Config.ARGB_8888) rotated
            else rotated.copy(Bitmap.Config.ARGB_8888, false)

            val mpImage = BitmapImageBuilder(argbBmp).build()

            // detectAsync is NON-BLOCKING — returns immediately, result fires in callback
            landmarker.detectAsync(mpImage, timestampMs)

            // Recycle conversion copy; original 'rotated' recycled below
            if (argbBmp !== rotated) argbBmp.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "detectAsync error", e)
        } finally {
            if (rotated !== bitmap) rotated.recycle() else bitmap.recycle()
        }
    }


    private fun classifyVideoShot(detections: TreeMap<Long, Detection>): String {
        if (detections.isEmpty()) return ""

        val keys  = detections.keys.toList()
        val total = keys.size

        val (startIdx, endIdx) = if (total <= 4) {
            0 to total
        } else {
            val skip = (total * 0.20).toInt().coerceAtLeast(1)
            skip to (total - skip)
        }

        fun wristPos(det: Detection): Pair<Float, Float>? {
            val kps = det.keypoints
            val lw = kps.getOrNull(9)?.takeIf  { it.confidence > 0.25f }
            val rw = kps.getOrNull(10)?.takeIf { it.confidence > 0.25f }
            return when {
                lw != null && rw != null -> (lw.x + rw.x) / 2f to (lw.y + rw.y) / 2f
                lw != null               -> lw.x to lw.y
                rw != null               -> rw.x to rw.y
                else                     -> null
            }
        }

        val labels     = mutableListOf<String>()
        var prevWrist: Pair<Float, Float>? = null

        for (i in startIdx until endIdx) {
            val det      = detections[keys[i]] ?: continue
            val curWrist = wristPos(det)
            val velocity = if (prevWrist != null && curWrist != null) {
                val dx = curWrist.first  - prevWrist!!.first
                val dy = curWrist.second - prevWrist!!.second
                sqrt(dx * dx + dy * dy)
            } else 0f
            if (curWrist != null) prevWrist = curWrist

            val label = ShotClassifier.classify(det.keypoints, velocity)
            Log.d("SHOT_FRAME", "frame[$i/$total] ts=${keys[i]} vel=${"%.4f".format(velocity)} → ${label.ifEmpty { "—" }}")
            if (label.isNotEmpty()) labels += label
        }

        // Fallback: if velocity guard blocked everything, classify pose geometry alone
        if (labels.isEmpty()) {
            Log.d("SHOT_DETECTOR", "All frames below velocity guard — falling back to geometry-only classify")
            for (i in startIdx until endIdx) {
                val det   = detections[keys[i]] ?: continue
                val label = ShotClassifier.classify(det.keypoints, wristVelocity = 1f)
                if (label.isNotEmpty()) labels += label
            }
        }

        if (labels.isEmpty()) return ""

        val nonDef = labels.filter { it != "Defensive" }
        val result = if (nonDef.isNotEmpty()) {
            nonDef.groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
        } else {
            "Defensive"
        }

        Log.d("SHOT_DETECTOR",
            "classifyVideoShot: frames=$total  middle=${endIdx - startIdx}  labels=$labels  result=$result")
        return result
    }


    /**
     * Parses the Roboflow Workflow API response for the updated workflow that uses:
     *   - sam3 (batsman segmentation)
     *   - yolov8x-pose-1280 (keypoint detection)
     *
     * New format: output_predictions_v2 is a JSONObject with shape:
     *   { "image": { "width": W, "height": H }, "predictions": [ { ...pred with keypoints... } ] }
     *
     * Falls back to the old JSONArray format if the new format is not found,
     * so this function is safe to use during any transition period.
     *
     * Coordinates come in pixel space relative to the image dimensions reported
     * in the response (coordinates_system = "own"), so we normalize using those
     * instead of the encoded frame size passed as w/h.
     */
    private fun parseDetectionFromJson(json: String, w: Int, h: Int): Detection? {
        try {
            val trimmed = json.trim()
            val outputs = if (trimmed.startsWith("[")) {
                JSONArray(trimmed)
            } else {
                val root = JSONObject(trimmed)
                root.optJSONArray("outputs") ?: JSONArray()
            }

            Log.d(TAG, "Parsing JSON: ${outputs.length()} outputs")

            for (i in 0 until outputs.length()) {
                val output = outputs.getJSONObject(i)

                // ── NEW WORKFLOW FORMAT ──────────────────────────────────────────
                // Field name: keypoint_detection_model_output
                val keypointOutput = output.optJSONObject("keypoint_detection_model_output")
                if (keypointOutput != null) {
                    val det = parseFromNewFormat(keypointOutput, w, h)
                    if (det != null) return det
                    continue
                }

                // ── PREVIOUS FORMAT ───────────────────────────────────────────────
                // output_predictions_v2 is a JSONObject:
                //   { "image": { "width": W, "height": H }, "predictions": [...] }
                val predsV2Obj = output.optJSONObject("output_predictions_v2")
                if (predsV2Obj != null) {
                    val det = parseFromNewFormat(predsV2Obj, w, h)
                    if (det != null) return det
                    // If it returned null (empty predictions), continue to next output
                    continue
                }

                // ── OLD FORMAT (fallback) ─────────────────────────────────────────
                // output_predictions_v2 was a JSONArray of objects, each containing
                //   { "predictions": { "predictions": [...] } }
                val predsV2Arr = output.optJSONArray("output_predictions_v2")
                if (predsV2Arr != null) {
                    val det = parseFromOldFormat(predsV2Arr, w, h)
                    if (det != null) return det
                }

                Log.d(TAG, "Output $i: No recognized keypoint fields found")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Parse JSON error: ${e.message}", e)
        }
        return null
    }

    /**
     * Flexible parser for Workflow outputs. Handles both:
     * 1. { "image": ..., "predictions": [...] }  <-- Direct (new workflow style)
     * 2. { "predictions": { "image": ..., "predictions": [...] } } <-- Wrapped (old style)
     */
    private fun parseFromNewFormat(predsObj: JSONObject, fallbackW: Int, fallbackH: Int): Detection? {
        // Robustly find the object containing 'image' and 'predictions'
        val data = if (predsObj.has("image") && predsObj.has("predictions")) {
            predsObj
        } else {
            predsObj.optJSONObject("predictions")
        }

        if (data == null) {
            Log.d(TAG, "PARSE: data (image/predictions) missing in output object")
            return null
        }

        // Use image dimensions from the response for accurate normalization
        val imgObj = data.optJSONObject("image")
        val imgW = imgObj?.optInt("width", fallbackW) ?: fallbackW
        val imgH = imgObj?.optInt("height", fallbackH) ?: fallbackH

        val predArray = data.optJSONArray("predictions") ?: run {
            Log.d(TAG, "PARSE: predictions array missing inside data object")
            return null
        }

        if (predArray.length() == 0) {
            Log.d(TAG, "PARSE: predictions array is empty")
            return null
        }

        // Pick the highest-confidence prediction (most likely the main batsman)
        var bestPred: JSONObject? = null
        var bestConf = -1f
        for (j in 0 until predArray.length()) {
            val candidate = predArray.getJSONObject(j)
            val conf = candidate.optDouble("confidence", 0.0).toFloat()
            if (conf > bestConf) {
                bestConf = conf
                bestPred = candidate
            }
        }

        val pred = bestPred ?: return null
        val confidence = bestConf
        Log.d(TAG, "PARSE: best prediction confidence=$confidence, imgSize=${imgW}x${imgH}")

        // Bounding box (centre-format -> normalized 0-1)
        val cx     = pred.optDouble("x", 0.0).toFloat()
        val cy     = pred.optDouble("y", 0.0).toFloat()
        val bw     = pred.optDouble("width", 0.0).toFloat()
        val bh     = pred.optDouble("height", 0.0).toFloat()
        val left   = ((cx - bw / 2f) / imgW).coerceIn(0f, 1f)
        val top    = ((cy - bh / 2f) / imgH).coerceIn(0f, 1f)
        val right  = ((cx + bw / 2f) / imgW).coerceIn(0f, 1f)
        val bottom = ((cy + bh / 2f) / imgH).coerceIn(0f, 1f)

        // Keypoints
        val keypointsJson = pred.optJSONArray("keypoints") ?: run {
            Log.d(TAG, "PARSE: keypoints array missing from prediction")
            return null
        }

        val kpList = mutableListOf<Keypoint>()
        for (k in 0 until keypointsJson.length()) {
            val kp = keypointsJson.getJSONObject(k)
            val kx    = kp.optDouble("x", 0.0).toFloat() / imgW
            val ky    = kp.optDouble("y", 0.0).toFloat() / imgH
            val kconf = kp.optDouble("confidence", 0.0).toFloat()
            kpList.add(Keypoint(kx, ky, kconf))
        }

        Log.d(TAG, "PARSE: parsed ${kpList.size} keypoints")

        return Detection(
            label      = "batsman",
            confidence = confidence,
            bbox       = RectF(left, top, right, bottom),
            keypoints  = kpList
        )
    }

    /**
     * OLD FORMAT parser (kept for backward compatibility) — output_predictions_v2
     * was a JSONArray where each element looked like:
     * { "predictions": { "predictions": [ { ...pred... } ] } }
     */
    private fun parseFromOldFormat(predsV2Arr: JSONArray, w: Int, h: Int): Detection? {
        Log.d(TAG, "OLD FORMAT: array length=${predsV2Arr.length()}")

        for (j in 0 until predsV2Arr.length()) {
            val item = predsV2Arr.optJSONObject(j) ?: continue
            val predictions = item.optJSONObject("predictions") ?: continue
            val predArray   = predictions.optJSONArray("predictions") ?: continue

            if (predArray.length() == 0) continue

            val pred       = predArray.getJSONObject(0)
            val confidence = pred.getDouble("confidence").toFloat()
            Log.d(TAG, "OLD FORMAT: confidence=$confidence")

            val parentOrigin = pred.optJSONObject("parent_origin") ?: JSONObject()
            val offsetX = parentOrigin.optInt("offset_x", 0)
            val offsetY = parentOrigin.optInt("offset_y", 0)

            val x      = pred.getDouble("x").toFloat()
            val y      = pred.getDouble("y").toFloat()
            val width  = pred.getDouble("width").toFloat()
            val height = pred.getDouble("height").toFloat()

            val left   = (x - width  / 2 + offsetX) / w
            val top    = (y - height / 2 + offsetY) / h
            val right  = (x + width  / 2 + offsetX) / w
            val bottom = (y + height / 2 + offsetY) / h

            val keypointsJson = pred.optJSONArray("keypoints") ?: continue
            val kpList = mutableListOf<Keypoint>()

            for (k in 0 until keypointsJson.length()) {
                val kp    = keypointsJson.getJSONObject(k)
                val kx    = kp.getDouble("x").toFloat() + offsetX
                val ky    = kp.getDouble("y").toFloat() + offsetY
                val kconf = kp.getDouble("confidence").toFloat()
                kpList.add(Keypoint(kx / w, ky / h, kconf))
            }

            Log.d(TAG, "OLD FORMAT: parsed ${kpList.size} keypoints")

            return Detection(
                label      = "batsman",
                confidence = confidence,
                bbox       = RectF(left, top, right, bottom),
                keypoints  = kpList
            )
        }
        return null
    }

    private fun parseRoboflowResponse(json: String, w: Int, h: Int, captureTimeMs: Long) {
        val raw = parseDetectionFromJson(json, w, h)

        if (raw == null || raw.keypoints.isEmpty()) return

        // Initialize Kalman filters once, reuse every frame
        if (liveKalmanFilters.size != raw.keypoints.size) {
            liveKalmanFilters.clear()
            repeat(raw.keypoints.size) { liveKalmanFilters.add(KeypointKalmanFilter()) }
        }

        // Apply Kalman filter to each keypoint
        val smoothedKps = raw.keypoints.mapIndexed { idx, kp ->
            val (sx, sy) = liveKalmanFilters[idx].update(kp.x, kp.y, kp.confidence)
            Keypoint(sx, sy, kp.confidence)
        }

        val smoothed = raw.copy(keypoints = smoothedKps)

        // FIX: Write directly to lastGoodLiveDetection instead of latestLiveDetection.
        // The render loop just reads lastGoodLiveDetection every 16ms without ever
        // nulling it, so there is no frame where it has "nothing to show" → no blink.
        lastGoodLiveDetection = smoothed

        val shot = liveShotDetector.feed(smoothedKps)
        if (shot.isNotEmpty()) currentShotType = shot
        // Show a Toast the moment a new shot is finalized
        val newCount = liveShotDetector.shotEventCount
        if (newCount > lastSeenShotEventCount && shot.isNotEmpty()) {
            lastSeenShotEventCount = newCount
            val shotLabel = shot
            runOnUiThread {
                Toast.makeText(this@PracticeSession, "Shot: $shotLabel", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun getCurrentOverlay() = if (videoContainer.visibility == View.VISIBLE) videoOverlay else cameraOverlay

    private fun fetchDetectionSyncBase64(base64: String): Detection? {
        val jsonPayload = """
        {
            "api_key": "$ROBOFLOW_API_KEY",
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": "$base64"
                }
            }
        }
        """.trimIndent()

        val request = Request.Builder()
            .url(INFERENCE_URL)
            .post(jsonPayload.toRequestBody("application/json".toMediaType()))
            .build()

        try {
            val response = client.newCall(request).execute()
            val body = response.body?.string()

            if (response.isSuccessful && !body.isNullOrEmpty()) {
                Log.d(TAG, "Roboflow Sync Response: $body")
                return parseDetectionFromJson(body, videoFrameW, videoFrameH)
            } else {
                Log.w(TAG, "API Error: ${response.code} - $body")
            }
            response.close()
        } catch (e: Exception) {
            Log.e(TAG, "Sync Fetch Error", e)
        }
        return null
    }

    private fun interpolateDetections(a: Detection, b: Detection, alpha: Float): Detection {
        fun lerp(x: Float, y: Float) = x + (y - x) * alpha
        val bbox = android.graphics.RectF(
            lerp(a.bbox.left,   b.bbox.left),
            lerp(a.bbox.top,    b.bbox.top),
            lerp(a.bbox.right,  b.bbox.right),
            lerp(a.bbox.bottom, b.bbox.bottom)
        )
        val keypoints = if (a.keypoints.size == b.keypoints.size) {
            a.keypoints.zip(b.keypoints).map { (k1, k2) ->
                Keypoint(lerp(k1.x, k2.x), lerp(k1.y, k2.y), lerp(k1.confidence, k2.confidence))
            }
        } else a.keypoints
        return Detection(a.label, lerp(a.confidence, b.confidence), bbox, keypoints)
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 60, baos)
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }

    private fun bitmapToBase64AndRecycle(bitmap: Bitmap): String {
        val resized = Bitmap.createScaledBitmap(bitmap, videoFrameW, videoFrameH, true)
        if (resized != bitmap) bitmap.recycle()

        val baos = ByteArrayOutputStream()
        resized.compress(Bitmap.CompressFormat.JPEG, 40, baos) // reduced from 60 → smaller payload, faster upload
        val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

        resized.recycle()
        return b64
    }

    // Returns the midpoint of both wrists (or one if only one is visible) from a detection.
    // Used in Pass 1 of preProcessVideo to measure wrist velocity between coarse frames.
    private fun getWristCenter(detection: Detection): Pair<Float, Float>? {
        val kps = detection.keypoints
        val lw = kps.getOrNull(9)?.takeIf  { it.confidence > 0.25f }
        val rw = kps.getOrNull(10)?.takeIf { it.confidence > 0.25f }
        return when {
            lw != null && rw != null -> (lw.x + rw.x) / 2f to (lw.y + rw.y) / 2f
            lw != null               -> lw.x to lw.y
            rw != null               -> rw.x to rw.y
            else                     -> null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        return if (degrees != 0f) Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, Matrix().apply { postRotate(degrees) }, true) else bitmap
    }

    private fun analyzePose(detection: Detection) {
        val kp = detection.keypoints

        fun getPt(idx: Int): Pair<Float, Float>? {
            if (idx >= kp.size) return null
            val k = kp[idx]
            if (k.confidence < 0.3f) return null
            return k.x to k.y
        }

        val nose = getPt(0)
        if (nose != null) {
            headHistory.addLast(nose)
            if (headHistory.size > HISTORY_SIZE) headHistory.removeFirst()

            if (headHistory.size > 2) {
                val avgX = headHistory.map { it.first }.average()
                val avgY = headHistory.map { it.second }.average()
                val distSq = headHistory.sumOf { (it.first - avgX).pow(2) + (it.second - avgY).pow(2) }
                val variance = distSq / headHistory.size

                val maxVariance = 0.0015f
                val headTolerance = 0.0002f
                val score = if (variance.toFloat() <= headTolerance) 100f
                else (100f - ((variance.toFloat() - headTolerance) / (maxVariance - headTolerance)) * 100f).coerceIn(0f, 100f)

                headBadCount = if (score < 100f) minOf(headBadCount + 1, MAX_PERSISTENCE) else maxOf(headBadCount - 2, 0)
                val finalScore = if (headBadCount >= PERSISTENCE_FRAMES) score else 100f

                headStabilityScore = (headStabilityScore * 0.8f) + (finalScore * 0.2f)

                headStatus = "${headStabilityScore.toInt()}%"
            }
        }

        val ls = getPt(5)
        val rs = getPt(6)
        if (ls != null && rs != null) {
            val dy = (rs.second - ls.second)
            val dx = (rs.first - ls.first)
            val angle = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()

            val maxAngle = 20f
            val shoulderTolerance = 4f
            val targetAngle = abs(angle)
            val score = if (targetAngle <= shoulderTolerance) 100f
            else (100f - ((targetAngle - shoulderTolerance) / (maxAngle - shoulderTolerance)) * 100f).coerceIn(0f, 100f)

            shoulderBadCount = if (score < 100f) minOf(shoulderBadCount + 1, MAX_PERSISTENCE) else maxOf(shoulderBadCount - 2, 0)
            val finalScore = if (shoulderBadCount >= PERSISTENCE_FRAMES) score else 100f

            shoulderScore = (shoulderScore * 0.8f) + (finalScore * 0.2f)

            shoulderStatus = "${shoulderScore.toInt()}%"
        }

        val lHip = getPt(11)
        val rHip = getPt(12)
        val lAnk = getPt(15)
        val rAnk = getPt(16)

        if (lHip != null && rHip != null && lAnk != null && rAnk != null) {
            val hipCenter = (lHip.first + rHip.first) / 2
            val ankCenter = (lAnk.first + rAnk.first) / 2

            val diff = abs(hipCenter - ankCenter)
            val maxDiff = 0.1f
            val weightTolerance = 0.02f
            val score = if (diff <= weightTolerance) 100f
            else (100f - ((diff - weightTolerance) / (maxDiff - weightTolerance)) * 100f).coerceIn(0f, 100f)

            weightBadCount = if (score < 100f) minOf(weightBadCount + 1, MAX_PERSISTENCE) else maxOf(weightBadCount - 2, 0)
            val finalScore = if (weightBadCount >= PERSISTENCE_FRAMES) score else 100f

            weightScore = (weightScore * 0.8f) + (finalScore * 0.2f)

            weightShiftText = "${weightScore.toInt()}%"
        }

        if (lAnk != null && rAnk != null) {
            val currentFeet = (lAnk.first + rAnk.first) / 2 to (lAnk.second + rAnk.second) / 2

            if (lastFootPosition != null) {
                val dist = sqrt((currentFeet.first - lastFootPosition!!.first).pow(2) + (currentFeet.second - lastFootPosition!!.second).pow(2))
                val maxDist = 0.05f
                val footTolerance = 0.005f
                val score = if (dist.toFloat() <= footTolerance) 100f
                else (100f - ((dist.toFloat() - footTolerance) / (maxDist - footTolerance)) * 100f).coerceIn(0f, 100f)

                footworkBadCount = if (score < 100f) minOf(footworkBadCount + 1, MAX_PERSISTENCE) else maxOf(footworkBadCount - 2, 0)
                val finalScore = if (footworkBadCount >= PERSISTENCE_FRAMES) score else 100f

                footworkScore = (footworkScore * 0.7f) + (finalScore * 0.3f)
                footworkStatus = "${footworkScore.toInt()}%"
            }
            lastFootPosition = currentFeet
        }
    }

    private fun resetAnalysisState() {
        liveShotDetector.reset()
        finalizedVideoShots.clear()
        currentShotType = ""
        headHistory.clear()
        headStabilityScore = 100f
        shoulderScore = 100f
        weightScore = 100f
        footworkScore = 100f
        weightShiftText = "100%"
        shoulderStatus = "100%"
        headStatus = "100%"
        footworkStatus = "100%"
        lastFootPosition = null

        headBadCount = 0
        shoulderBadCount = 0
        weightBadCount = 0
        footworkBadCount = 0

        lastGoodLiveDetection = null
        liveDetections.clear()
        lastSeenShotEventCount = 0

        runOnUiThread {
            headTv.text = "--"
            shouldersTv.text = "--"
            weightTv.text = " --"
            feetTv.text = "--"
        }
    }

    private fun saveSession() {
        if (sessionStartTime == 0L) sessionStartTime = System.currentTimeMillis()

        if (isLiveMode) liveShotDetector.flush()

        val durationSeconds = ((System.currentTimeMillis() - sessionStartTime) / 1000).toInt().coerceAtLeast(30)
        val summary = "Head: $headStatus | Shoulders: $shoulderStatus | Weight: $weightShiftText | Feet: $footworkStatus"

        val shotSummaryText: String
        val totalShots: Int
        if (isLiveMode) {
            val shotSummary = liveShotDetector.getShotSummary()
            totalShots = liveShotDetector.shotEventCount
            shotSummaryText = if (shotSummary.isEmpty()) "No shots detected." else
                shotSummary.entries.sortedByDescending { it.value }
                    .joinToString("\n") { (label, count) -> "  • $label  ×$count" }
        } else {
            val label = finalizedVideoShots.values.lastOrNull()?.takeIf { it.isNotEmpty() }
                ?: currentShotType
            totalShots = if (label.isNotEmpty()) 1 else 0
            shotSummaryText = if (label.isNotEmpty()) "  • $label" else "No shot detected."
        }

        val message = buildString {
            append("Shots detected: $totalShots\n\n")
            append(shotSummaryText)
            append("\n\n─────────────────────\n")
            append("Pose Quality\n")
            append("  Head:      $headStatus\n")
            append("  Shoulders: $shoulderStatus\n")
            append("  Weight:    $weightShiftText\n")
            append("  Feet:      $footworkStatus")
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("Session Summary")
            .setMessage(message)
            .setPositiveButton("Save") { _, _ ->
                val session = com.fyp.nextshot.data.local.models.SessionEntity(
                    userId = userId,
                    drillType = "Pose Analysis",
                    durationSeconds = durationSeconds,
                    successRate = 1.0,
                    flawDetails = summary,
                    dateMillis = System.currentTimeMillis()
                )
                sessionViewModel.insert(session)
                Toast.makeText(this, "Session Saved!", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Discard", null)
            .show()
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21, android.graphics.ImageFormat.NV21, width, height, null
        )
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 85, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)!!
    }

    // 25002500 MediaPipe helpers 250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500250025002500

    /** Initialises the PoseLandmarker once; safe to call multiple times. */
    private fun initMediaPipe() {
        if (mediaPipeLandmarker != null) return
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()
            mediaPipeLandmarker = PoseLandmarker.createFromOptions(this, options)
            Log.d(TAG, "MediaPipe PoseLandmarker initialised OK")
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe init failed", e)
        }
    }

    /** Initialises the person ObjectDetector from EfficientDet-Lite0 (COCO). */
    private fun initObjectDetector() {
        if (mediaPipeDetector != null) return
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("efficientdet_lite0.tflite")
                .build()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setMaxResults(5)
                .setScoreThreshold(0.4f)
                .build()
            mediaPipeDetector = ObjectDetector.createFromOptions(this, options)
            Log.d(TAG, "MediaPipe ObjectDetector initialised OK")
        } catch (e: Exception) {
            Log.e(TAG, "ObjectDetector init failed", e)
        }
    }

    /**
     * Two-step on-device pipeline — mirrors the original Roboflow workflow:
     *   Step 1 — ObjectDetector (EfficientDet-Lite0): detect the batsman / person.
     *   Step 2 — PoseLandmarker (BlazePose-Lite): estimate 33 body landmarks.
     *
     * Keypoints are remapped to COCO-17 ordering via COCO_TO_MP.
     * Bounding box comes from ObjectDetector (normalised 0-1); falls back to
     * keypoint envelope if no person is detected.
     */
    private fun runMediaPipePose(bitmap: Bitmap): Detection? {
        val landmarker = mediaPipeLandmarker ?: return null
        var argbBmp: Bitmap? = null
        return try {
            val imgW = bitmap.width.toFloat().coerceAtLeast(1f)
            val imgH = bitmap.height.toFloat().coerceAtLeast(1f)
            // MediaPipe requires ARGB_8888 — convert if needed (MediaMetadataRetriever may return RGB_565)
            argbBmp = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
            val mpImage = BitmapImageBuilder(argbBmp).build()

            // ── Step 1: Batsman / person detection (replaces Roboflow SAM3) ─────
            var batsmanBbox: RectF? = null
            val detector = mediaPipeDetector
            if (detector != null) {
                val detResult = detector.detect(mpImage)
                val personDet = detResult.detections()
                    .filter { d -> d.categories().any { it.categoryName() == "person" } }
                    .maxByOrNull { d ->
                        d.categories().firstOrNull { it.categoryName() == "person" }?.score() ?: 0f
                    }
                if (personDet != null) {
                    val r = personDet.boundingBox()
                    batsmanBbox = RectF(
                        r.left  / imgW, r.top    / imgH,
                        r.right / imgW, r.bottom / imgH
                    )
                    Log.d(TAG, "Batsman detected: bbox=$batsmanBbox")
                }
            }

            // ── Step 2: Pose keypoints (replaces Roboflow YOLOv8x-pose) ─────────
            val poseResult = landmarker.detect(mpImage)
            if (poseResult.landmarks().isEmpty()) return null

            val landmarks = poseResult.landmarks()[0]
            val keypoints = COCO_TO_MP.map { mpIdx ->
                val lm = landmarks.getOrNull(mpIdx)
                if (lm != null) Keypoint(lm.x(), lm.y(), lm.visibility().orElse(0f))
                else            Keypoint(0f, 0f, 0f)
            }

            // ObjectDetector bbox if available, else keypoint envelope
            val bbox = batsmanBbox ?: run {
                val visible = keypoints.filter { it.confidence > 0.3f }
                if (visible.isNotEmpty()) RectF(
                    visible.minOf { it.x }, visible.minOf { it.y },
                    visible.maxOf { it.x }, visible.maxOf { it.y }
                ) else RectF(0f, 0f, 1f, 1f)
            }

            val avgConf = keypoints.map { it.confidence }.average().toFloat()
            Detection(label = "batsman", confidence = avgConf, bbox = bbox, keypoints = keypoints)
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe inference error", e)
            null
        } finally {
            // Recycle the ARGB conversion copy if we made one
            if (argbBmp !== null && argbBmp !== bitmap) argbBmp!!.recycle()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPipeLandmarker?.close()
        mediaPipeLandmarker = null
        mediaPipeDetector?.close()
        mediaPipeDetector = null
        livePoseLandmarker?.close()
        livePoseLandmarker = null
        cameraExecutor.shutdown()
        syncHandler.removeCallbacksAndMessages(null)
        liveHandler.removeCallbacksAndMessages(null)
    }
}