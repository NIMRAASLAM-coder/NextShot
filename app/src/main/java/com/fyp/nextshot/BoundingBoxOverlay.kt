package com.fyp.nextshot

import android.content.Context
import android.util.Log
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

data class Detection(
    val label: String,
    val confidence: Float,
    val bbox: RectF,
    val keypoints: List<Keypoint> = emptyList()  // uses the shared Keypoint class
)

class BoundingBoxOverlay : View {

    constructor(context: Context) : super(context)
    constructor(context: Context, attrs: AttributeSet?) : super(context, attrs)
    constructor(context: Context, attrs: AttributeSet?, defStyleAttr: Int) : super(context, attrs, defStyleAttr)

    var imageWidth = 0f
        private set
    var imageHeight = 0f
        private set

    fun setImageSize(w: Int, h: Int) {
        imageWidth = w.toFloat()
        imageHeight = h.toFloat()
        Log.d("BBOX_DEBUG", "setImageSize called: ${w}x${h}, view is ${this.width}x${this.height}")
        invalidate()
    }

    private var detections: List<Detection> = emptyList()
    private var shotTypeLabel: String = ""

    fun setShotType(label: String) {
        shotTypeLabel = label
        invalidate()
    }

    fun setDetections(newDetections: List<Detection>) {
        detections = newDetections
        invalidate()
    }
    /** Sets detections + shot type together and redraws exactly once. */
    fun update(newDetections: List<Detection>, shotType: String) {
        detections = newDetections
        shotTypeLabel = shotType
        invalidate()
    }
    fun clear() {
        detections = emptyList()
        shotTypeLabel = ""
        invalidate()
    }

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.CYAN
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 30f
        isAntiAlias = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    private val linePaint = Paint().apply {
        color = Color.CYAN
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val pointPaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 1f
        style = Paint.Style.FILL
    }

    private val shotTypeBgPaint = Paint().apply {
        color = Color.argb(210, 255, 140, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val shotTypeTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        isFakeBoldText = true
        isAntiAlias = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    private val skeleton = listOf(
        0 to 1, 0 to 2,
        1 to 3, 2 to 4,
        5 to 6,
        5 to 7, 7 to 9,
        6 to 8, 8 to 10,
        5 to 11, 6 to 12,
        11 to 12,
        11 to 13, 13 to 15,
        12 to 14, 14 to 16
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        Log.d("BBOX_DEBUG", "onDraw: detections=${detections.size}, imageWidth=$imageWidth, imageHeight=$imageHeight, viewW=$width, viewH=$height")

        if (detections.isEmpty() || imageWidth == 0f || imageHeight == 0f) {
            Log.d("BBOX_DEBUG", "onDraw EARLY RETURN: detections=${detections.size} imgW=$imageWidth imgH=$imageHeight")
            return
        }

        val scale = min(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        Log.d("BBOX_DEBUG", "onDraw drawing: scale=$scale, offsetX=$offsetX, offsetY=$offsetY")

        for (det in detections) {
            val left   = det.bbox.left   * imageWidth  * scale + offsetX
            val top    = det.bbox.top    * imageHeight * scale + offsetY
            val right  = det.bbox.right  * imageWidth  * scale + offsetX
            val bottom = det.bbox.bottom * imageHeight * scale + offsetY

            Log.d("BBOX_DEBUG", "Drawing rect: left=$left top=$top right=$right bottom=$bottom (bbox=${det.bbox})")

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val confText = "${det.label.uppercase()} ${(det.confidence * 100).toInt()}%"
            canvas.drawText(confText, left + 20, top + 60, textPaint)

            // Shot type badge — drawn above the bounding box
            if (shotTypeLabel.isNotEmpty()) {
                val padding = 14f
                val textW = shotTypeTextPaint.measureText(shotTypeLabel)
                val badgeLeft   = left
                val badgeRight  = left + textW + padding * 2
                val badgeBottom = (top - 6f).coerceAtLeast(offsetY + 52f)
                val badgeTop    = (badgeBottom - 52f).coerceAtLeast(offsetY)
                val badgeRect   = RectF(badgeLeft, badgeTop, badgeRight, badgeBottom)
                canvas.drawRoundRect(badgeRect, 12f, 12f, shotTypeBgPaint)
                canvas.drawText(shotTypeLabel, badgeLeft + padding, badgeBottom - 12f, shotTypeTextPaint)
            }

            // Draw keypoints
            det.keypoints.forEach { kp ->
                if (kp.confidence > 0.3f) {
                    val cx = kp.x * imageWidth * scale + offsetX
                    val cy = kp.y * imageHeight * scale + offsetY
                    canvas.drawCircle(cx, cy, 6f, pointPaint)
                }
            }

            // Draw skeleton
            skeleton.forEach { (i, j) ->
                val kp1 = det.keypoints.getOrNull(i) ?: return@forEach
                val kp2 = det.keypoints.getOrNull(j) ?: return@forEach
                if (kp1.confidence > 0.3f && kp2.confidence > 0.3f) {
                    val x1 = kp1.x * imageWidth * scale + offsetX
                    val y1 = kp1.y * imageHeight * scale + offsetY
                    val x2 = kp2.x * imageWidth * scale + offsetX
                    val y2 = kp2.y * imageHeight * scale + offsetY
                    canvas.drawLine(x1, y1, x2, y2, linePaint)
                }
            }
        }
    }
}