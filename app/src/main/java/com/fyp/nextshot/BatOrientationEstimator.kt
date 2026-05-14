package com.fyp.nextshot

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

// ═══════════════════════════════════════════════════════════════════════════════
// BatOrientationEstimator.kt
//
// Infers bat orientation and swing phase purely from COCO-17 keypoints.
//
// Strategy (two methods, best-available):
//   Method 1 — Two-wrist line (L.Wrist → R.Wrist).
//              Both hands grip the bat, so the wrist-to-wrist vector is the
//              best available approximation of bat angle.  Requires both wrists
//              to be at least 5% of frame-width apart (avoids near-zero vectors
//              at the end of a shot when wrists converge).
//   Method 2 — Wrist → Elbow vector (fallback when wrists overlap).
//              Forearm and bat are roughly co-linear for most cricket shots
//              viewed from the side.
//
// Accuracy note:
//   Side-view footage: ~80 % orientation accuracy.
//   Front-on / mixed view: less reliable — wrists overlap frequently.
//   When MediaPipe confidence drops (motion blur, fast swing) UNKNOWN is returned
//   so callers can simply skip that frame rather than act on bad data.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Data types ────────────────────────────────────────────────────────────────

enum class BatOrientation {
    VERTICAL,           // straight bat: backlift, defensive push, drives
    DIAGONAL_UPLEFT,    // bat going up-left: backlift phase, off-drive follow-through
    DIAGONAL_UPRIGHT,   // bat going up-right: cut shot, leg-side follow-through
    HORIZONTAL,         // cross bat: sweep, pull, hook
    UNKNOWN             // insufficient confidence or geometry
}

enum class SwingPhase {
    BACKLIFT,           // bat moving upward/backward before delivery
    DOWNSWING,          // bat accelerating downward toward ball
    IMPACT_ZONE,        // bat roughly horizontal — near-contact moment
    FOLLOW_THROUGH,     // bat continuing upward after contact
    IDLE                // no meaningful angular movement this frame
}

data class BatAngleResult(
    val orientation: BatOrientation,
    val angleDegrees: Float,    // raw signed angle in [-180, 180]
    val confidence: Float,      // average keypoint confidence used
    val methodUsed: Int         // 1 = two-wrist, 2 = wrist-elbow
)

data class BatMovement(
    val current: BatAngleResult,
    val swingPhase: SwingPhase,
    val angularVelocity: Float, // degrees/frame, + = clockwise
    val description: String     // human-readable for HUD / logs
)

// ── Single-frame estimator ────────────────────────────────────────────────────

object BatOrientationEstimator {

    private const val CONF_THRESHOLD    = 0.30f
    private const val MIN_WRIST_DIST    = 0.05f  // normalised — ~5 % of frame width

    /**
     * Estimate bat orientation from one frame of keypoints.
     * Returns BatAngleResult with orientation = UNKNOWN when data is insufficient.
     */
    fun estimate(keypoints: List<Keypoint>): BatAngleResult {
        val lWrist = keypoints.getOrNull(9)?.takeIf  { it.confidence >= CONF_THRESHOLD }
        val rWrist = keypoints.getOrNull(10)?.takeIf { it.confidence >= CONF_THRESHOLD }
        val lElbow = keypoints.getOrNull(7)?.takeIf  { it.confidence >= CONF_THRESHOLD }
        val rElbow = keypoints.getOrNull(8)?.takeIf  { it.confidence >= CONF_THRESHOLD }

        // ── Method 1: Two-wrist line ──────────────────────────────────────────
        if (lWrist != null && rWrist != null) {
            val dx   = rWrist.x - lWrist.x
            val dy   = rWrist.y - lWrist.y   // Y increases downward
            val dist = sqrt(dx * dx + dy * dy)
            if (dist >= MIN_WRIST_DIST) {
                val angle   = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
                val avgConf = (lWrist.confidence + rWrist.confidence) / 2f
                return classifyAngle(angle, avgConf, methodUsed = 1)
            }
        }

        // ── Method 2: Dominant wrist → elbow fallback ────────────────────────
        val (wrist, elbow) = when {
            lWrist != null && lElbow != null -> lWrist to lElbow
            rWrist != null && rElbow != null -> rWrist to rElbow
            else -> return BatAngleResult(BatOrientation.UNKNOWN, 0f, 0f, 0)
        }

        val dx      = wrist.x - elbow.x
        val dy      = wrist.y - elbow.y
        val angle   = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
        val avgConf = (wrist.confidence + elbow.confidence) / 2f
        return classifyAngle(angle, avgConf, methodUsed = 2)
    }

    // Normalise the raw signed angle into an orientation bucket.
    // We work in the 0–180° half-plane because orientation is undirected
    // (a bat at 30° looks the same as one at 210°).
    private fun classifyAngle(
        angleDeg: Float,
        confidence: Float,
        methodUsed: Int
    ): BatAngleResult {
        // Map to [0, 180)
        val norm = ((angleDeg % 180f) + 180f) % 180f

        val orientation = when {
            norm < 20f || norm >= 160f  -> BatOrientation.HORIZONTAL
            norm in 70f..110f           -> BatOrientation.VERTICAL
            norm in 20f..70f            -> BatOrientation.DIAGONAL_UPLEFT
            else                        -> BatOrientation.DIAGONAL_UPRIGHT  // 110–160
        }

        return BatAngleResult(orientation, angleDeg, confidence, methodUsed)
    }
}

// ── Multi-frame tracker ───────────────────────────────────────────────────────

/**
 * BatMovementTracker — call update() once per frame to get both orientation
 * and swing phase, derived from the recent angle history.
 *
 * Thread-safety: NOT thread-safe. Call from a single thread (the render/
 * analysis thread).  Create one instance per session; call reset() on
 * enterLiveMode() / startVideoAnalysis().
 */
class BatMovementTracker(private val historySize: Int = 6) {

    // Ring buffer of raw angles (degrees) from recent frames
    private val angleHistory = ArrayDeque<Float>(historySize)

    /**
     * Process one frame.  Returns BatMovement combining current orientation,
     * angular velocity, swing phase, and a human-readable description.
     */
    fun update(keypoints: List<Keypoint>): BatMovement {
        val result = BatOrientationEstimator.estimate(keypoints)

        if (result.orientation == BatOrientation.UNKNOWN) {
            return BatMovement(
                current          = result,
                swingPhase       = SwingPhase.IDLE,
                angularVelocity  = 0f,
                description      = "Bat not detected"
            )
        }

        // Push angle into history
        angleHistory.addLast(result.angleDegrees)
        if (angleHistory.size > historySize) angleHistory.removeFirst()

        // Angular velocity = average per-frame delta over the window
        val angVel = computeAngularVelocity()

        val phase       = determineSwingPhase(result.orientation, angVel)
        val description = buildDescription(result.orientation, angVel, phase)

        return BatMovement(result, phase, angVel, description)
    }

    /** Expose the most recent raw angle for external callers (e.g. ShotClassifier). */
    fun lastAngle(): Float = angleHistory.lastOrNull() ?: 0f

    fun reset() { angleHistory.clear() }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun computeAngularVelocity(): Float {
        if (angleHistory.size < 2) return 0f
        var total = 0f
        for (i in 1 until angleHistory.size) {
            var delta = angleHistory[i] - angleHistory[i - 1]
            // Wrap-around guard (e.g. 179° → -179° should be ~2°, not 358°)
            if (delta >  90f) delta -= 180f
            if (delta < -90f) delta += 180f
            total += delta
        }
        return total / (angleHistory.size - 1)
    }

    private fun determineSwingPhase(
        orientation: BatOrientation,
        angVel: Float
    ): SwingPhase {
        val isMoving = abs(angVel) > 2f   // > 2°/frame = genuine rotation

        if (!isMoving) return SwingPhase.IDLE

        return when (orientation) {
            // Bat is roughly vertical and rotating → backlift or start of downswing
            BatOrientation.VERTICAL,
            BatOrientation.DIAGONAL_UPLEFT -> {
                if (angVel < 0f) SwingPhase.BACKLIFT else SwingPhase.DOWNSWING
            }
            // Bat is horizontal → impact zone (cross-bat shots or just before contact)
            BatOrientation.HORIZONTAL -> SwingPhase.IMPACT_ZONE

            // Bat has passed through horizontal and is going up on the other side
            BatOrientation.DIAGONAL_UPRIGHT -> SwingPhase.FOLLOW_THROUGH

            BatOrientation.UNKNOWN -> SwingPhase.IDLE
        }
    }

    private fun buildDescription(
        orientation: BatOrientation,
        angVel: Float,
        phase: SwingPhase
    ): String {
        val oriStr = when (orientation) {
            BatOrientation.VERTICAL          -> "Vertical"
            BatOrientation.HORIZONTAL        -> "Horizontal"
            BatOrientation.DIAGONAL_UPLEFT   -> "Diagonal ↖"
            BatOrientation.DIAGONAL_UPRIGHT  -> "Diagonal ↗"
            BatOrientation.UNKNOWN           -> "Unknown"
        }
        val phaseStr = when (phase) {
            SwingPhase.BACKLIFT        -> "Backlift"
            SwingPhase.DOWNSWING       -> "Downswing"
            SwingPhase.IMPACT_ZONE     -> "Impact"
            SwingPhase.FOLLOW_THROUGH  -> "Follow-through"
            SwingPhase.IDLE            -> "Idle"
        }
        val velStr = "${"%.1f".format(abs(angVel))}°/f"
        return "$oriStr · $phaseStr · $velStr"
    }
}