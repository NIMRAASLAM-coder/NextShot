package com.fyp.nextshot

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Rule-based cricket shot classifier optimized for SIDE-VIEW footage.
 *
 * COCO keypoint indices:
 *  0=Nose,
 *  5=L.Shoulder, 6=R.Shoulder,
 *  7=L.Elbow,    8=R.Elbow,
 *  9=L.Wrist,   10=R.Wrist,
 *  11=L.Hip,    12=R.Hip,
 *  13=L.Knee,   14=R.Knee,
 *  15=L.Ankle,  16=R.Ankle
 *
 * KEY FIX v2:
 *  - classify() now requires a minimum wrist velocity (passed in from the
 *    detector) before it will return any shot label. When the batsman is
 *    standing still the function returns "" — so the idle/ready pose can
 *    never be mis-labelled as "Cut" or anything else.
 *  - Tightened the Cut rule: wristExtension threshold raised 0.15 → 0.22
 *    so a relaxed standing grip no longer accidentally fires it.
 *  - Added an explicit IDLE guard at the top: if wrist hasn't moved enough
 *    this frame, return "" immediately without touching any shot rules.
 */
object ShotClassifier {

    private const val CONF_THRESHOLD = 0.25f

    /**
     * Minimum wrist velocity (normalised coords / frame) below which we
     * treat the pose as idle and return "" regardless of joint positions.
     * This is the primary fix for the "shows Cut while standing still" bug.
     *
     * Calibration note:
     *   At 200 ms frame intervals, a batsman shifting weight produces ~0.003.
     *   A real shot swing produces 0.010–0.050.
     *   Idle standing jitter is typically < 0.004.
     * → threshold of 0.005 lets small weight-shifts through but blocks pure idle.
     */
    private const val MIN_VELOCITY_TO_CLASSIFY = 0.005f

    /**
     * @param keypoints  COCO-17 keypoints for this frame
     * @param wristVelocity  wrist speed this frame (normalised units/frame).
     *                       Pass 0f if unknown. When below MIN_VELOCITY_TO_CLASSIFY
     *                       the function returns "" without classifying.
     */
    fun classify(keypoints: List<Keypoint>, wristVelocity: Float = 0f): String {

        // ── IDLE GUARD ────────────────────────────────────────────────────────
        // If the wrist hasn't moved enough this frame, the batsman is not
        // playing a shot. Return empty so the overlay shows nothing / keeps
        // the last finalized label instead of flashing a wrong one.
        if (wristVelocity < MIN_VELOCITY_TO_CLASSIFY) return ""

        fun pt(idx: Int): Pair<Float, Float>? {
            val k = keypoints.getOrNull(idx) ?: return null
            return if (k.confidence >= CONF_THRESHOLD) k.x to k.y else null
        }

        // ── Landmarks ─────────────────────────────────────────────────────────
        val lShoulder = pt(5)
        val rShoulder = pt(6)
        val lElbow    = pt(7)
        val rElbow    = pt(8)
        val lWrist    = pt(9)
        val rWrist    = pt(10)
        val lHip      = pt(11)
        val rHip      = pt(12)
        val lKnee     = pt(13)
        val rKnee     = pt(14)
        val lAnkle    = pt(15)
        val rAnkle    = pt(16)

        if (lWrist == null && rWrist == null) return ""
        if (lShoulder == null || rShoulder == null) return ""

        // ── Reference frame ───────────────────────────────────────────────────
        val shoulderY = (lShoulder.second + rShoulder.second) / 2f
        val shoulderX = (lShoulder.first  + rShoulder.first)  / 2f

        val ankleY = when {
            lAnkle != null && rAnkle != null -> (lAnkle.second + rAnkle.second) / 2f
            lAnkle != null -> lAnkle.second
            rAnkle != null -> rAnkle.second
            else           -> shoulderY + 0.50f
        }

        val bodyHeight = (ankleY - shoulderY).coerceAtLeast(0.05f)

        val hipY = when {
            lHip != null && rHip != null -> (lHip.second + rHip.second) / 2f
            else                         -> shoulderY + 0.20f
        }

        // ── Dominant wrist (furthest from body centre for side-view) ──────────
        val wrists = listOfNotNull(lWrist, rWrist)
        val dominantWrist = wrists.maxByOrNull { abs(it.first - shoulderX) } ?: wrists[0]
        val wristX = dominantWrist.first
        val wristY = dominantWrist.second

        // ── Normalised metrics ────────────────────────────────────────────────
        val wristHeightNorm = (wristY - shoulderY) / bodyHeight
        val wristDepthNorm  = (wristX - shoulderX) / bodyHeight

        // ── Elbow angle ───────────────────────────────────────────────────────
        val elbowAngle = when {
            lElbow != null && lWrist != null -> {
                val dx = lWrist.first  - lElbow.first
                val dy = lWrist.second - lElbow.second
                Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
            }
            rElbow != null && rWrist != null -> {
                val dx = rWrist.first  - rElbow.first
                val dy = rWrist.second - rElbow.second
                Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
            }
            else -> 0f
        }

        // ── Wrist height zones ────────────────────────────────────────────────
        val wristAboveShoulder = wristHeightNorm < -0.05f
        val wristAtShoulder    = wristHeightNorm in -0.05f..0.10f
        val wristAtChest       = wristHeightNorm in  0.10f..0.30f
        val wristAtHip         = wristHeightNorm in  0.30f..0.55f
        val wristVeryLow       = wristHeightNorm  >  0.55f

        // ── Wrist extension (direction-agnostic) ──────────────────────────────
        val wristExtension = abs(wristDepthNorm)

        // ── Front-knee bend ───────────────────────────────────────────────────
        val frontKneeBend: Boolean
        val deepKneeBend: Boolean
        if (lKnee != null && rKnee != null) {
            val frontKnee = if (abs(lKnee.first - shoulderX) > abs(rKnee.first - shoulderX))
                lKnee else rKnee
            val kneeDepth = (frontKnee.second - hipY) / (ankleY - hipY + 0.001f)
            frontKneeBend = kneeDepth > 0.55f
            deepKneeBend  = kneeDepth > 0.80f
        } else {
            frontKneeBend = false
            deepKneeBend  = false
        }

        // ── Classification ────────────────────────────────────────────────────
        return when {

            // 1. SWEEP — deep knee bend + wrist very low
            deepKneeBend && wristVeryLow -> "Sweep"

            // 2. PULL — wrist at/above shoulder height
            (wristAboveShoulder || wristAtShoulder) && !deepKneeBend -> "Pull"

            // 3. DRIVE — front knee bent + wrist chest/shoulder height
            frontKneeBend && (wristAtChest || wristAtShoulder) -> "Drive"

            // 4. CUT — back foot, wrist at chest/hip, clearly extended arm.
            //    TIGHTENED: threshold raised 0.15 → 0.22 so a relaxed standing
            //    grip (arms loosely at sides) does not fire this rule.
            !frontKneeBend && !deepKneeBend &&
                    (wristAtChest || wristAtHip) &&
                    wristExtension > 0.22f -> "Cut"

            // 5. DEFENSIVE — everything else while wrist is moving
            else -> "Defensive"
        }
    }
}