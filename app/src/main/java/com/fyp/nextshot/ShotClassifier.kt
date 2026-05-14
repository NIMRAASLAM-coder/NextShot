package com.fyp.nextshot

import kotlin.math.abs
import kotlin.math.atan2

/**
 * ShotClassifier v4 — five geometrically distinct cricket shots for SIDE-VIEW footage.
 *
 * Shots chosen for minimal overlap in a pure side-on camera angle:
 *
 *  1. Sweep             — deep knee bend + wrist very low + horizontal bat arc
 *  2. Pull              — wrist above / at shoulder, back foot, high horizontal swing
 *  3. Cover Drive       — front foot, wrist at chest/shoulder height, arm extended off-side
 *  4. Leg Glance        — wrist at hip level, compact extension, leg-side deflection / hip rotation
 *  5. Front Foot Defensive — front foot, wrist low/compact, vertical bat near pitch
 *
 * Why these five are distinct in side-view:
 *  • Sweep:      body crouches (knee drops), bat at ground level → totally unique silhouette
 *  • Pull:       wrist rises ABOVE shoulder on BACK foot        → clearly above the shoulder line
 *  • Cover Drive:wrist at chest, arm pushes FORWARD away from body → large wrist-extension value
 *  • Leg Glance: wrist at hip alongside body, minimal extension  → compact, hip-rotated silhouette
 *  • Defensive:  front foot but wrist stays LOW and COMPACT     → residual / catch-all
 *
 * COCO keypoint indices:
 *  0=Nose, 5=L.Shoulder, 6=R.Shoulder,
 *  7=L.Elbow, 8=R.Elbow, 9=L.Wrist, 10=R.Wrist,
 *  11=L.Hip, 12=R.Hip, 13=L.Knee, 14=R.Knee,
 *  15=L.Ankle, 16=R.Ankle
 */
object ShotClassifier {

    private const val CONF_THRESHOLD = 0.25f

    /**
     * Minimum wrist velocity (normalised coords/frame) required before any shot is labelled.
     * Prevents idle standing poses from being misclassified.
     */
    private const val MIN_VELOCITY_TO_CLASSIFY = 0.005f

    // ── Shot label constants ─────────────────────────────────────────────────
    const val SWEEP              = "Sweep"
    const val PULL               = "Pull"
    const val COVER_DRIVE        = "Cover Drive"
    const val LEG_GLANCE         = "Leg Glance"
    const val FRONT_FOOT_DEF     = "Front Foot Defensive"

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Classify one frame.
     *
     * @param keypoints     COCO-17 keypoints for the frame.
     * @param wristVelocity Wrist speed in normalised coords/frame.
     *                      Pass 0f if unknown; below [MIN_VELOCITY_TO_CLASSIFY] → returns "".
     * @param batAngle      Optional bat orientation from [BatOrientationEstimator].
     */
    fun classify(
        keypoints: List<Keypoint>,
        wristVelocity: Float = 0f,
        batAngle: BatAngleResult? = null
    ): String {

        // ── Idle guard ────────────────────────────────────────────────────────
        if (wristVelocity < MIN_VELOCITY_TO_CLASSIFY) return ""

        // ── Landmark helpers ──────────────────────────────────────────────────
        fun pt(idx: Int): Pair<Float, Float>? {
            val k = keypoints.getOrNull(idx) ?: return null
            return if (k.confidence >= CONF_THRESHOLD) k.x to k.y else null
        }

        val lShoulder = pt(5);  val rShoulder = pt(6)
        val lHip      = pt(11); val rHip      = pt(12)
        val lKnee     = pt(13); val rKnee     = pt(14)
        val lAnkle    = pt(15); val rAnkle    = pt(16)
        val lWrist    = pt(9);  val rWrist    = pt(10)

        // Need at least one wrist and both shoulders to proceed
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

        // Normalise body height to make metrics resolution-independent
        val bodyHeight = (ankleY - shoulderY).coerceAtLeast(0.05f)

        val hipY = when {
            lHip != null && rHip != null -> (lHip.second + rHip.second) / 2f
            else                         -> shoulderY + bodyHeight * 0.45f
        }

        // ── Wrist metrics (dominant = the one furthest from shoulder centre) ──

        val wrists        = listOfNotNull(lWrist, rWrist)
        val domWrist      = wrists.maxByOrNull { abs(it.first - shoulderX) } ?: wrists[0]
        val wristX        = domWrist.first
        val wristY        = domWrist.second

        // Positive = below shoulder; negative = above shoulder (Y increases downward)
        val wristHeightNorm = (wristY - shoulderY) / bodyHeight
        // How far the wrist is pushed horizontally from the body midline (off-side reach)
        val wristExtension  = abs((wristX - shoulderX) / bodyHeight)

        // ── Wrist height zones ────────────────────────────────────────────────
        //
        //  ABOVE_SHOULDER : wristHeightNorm < -0.05      → Pull territory
        //  AT_SHOULDER    :                -0.05 .. 0.15  → Pull / Cover Drive boundary
        //  AT_CHEST       :                 0.15 .. 0.42  → Cover Drive / Defensive
        //  AT_HIP         :                 0.42 .. 0.62  → Leg Glance
        //  VERY_LOW       :                > 0.62          → Sweep
        //
        val wristAboveShoulder = wristHeightNorm < -0.05f
        val wristAtShoulder    = wristHeightNorm in -0.05f..0.15f
        val wristAtChest       = wristHeightNorm in  0.15f..0.42f
        val wristAtHip         = wristHeightNorm in  0.42f..0.62f
        val wristVeryLow       = wristHeightNorm  >  0.62f

        // ── Knee bend analysis ────────────────────────────────────────────────
        var frontKneeBend   = false  // front foot engaged (stride forward)
        var deepKneeBend    = false  // knee very close to ankle — sweep crouch
        var kneesVisibility = false

        if (lKnee != null && rKnee != null) {
            kneesVisibility = true
            val kneeSpan = (ankleY - hipY).coerceAtLeast(0.01f)

            // In side view the "front" knee is the one further from the shoulder midpoint
            val frontKnee = if (abs(lKnee.first - shoulderX) >= abs(rKnee.first - shoulderX))
                lKnee else rKnee

            val frontKneeDepth = (frontKnee.second - hipY) / kneeSpan
            frontKneeBend = frontKneeDepth > 0.55f
            // Deep = at least one knee is almost at ankle level (sweep crouch)
            val lowestKneeDepth = maxOf(lKnee.second, rKnee.second)
            deepKneeBend = (lowestKneeDepth - hipY) / kneeSpan > 0.82f
        }

        // ── Bat orientation helpers ───────────────────────────────────────────
        val batOri      = batAngle?.orientation ?: BatOrientation.UNKNOWN
        val batKnown    = batOri != BatOrientation.UNKNOWN
        val batCross    = batKnown &&
                (batOri == BatOrientation.HORIZONTAL || batOri == BatOrientation.DIAGONAL_UPRIGHT)
        val batStraight = batKnown &&
                (batOri == BatOrientation.VERTICAL || batOri == BatOrientation.DIAGONAL_UPLEFT)

        // ── Classification — most specific rules first ─────────────────────────
        return when {

            // ─────────────────────────────────────────────────────────────────
            // 1. SWEEP  — knee drops to ground + bat sweeps at low height
            //    Signature: body crouches (deepKneeBend OR wristVeryLow AND frontKneeBend),
            //               bat is horizontal/cross-bat (or unknown)
            //    Impossible overlap: no other shot has wristVeryLow + deep crouch.
            // ─────────────────────────────────────────────────────────────────
            deepKneeBend && (wristVeryLow || wristAtHip) ->
                SWEEP

            // Catch sweep even when knees aren't perfectly detected:
            // wrist very low + front foot engaged is still a sweep-like posture
            wristVeryLow && frontKneeBend ->
                SWEEP

            // ─────────────────────────────────────────────────────────────────
            // 2. PULL SHOT — wrist at or ABOVE shoulder + back foot preferred
            //    Signature: short-ball response; high, horizontal bat swing.
            //    Guard: if bat is clearly vertical (straight bat) at shoulder height
            //           and front foot is engaged, it's more likely Defensive backlift.
            // ─────────────────────────────────────────────────────────────────
            wristAboveShoulder && !deepKneeBend ->
                PULL

            wristAtShoulder && !frontKneeBend && !deepKneeBend ->
                if (batStraight) FRONT_FOOT_DEF else PULL

            // ─────────────────────────────────────────────────────────────────
            // 3. COVER DRIVE — front foot, wrist at chest/shoulder, arm
            //    pushed well FORWARD away from the body (off-side extension).
            //    Key discriminator vs Defensive: wristExtension must be LARGE.
            //    Cover drive arm extends toward mid-off/cover point.
            // ─────────────────────────────────────────────────────────────────
            frontKneeBend &&
                    (wristAtChest || wristAtShoulder) &&
                    wristExtension > 0.24f ->
                COVER_DRIVE

            // ─────────────────────────────────────────────────────────────────
            // 4. LEG GLANCE — wrist at hip alongside body, compact extension,
            //    hip rotation deflects the ball fine leg-wards.
            //    Key discriminators:
            //      • wrist NOT as low as Sweep (at hip, not very low)
            //      • wrist NOT extended far from body (compact flick)
            //      • unlike Cover Drive: smaller wristExtension
            // ─────────────────────────────────────────────────────────────────
            wristAtHip && wristExtension < 0.28f ->
                LEG_GLANCE

            // Leg glance can also appear as a chest-height compact flick on the
            // front foot when the wrist is barely extended (on-drive-flick variant)
            wristAtChest && frontKneeBend && wristExtension < 0.14f ->
                LEG_GLANCE

            // ─────────────────────────────────────────────────────────────────
            // 5. FRONT FOOT DEFENSIVE — catch-all for front-foot, low, compact postures
            //    The bat is held close to the body (small extension) with a
            //    vertical face; the wrist stays at chest or hip height.
            // ─────────────────────────────────────────────────────────────────
            frontKneeBend && (wristAtChest || wristAtHip) ->
                FRONT_FOOT_DEF

            // Pure default
            else -> FRONT_FOOT_DEF
        }
    }
}