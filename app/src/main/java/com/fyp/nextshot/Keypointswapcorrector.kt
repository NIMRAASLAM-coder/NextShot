package com.fyp.nextshot

import kotlin.math.abs

/**
 * KeypointSwapCorrector v2
 *
 * Root cause: COCO labels are body-relative ("left" = person's left). When the
 * model momentarily loses orientation, symmetric pairs flip for 1-3 frames.
 *
 * v1 weakness: pure X-order comparison breaks when both wrists are at similar X
 * (e.g. during a straight drive with the bat vertical).
 *
 * v2 fix — shoulder-anchored side assignment:
 *   Instead of comparing the two joints to each other, we compare each joint's X
 *   to the SHOULDER MIDPOINT X.  The shoulder midpoint is the most stable landmark
 *   on the body and doesn't move much between frames.
 *
 *   For each pair (leftIdx, rightIdx) we compute:
 *     leftSide  = sign(kp[leftIdx].x  - shoulderMidX)   (+1 right of mid, -1 left)
 *     rightSide = sign(kp[rightIdx].x - shoulderMidX)
 *
 *   "Trusted side" = the (leftSide, rightSide) tuple seen over the last N frames.
 *   If the new frame's tuple disagrees AND the joints are far enough apart (gap
 *   threshold), we swap them back.  If they're too close together (bat vertical,
 *   etc.) we skip the check entirely — ambiguous frames can't cause wrong swaps.
 *
 *   Commit hysteresis: a new side assignment is only accepted after COMMIT_FRAMES
 *   consecutive agreeing frames, so single-frame model noise is invisible.
 */
class KeypointSwapCorrector(
    /** Frames of consistent new ordering needed before we accept orientation change */
    private val commitFrames: Int = 5,
    /** Min confidence to participate in swap check */
    private val minConfidence: Float = 0.25f,
    /**
     * Min normalised distance between the two paired joints (relative to body
     * height = shoulder-to-ankle distance) before we bother checking.
     * Below this the joints are too close to tell which side they're on.
     */
    private val minPairGap: Float = 0.04f
) {

    // COCO symmetric pairs: (leftBodyIdx, rightBodyIdx)
    private val PAIRS = listOf(
        5  to 6,   // shoulders  — also our reference; still corrected if needed
        7  to 8,   // elbows
        9  to 10,  // wrists
        11 to 12,  // hips
        13 to 14,  // knees
        15 to 16   // ankles
    )

    /**
     * Signed side relative to shoulder mid:
     *   LEFT_OF_MID  (-1) = joint is left  of shoulder centre in image space
     *   RIGHT_OF_MID (+1) = joint is right of shoulder centre in image space
     */
    private enum class Side { LEFT_OF_MID, RIGHT_OF_MID, UNKNOWN }

    private data class PairState(
        // Currently committed (leftSide, rightSide) relative to shoulder mid
        var trustedLeft: Side = Side.UNKNOWN,
        var trustedRight: Side = Side.UNKNOWN,
        // How many consecutive frames have arrived with the opposite assignment
        var pendingCount: Int = 0,
        var pendingLeft: Side = Side.UNKNOWN,
        var pendingRight: Side = Side.UNKNOWN
    )

    private val states: List<PairState> = PAIRS.map { PairState() }

    fun correct(keypoints: List<Keypoint>): List<Keypoint> {
        if (keypoints.size < 17) return keypoints

        // ── Compute reference anchors ─────────────────────────────────────────
        val lSh = keypoints[5]
        val rSh = keypoints[6]

        val shouldersVisible = lSh.confidence >= minConfidence && rSh.confidence >= minConfidence
        val shoulderMidX = if (shouldersVisible) (lSh.x + rSh.x) / 2f else 0.5f

        // Body height for gap normalisation (shoulder Y → ankle Y)
        val lAnk = keypoints[15]
        val rAnk = keypoints[16]
        val ankleY = when {
            lAnk.confidence >= minConfidence && rAnk.confidence >= minConfidence -> (lAnk.y + rAnk.y) / 2f
            lAnk.confidence >= minConfidence -> lAnk.y
            rAnk.confidence >= minConfidence -> rAnk.y
            else -> (lSh.y + 0.5f).coerceAtMost(1f)
        }
        val bodyHeight = (ankleY - lSh.y).coerceAtLeast(0.10f)

        val result = keypoints.toMutableList()

        PAIRS.forEachIndexed { pIdx, (lIdx, rIdx) ->
            val lKp = result[lIdx]
            val rKp = result[rIdx]

            // Skip low-confidence joints
            if (lKp.confidence < minConfidence || rKp.confidence < minConfidence) return@forEachIndexed

            // Skip if the two joints are too close together (ambiguous frame)
            val pairGap = abs(lKp.x - rKp.x) / bodyHeight
            if (pairGap < minPairGap) return@forEachIndexed

            val currentLeft  = side(lKp.x, shoulderMidX)
            val currentRight = side(rKp.x, shoulderMidX)

            val st = states[pIdx]

            if (st.trustedLeft == Side.UNKNOWN) {
                // First good observation — commit immediately
                st.trustedLeft  = currentLeft
                st.trustedRight = currentRight
                st.pendingCount = 0
                return@forEachIndexed
            }

            val agreesWithTrusted = currentLeft == st.trustedLeft && currentRight == st.trustedRight

            if (agreesWithTrusted) {
                st.pendingCount = 0
                st.pendingLeft  = Side.UNKNOWN
                st.pendingRight = Side.UNKNOWN
            } else {
                // Disagrees — is it the same disagreement as last frame?
                if (currentLeft == st.pendingLeft && currentRight == st.pendingRight) {
                    st.pendingCount++
                } else {
                    // New/different disagreement — restart counter
                    st.pendingCount = 1
                    st.pendingLeft  = currentLeft
                    st.pendingRight = currentRight
                }

                if (st.pendingCount >= commitFrames) {
                    // Genuine orientation change — accept it
                    st.trustedLeft  = currentLeft
                    st.trustedRight = currentRight
                    st.pendingCount = 0
                    st.pendingLeft  = Side.UNKNOWN
                    st.pendingRight = Side.UNKNOWN
                } else {
                    // Transient noise — swap back to restore trusted assignment
                    result[lIdx] = rKp
                    result[rIdx] = lKp
                }
            }
        }

        return result
    }

    private fun side(x: Float, midX: Float): Side =
        if (x < midX) Side.LEFT_OF_MID else Side.RIGHT_OF_MID

    fun reset() {
        states.forEach { st ->
            st.trustedLeft  = Side.UNKNOWN
            st.trustedRight = Side.UNKNOWN
            st.pendingCount = 0
            st.pendingLeft  = Side.UNKNOWN
            st.pendingRight = Side.UNKNOWN
        }
    }
}