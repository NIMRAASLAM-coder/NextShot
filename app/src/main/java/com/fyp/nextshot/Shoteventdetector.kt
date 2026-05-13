package com.fyp.nextshot

import kotlin.math.sqrt
import android.util.Log

/**
 * ShotEventDetector v4
 *
 * Fixes over v3:
 *
 *  1. PASSES VELOCITY TO CLASSIFIER — ShotClassifier.classify() now takes
 *     the current wrist velocity so it can return "" for idle frames.
 *     This eliminates the "Cut shown while standing still" bug completely.
 *
 *  2. OVERLAY FREEZE on video end — finalizedLabel is NEVER cleared after
 *     a shot is committed. The overlay keeps showing the last real shot
 *     even when the video loops back to frame 0 (idle stance).
 *     Previously reset() was called on re-entry which wiped the label.
 *
 *  3. PEAK-LABEL selection — label taken from the highest-velocity frame
 *     (impact moment), not the last frame (follow-through/idle).
 *
 *  4. SHORT-VIDEO tuning — MIN_SWING_FRAMES=2, SETTLE_FRAMES=2.
 *     A 2-second video at 200ms = 10 frames; old values (4+3) consumed 70%.
 *
 *  5. VELOCITY_PEAK_REQUIRED lowered 0.015 → 0.010 for short clips.
 */
class ShotEventDetector {

    // ── Velocity thresholds ───────────────────────────────────────────────────
    private val VELOCITY_SWING_START   = 0.006f   // enter SWINGING
    private val VELOCITY_SWING_HOLD    = 0.004f   // stay in SWINGING from SETTLING
    private val VELOCITY_SETTLE_ENTER  = 0.003f   // enter SETTLING
    private val VELOCITY_PEAK_REQUIRED = 0.010f   // reject micro-movements

    private val SETTLE_FRAMES    = 2
    private val MIN_SWING_FRAMES = 2
    private val CONF_THRESHOLD   = 0.25f

    // ── State machine ─────────────────────────────────────────────────────────
    private enum class State { IDLE, SWINGING, SETTLING }
    private var state = State.IDLE

    private val swingLabels     = mutableListOf<String>()
    private val swingVelocities = mutableListOf<Float>()

    private var settleCount        = 0
    private var swingFrameCount    = 0
    private var lastWristPos: Pair<Float, Float>? = null
    private var maxVelocityInSwing = 0f

    /**
     * The last committed shot label. Intentionally NEVER reset to "" after
     * a shot is finalized — the UI should keep showing the last real result
     * rather than reverting to blank (which makes it look like "Cut" flashed
     * from the idle-frame detection at video start).
     */
    var finalizedLabel = ""
        private set

    /** Incremented every time a new shot is committed. */
    var shotEventCount = 0
        private set

    /**
     * Ordered list of every committed shot label for this session.
     * Use getShotSummary() to get a formatted count map for display.
     */
    private val shotHistory = mutableListOf<String>()

    /** Returns a map of shot label → count, e.g. {"Drive"→3, "Pull"→1}. */
    fun getShotSummary(): Map<String, Int> =
        shotHistory.groupingBy { it }.eachCount()

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Full reset — call only when starting a brand-new session/video.
     * Clears finalizedLabel so the previous video's result does not bleed in.
     */
    fun reset() {
        state = State.IDLE
        swingLabels.clear()
        swingVelocities.clear()
        settleCount = 0
        swingFrameCount = 0
        lastWristPos = null
        finalizedLabel = ""          // intentional: new session, clean slate
        maxVelocityInSwing = 0f
        shotEventCount = 0
        shotHistory.clear()
    }

    /**
     * Call after the last frame of a video (or when live session stops).
     * Forces finalization of any in-progress swing so the label is never lost
     * in short videos that end before the wrist fully settles.
     */
    fun flush(): String {
        if ((state == State.SWINGING || state == State.SETTLING)
            && swingFrameCount >= MIN_SWING_FRAMES
            && swingLabels.isNotEmpty()
            && maxVelocityInSwing >= VELOCITY_PEAK_REQUIRED
        ) {
            val label = pickBestLabel()
            if (label.isNotEmpty()) {
                finalizedLabel = label
                shotEventCount++
                shotHistory += label
                Log.d("SHOT_DETECTOR",
                    "flush() → $finalizedLabel  frames=$swingFrameCount  " +
                            "peak=${"%.4f".format(maxVelocityInSwing)}  labels=$swingLabels")
            }
        }
        clearSwingWindow()   // clear window state but NOT finalizedLabel
        return finalizedLabel
    }

    /**
     * Feed one frame. Returns finalizedLabel (the last committed shot).
     * Returns "" only if no shot has ever been finalized in this session.
     *
     * The overlay should display finalizedLabel; "" means "no shot yet —
     * show nothing", NOT "show Cut".
     */
    fun feed(keypoints: List<Keypoint>): String {
        val wristPos = getBestWristPos(keypoints) ?: return finalizedLabel

        if (!wristPos.first.isFinite() || !wristPos.second.isFinite()) {
            lastWristPos = null
            return finalizedLabel
        }

        // Compute wrist velocity this frame
        val velocity = if (lastWristPos != null) {
            val dx = wristPos.first  - lastWristPos!!.first
            val dy = wristPos.second - lastWristPos!!.second
            val v  = sqrt(dx * dx + dy * dy)
            if (v.isFinite()) v else 0f
        } else 0f

        lastWristPos = wristPos

        // Pass velocity into the classifier — it returns "" when wrist is still.
        // This is what prevents the idle standing pose from showing as "Cut".
        val frameLabel = ShotClassifier.classify(keypoints, velocity)

        Log.d("SHOT_DETECTOR",
            "feed  state=$state  vel=${"%.4f".format(velocity)}  " +
                    "label=${frameLabel.ifEmpty { "—" }}  peak=${"%.4f".format(maxVelocityInSwing)}")

        when (state) {
            State.IDLE -> {
                if (velocity > VELOCITY_SWING_START) {
                    state = State.SWINGING
                    swingLabels.clear()
                    swingVelocities.clear()
                    swingFrameCount = 1
                    maxVelocityInSwing = velocity
                    if (frameLabel.isNotEmpty()) {
                        swingLabels += frameLabel
                        swingVelocities += velocity
                    }
                }
                // While IDLE, do NOT update finalizedLabel — keep the last real shot visible
            }

            State.SWINGING -> {
                swingFrameCount++
                maxVelocityInSwing = maxOf(maxVelocityInSwing, velocity)
                if (frameLabel.isNotEmpty()) {
                    swingLabels += frameLabel
                    swingVelocities += velocity
                }
                if (velocity < VELOCITY_SETTLE_ENTER) {
                    state = State.SETTLING
                    settleCount = 1
                }
            }

            State.SETTLING -> {
                if (frameLabel.isNotEmpty()) {
                    swingLabels += frameLabel
                    swingVelocities += velocity
                }
                if (velocity > VELOCITY_SWING_HOLD) {
                    state = State.SWINGING
                    settleCount = 0
                    swingFrameCount++
                    maxVelocityInSwing = maxOf(maxVelocityInSwing, velocity)
                } else {
                    settleCount++
                    if (settleCount >= SETTLE_FRAMES) {
                        tryFinalize()
                        clearSwingWindow()
                    }
                }
            }
        }

        return finalizedLabel
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun tryFinalize() {
        if (swingFrameCount < MIN_SWING_FRAMES
            || swingLabels.isEmpty()
            || maxVelocityInSwing < VELOCITY_PEAK_REQUIRED
        ) {
            Log.d("SHOT_DETECTOR",
                "REJECTED  frames=$swingFrameCount  " +
                        "peak=${"%.4f".format(maxVelocityInSwing)}")
            return
        }
        val label = pickBestLabel()
        if (label.isEmpty()) return

        finalizedLabel = label
        shotEventCount++
        shotHistory += label
        Log.d("SHOT_DETECTOR",
            "COMMITTED → $finalizedLabel  frames=$swingFrameCount  " +
                    "peak=${"%.4f".format(maxVelocityInSwing)}  labels=$swingLabels")
    }

    /**
     * Pick the best label from the current swing window:
     *  1. Label at peak-velocity frame (impact moment)
     *  2. Majority vote excluding "Defensive" (real shot dominates mid-swing)
     *  3. Majority vote over everything
     */
    private fun pickBestLabel(): String {
        if (swingLabels.isEmpty()) return ""

        // 1. Peak-velocity frame
        if (swingVelocities.isNotEmpty()) {
            val maxVel  = swingVelocities.max()
            val peakIdx = swingVelocities.indexOfFirst { it == maxVel }
            val peak    = swingLabels.getOrNull(peakIdx) ?: ""
            if (peak.isNotEmpty() && peak != "Defensive") {
                Log.d("SHOT_DETECTOR", "pickBest: peak-frame → $peak (v=${"%.4f".format(maxVel)})")
                return peak
            }
        }

        // 2. Majority vote (non-Defensive)
        val nonDef = swingLabels.filter { it.isNotEmpty() && it != "Defensive" }
        if (nonDef.isNotEmpty()) {
            val v = majorityVote(nonDef)
            Log.d("SHOT_DETECTOR", "pickBest: majority (non-def) → $v")
            return v
        }

        // 3. Majority vote (all)
        val v = majorityVote(swingLabels.filter { it.isNotEmpty() })
        Log.d("SHOT_DETECTOR", "pickBest: majority (all) → $v")
        return v
    }

    /** Clears the per-swing window but intentionally leaves finalizedLabel alone. */
    private fun clearSwingWindow() {
        state = State.IDLE
        swingLabels.clear()
        swingVelocities.clear()
        settleCount = 0
        swingFrameCount = 0
        maxVelocityInSwing = 0f
    }

    private fun getBestWristPos(keypoints: List<Keypoint>): Pair<Float, Float>? {
        val lw = keypoints.getOrNull(9)?.takeIf  { it.confidence >= CONF_THRESHOLD }
        val rw = keypoints.getOrNull(10)?.takeIf { it.confidence >= CONF_THRESHOLD }
        val wr = when {
            lw != null && rw != null -> (lw.x + rw.x) / 2f to (lw.y + rw.y) / 2f
            lw != null               -> lw.x to lw.y
            rw != null               -> rw.x to rw.y
            else                     -> null
        }
        if (wr != null) return wr

        // Fallback to elbows when wrists are occluded
        val le = keypoints.getOrNull(7)?.takeIf  { it.confidence >= CONF_THRESHOLD }
        val re = keypoints.getOrNull(8)?.takeIf  { it.confidence >= CONF_THRESHOLD }
        return when {
            le != null && re != null -> (le.x + re.x) / 2f to (le.y + re.y) / 2f
            le != null               -> le.x to le.y
            re != null               -> re.x to re.y
            else                     -> null
        }
    }

    private fun majorityVote(labels: List<String>): String =
        labels.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: ""
}