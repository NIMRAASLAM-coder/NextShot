package com.fyp.nextshot

import kotlin.math.sqrt
import android.util.Log

/**
 * ShotEventDetector v5
 *
 * Changes over v4:
 *
 *  1. BAT ORIENTATION INTEGRATION
 *     A BatMovementTracker instance lives inside the detector.
 *     Every call to feed() also calls batTracker.update() and passes the
 *     resulting BatAngleResult into ShotClassifier.classify() so that bat
 *     orientation contributes to per-frame labelling.
 *
 *  2. SWING-PHASE-AWARE STATE TRANSITIONS
 *     The state machine now cross-checks BatMovementTracker.swingPhase:
 *
 *     • BACKLIFT  phase → suppresses premature SWINGING entry.
 *       (Wrist can move fast during backlift; we don't want that to start
 *        a shot window before the downswing begins.)
 *
 *     • DOWNSWING phase → accelerates SWINGING entry even at lower wrist velocity.
 *       (Bat is definitely on its way to the ball — open the window early.)
 *
 *     • IMPACT_ZONE phase → marks the peak-velocity frame as the best label frame.
 *
 *     • FOLLOW_THROUGH phase → counts toward SETTLING faster (shot is done).
 *
 *  3. DUAL-SIGNAL PEAK SELECTION
 *     pickBestLabel() now uses both peak wrist-velocity AND the frame(s)
 *     that coincided with IMPACT_ZONE phase to vote for the label.
 *
 *  4. BAT TRACKER RESET
 *     reset() and clearSwingWindow() also reset the BatMovementTracker so
 *     stale angle history from a previous video doesn't pollute a new session.
 */
class ShotEventDetector {

    // ── Velocity thresholds ───────────────────────────────────────────────────
    private val VELOCITY_SWING_START      = 0.006f
    private val VELOCITY_SWING_START_DOWN = 0.004f  // lower threshold during DOWNSWING phase
    private val VELOCITY_SWING_HOLD       = 0.004f
    private val VELOCITY_SETTLE_ENTER     = 0.003f
    private val VELOCITY_PEAK_REQUIRED    = 0.010f

    private val SETTLE_FRAMES    = 2
    private val MIN_SWING_FRAMES = 2
    private val CONF_THRESHOLD   = 0.25f

    // ── Bat tracker (one per detector instance) ───────────────────────────────
    private val batTracker = BatMovementTracker(historySize = 6)

    // ── State machine ─────────────────────────────────────────────────────────
    private enum class State { IDLE, SWINGING, SETTLING }
    private var state = State.IDLE

    // Per-swing accumulators
    private val swingLabels      = mutableListOf<String>()
    private val swingVelocities  = mutableListOf<Float>()
    private val swingBatPhases   = mutableListOf<SwingPhase>()  // NEW: parallel bat-phase list

    private var settleCount          = 0
    private var swingFrameCount      = 0
    private var lastWristPos: Pair<Float, Float>? = null
    private var maxVelocityInSwing   = 0f
    private var impactFrameLabel     = ""   // label captured at IMPACT_ZONE phase

    /**
     * The last committed shot label. Never reset to "" after finalization —
     * overlay keeps the last real shot even when wrist becomes idle.
     */
    var finalizedLabel = ""
        private set

    /** Incremented every time a new shot is committed. */
    var shotEventCount = 0
        private set

    private val shotHistory = mutableListOf<String>()

    /** Returns a map of shot label → count, e.g. {"Drive"→3, "Pull"→1}. */
    fun getShotSummary(): Map<String, Int> =
        shotHistory.groupingBy { it }.eachCount()

    // ── Public API ────────────────────────────────────────────────────────────

    /** Full reset — call when starting a brand-new session or video. */
    fun reset() {
        state = State.IDLE
        swingLabels.clear()
        swingVelocities.clear()
        swingBatPhases.clear()
        settleCount          = 0
        swingFrameCount      = 0
        lastWristPos         = null
        finalizedLabel       = ""
        maxVelocityInSwing   = 0f
        impactFrameLabel     = ""
        shotEventCount       = 0
        shotHistory.clear()
        batTracker.reset()
    }

    /**
     * Force-finalize any in-progress swing.
     * Call after the last frame of a video or when a live session stops.
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
        clearSwingWindow()
        return finalizedLabel
    }

    /**
     * Feed one frame of keypoints.
     *
     * Internally:
     *  1. Updates BatMovementTracker to get current orientation + phase.
     *  2. Computes wrist velocity.
     *  3. Calls ShotClassifier with BOTH wrist velocity and bat angle.
     *  4. Runs the state machine with phase-aware thresholds.
     *
     * Returns finalizedLabel (the last committed shot, or "" if none yet).
     */
    fun feed(keypoints: List<Keypoint>): String {
        val wristPos = getBestWristPos(keypoints) ?: return finalizedLabel

        if (!wristPos.first.isFinite() || !wristPos.second.isFinite()) {
            lastWristPos = null
            return finalizedLabel
        }

        // ── 1. Bat orientation this frame ──────────────────────────────────
        val batMovement   = batTracker.update(keypoints)
        val batAngle      = batMovement.current        // BatAngleResult
        val batPhase      = batMovement.swingPhase     // SwingPhase

        // ── 2. Wrist velocity ──────────────────────────────────────────────
        val velocity = if (lastWristPos != null) {
            val dx = wristPos.first  - lastWristPos!!.first
            val dy = wristPos.second - lastWristPos!!.second
            val v  = sqrt(dx * dx + dy * dy)
            if (v.isFinite()) v else 0f
        } else 0f

        lastWristPos = wristPos

        // ── 3. Per-frame label (pose + bat angle) ──────────────────────────
        val frameLabel = ShotClassifier.classify(keypoints, velocity, batAngle)

        Log.d("SHOT_DETECTOR",
            "feed  state=$state  vel=${"%.4f".format(velocity)}  " +
                    "batPhase=$batPhase  batOri=${batAngle.orientation}  " +
                    "label=${frameLabel.ifEmpty { "—" }}  peak=${"%.4f".format(maxVelocityInSwing)}")

        // ── 4. Phase-aware entry threshold ────────────────────────────────
        // During DOWNSWING the bat is already on its way — open the swing
        // window at a lower velocity so we don't miss slow deliberate shots.
        val swingStartThreshold = if (batPhase == SwingPhase.DOWNSWING)
            VELOCITY_SWING_START_DOWN
        else
            VELOCITY_SWING_START

        // ── 5. State machine ───────────────────────────────────────────────
        when (state) {

            State.IDLE -> {
                // Suppress SWINGING entry during BACKLIFT — wrist may be fast
                // but the bat hasn't started coming down yet.
                val inBacklift = (batPhase == SwingPhase.BACKLIFT)

                if (velocity > swingStartThreshold && !inBacklift) {
                    state = State.SWINGING
                    swingLabels.clear()
                    swingVelocities.clear()
                    swingBatPhases.clear()
                    swingFrameCount      = 1
                    maxVelocityInSwing   = velocity
                    impactFrameLabel     = ""

                    if (frameLabel.isNotEmpty()) {
                        swingLabels    += frameLabel
                        swingVelocities += velocity
                        swingBatPhases  += batPhase
                    }
                }
                // Do NOT update finalizedLabel while IDLE
            }

            State.SWINGING -> {
                swingFrameCount++
                maxVelocityInSwing = maxOf(maxVelocityInSwing, velocity)

                if (frameLabel.isNotEmpty()) {
                    swingLabels    += frameLabel
                    swingVelocities += velocity
                    swingBatPhases  += batPhase
                }

                // Capture label at IMPACT_ZONE — this is the most reliable frame
                if (batPhase == SwingPhase.IMPACT_ZONE && frameLabel.isNotEmpty()) {
                    impactFrameLabel = frameLabel
                }

                // FOLLOW_THROUGH phase counts as settling immediately
                if (batPhase == SwingPhase.FOLLOW_THROUGH || velocity < VELOCITY_SETTLE_ENTER) {
                    state       = State.SETTLING
                    settleCount = 1
                }
            }

            State.SETTLING -> {
                if (frameLabel.isNotEmpty()) {
                    swingLabels    += frameLabel
                    swingVelocities += velocity
                    swingBatPhases  += batPhase
                }

                if (velocity > VELOCITY_SWING_HOLD && batPhase != SwingPhase.FOLLOW_THROUGH) {
                    // Genuine continuation of swing (not just follow-through wobble)
                    state = State.SWINGING
                    settleCount = 0
                    swingFrameCount++
                    maxVelocityInSwing = maxOf(maxVelocityInSwing, velocity)
                } else {
                    settleCount++
                    // Settle faster when bat phase confirms shot is done
                    val settleTarget = if (batPhase == SwingPhase.FOLLOW_THROUGH ||
                        batPhase == SwingPhase.IDLE) 1
                    else SETTLE_FRAMES
                    if (settleCount >= settleTarget) {
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
                    "peak=${"%.4f".format(maxVelocityInSwing)}  labels=$swingLabels  " +
                    "impact=$impactFrameLabel")
    }

    /**
     * Pick the best label from the current swing window using three signals
     * in priority order:
     *
     *  1. impactFrameLabel — label captured exactly when bat was in IMPACT_ZONE.
     *     This is the most geometrically accurate moment (bat meets ball).
     *
     *  2. Peak-velocity frame label — highest wrist speed = actual stroke.
     *
     *  3. Majority vote (non-Defensive preferred over Defensive).
     */
    private fun pickBestLabel(): String {
        if (swingLabels.isEmpty()) return ""

        // 1. Impact-zone frame
        if (impactFrameLabel.isNotEmpty() && impactFrameLabel != ShotClassifier.FRONT_FOOT_DEF) {
            Log.d("SHOT_DETECTOR", "pickBest: impact-frame → $impactFrameLabel")
            return impactFrameLabel
        }

        // 2. Peak-velocity frame
        if (swingVelocities.isNotEmpty()) {
            val maxVel  = swingVelocities.max()
            val peakIdx = swingVelocities.indexOfFirst { it == maxVel }
            val peak    = swingLabels.getOrNull(peakIdx) ?: ""
            if (peak.isNotEmpty() && peak != ShotClassifier.FRONT_FOOT_DEF) {
                Log.d("SHOT_DETECTOR", "pickBest: peak-frame → $peak (v=${"%.4f".format(maxVel)})")
                return peak
            }
        }

        // 3. Majority vote — non-Defensive first
        val nonDef = swingLabels.filter { it.isNotEmpty() && it != ShotClassifier.FRONT_FOOT_DEF }
        if (nonDef.isNotEmpty()) {
            val v = majorityVote(nonDef)
            Log.d("SHOT_DETECTOR", "pickBest: majority (non-def) → $v")
            return v
        }

        val v = majorityVote(swingLabels.filter { it.isNotEmpty() })
        Log.d("SHOT_DETECTOR", "pickBest: majority (all) → $v")
        return v
    }

    /** Clear per-swing accumulators without touching finalizedLabel. */
    private fun clearSwingWindow() {
        state = State.IDLE
        swingLabels.clear()
        swingVelocities.clear()
        swingBatPhases.clear()
        settleCount        = 0
        swingFrameCount    = 0
        maxVelocityInSwing = 0f
        impactFrameLabel   = ""
        // NOTE: batTracker is NOT reset here — its angle history should carry
        // over between shots so the first frame of the next swing has context.
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

        // Fallback to elbows
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