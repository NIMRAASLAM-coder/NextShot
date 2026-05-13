package com.fyp.nextshot

// KeypointSmootherConfig.kt — Preset Configurations

/**
 * Preset Kalman configurations for different scenarios.
 * Copy the appropriate filter creation logic into your KeypointKalmanFilter instantiation.
 */
object KeypointSmootherConfig {

    // ════════════════════════════════════════════════════════════════════════
    // PRESET 1: BALANCED (Default — good for most cases)
    // ════════════════════════════════════════════════════════════════════════
    val BALANCED_PROCESS_NOISE = 0.00005f
    val BALANCED_MEASUREMENT_NOISE = 0.001f
    val BALANCED_BUFFER_DELAY_MS = 700L

    /**
     * Use this if:
     * - You have decent WiFi (RTT ~200-400ms)
     * - You want smooth motion without noticeable lag
     * - Your Roboflow confidence is generally > 0.6
     */
    fun createBalancedFilters(keypointCount: Int): List<KeypointKalmanFilter> =
        List(keypointCount) {
            KeypointKalmanFilter(BALANCED_PROCESS_NOISE, BALANCED_MEASUREMENT_NOISE)
        }

    // ════════════════════════════════════════════════════════════════════════
    // PRESET 2: SNAPPY (Fast response, more jitter tolerance)
    // ════════════════════════════════════════════════════════════════════════
    val SNAPPY_PROCESS_NOISE = 0.0001f
    val SNAPPY_MEASUREMENT_NOISE = 0.0005f
    val SNAPPY_BUFFER_DELAY_MS = 500L

    /**
     * Use this if:
     * - You have very fast WiFi (RTT < 200ms)
     * - You want skeleton to respond immediately to motion
     * - You can tolerate a little jitter (will be absorbed by Kalman over ~3 frames)
     * - Good for real-time coaching feedback
     */
    fun createSnappyFilters(keypointCount: Int): List<KeypointKalmanFilter> =
        List(keypointCount) {
            KeypointKalmanFilter(SNAPPY_PROCESS_NOISE, SNAPPY_MEASUREMENT_NOISE)
        }

    // ════════════════════════════════════════════════════════════════════════
    // PRESET 3: BUTTERY (Ultra-smooth, slight lag acceptable)
    // ════════════════════════════════════════════════════════════════════════
    val BUTTERY_PROCESS_NOISE = 0.00003f
    val BUTTERY_MEASUREMENT_NOISE = 0.002f
    val BUTTERY_BUFFER_DELAY_MS = 900L

    /**
     * Use this if:
     * - You have slower WiFi (RTT 400-600ms)
     * - You want absolutely zero jitter (analysis mode, not real-time coaching)
     * - You can accept ~500ms+ latency
     * - Roboflow confidence tends to be inconsistent (< 0.5)
     */
    fun createButteryFilters(keypointCount: Int): List<KeypointKalmanFilter> =
        List(keypointCount) {
            KeypointKalmanFilter(BUTTERY_PROCESS_NOISE, BUTTERY_MEASUREMENT_NOISE)
        }

    // ════════════════════════════════════════════════════════════════════════
    // PRESET 4: PER-JOINT CUSTOM (Different smoothing per keypoint index)
    // ════════════════════════════════════════════════════════════════════════

    /**
     * COCO Keypoint indices:
     *  0=Nose, 1=L.Eye, 2=R.Eye, 3=L.Ear, 4=R.Ear,
     *  5=L.Shoulder, 6=R.Shoulder,
     *  7=L.Elbow, 8=R.Elbow, 9=L.Wrist, 10=R.Wrist,
     *  11=L.Hip, 12=R.Hip, 13=L.Knee, 14=R.Knee,
     *  15=L.Ankle, 16=R.Ankle
     *
     * Insight: Extremities (wrists, ankles) are usually noisier.
     *          Core (shoulders, hips) are more stable.
     *          Face (nose, eyes) varies by model confidence.
     */
    fun createPerJointFilters(): List<KeypointKalmanFilter> = listOf(
        // Head & Face (indices 0-4)
        KeypointKalmanFilter(0.00004f, 0.0008f),  // 0: Nose — stable, medium smoothing
        KeypointKalmanFilter(0.00006f, 0.0015f),  // 1: L.Eye — noisy, more smoothing
        KeypointKalmanFilter(0.00006f, 0.0015f),  // 2: R.Eye
        KeypointKalmanFilter(0.00008f, 0.002f),   // 3: L.Ear — can be unstable
        KeypointKalmanFilter(0.00008f, 0.002f),   // 4: R.Ear

        // Shoulders (indices 5-6) — very stable
        KeypointKalmanFilter(0.00003f, 0.0005f),  // 5: L.Shoulder — snappy
        KeypointKalmanFilter(0.00003f, 0.0005f),  // 6: R.Shoulder

        // Elbows (indices 7-8) — medium stability
        KeypointKalmanFilter(0.00005f, 0.001f),   // 7: L.Elbow
        KeypointKalmanFilter(0.00005f, 0.001f),   // 8: R.Elbow

        // Wrists (indices 9-10) — VERY NOISY, heavy smoothing
        KeypointKalmanFilter(0.0001f, 0.002f),    // 9: L.Wrist — extra smoothing
        KeypointKalmanFilter(0.0001f, 0.002f),    // 10: R.Wrist

        // Hips (indices 11-12) — very stable
        KeypointKalmanFilter(0.00003f, 0.0005f),  // 11: L.Hip
        KeypointKalmanFilter(0.00003f, 0.0005f),  // 12: R.Hip

        // Knees (indices 13-14) — medium stability
        KeypointKalmanFilter(0.00005f, 0.001f),   // 13: L.Knee
        KeypointKalmanFilter(0.00005f, 0.001f),   // 14: R.Knee

        // Ankles (indices 15-16) — noisy, heavy smoothing
        KeypointKalmanFilter(0.00008f, 0.0015f),  // 15: L.Ankle
        KeypointKalmanFilter(0.00008f, 0.0015f)   // 16: R.Ankle
    )

    /**
     * Use this if:
     * - You notice wrists/ankles jitter more than shoulders
     * - You want to fine-tune stability per joint
     * - You have good understanding of your Roboflow's per-joint accuracy
     */

    // ════════════════════════════════════════════════════════════════════════
    // PRESET 5: LOW-LATENCY (Minimum delay, responsive to motion)
    // ════════════════════════════════════════════════════════════════════════
    val LOW_LATENCY_PROCESS_NOISE = 0.00015f
    val LOW_LATENCY_MEASUREMENT_NOISE = 0.0003f
    val LOW_LATENCY_BUFFER_DELAY_MS = 350L

    /**
     * Use this if:
     * - You have very stable, fast connection (< 100ms RTT)
     * - You need real-time feedback (coaching during live demo)
     * - You're willing to accept visible jitter between frames
     * - Kalman will smooth it out over 2-3 frames
     *
     * WARNING: If RTT is actually 200+ms, buffer will run dry and skeleton freezes.
     */
    fun createLowLatencyFilters(keypointCount: Int): List<KeypointKalmanFilter> =
        List(keypointCount) {
            KeypointKalmanFilter(LOW_LATENCY_PROCESS_NOISE, LOW_LATENCY_MEASUREMENT_NOISE)
        }

    // ════════════════════════════════════════════════════════════════════════
    // Interpolation Confidence Thresholds
    // ════════════════════════════════════════════════════════════════════════

    data class InterpolationConfig(
        val highConfThreshold: Float = 0.7f,      // >= this: smooth lerp
        val mediumConfThreshold: Float = 0.3f,    // between this and high: dampened lerp
        // < medium: snap to nearest
        val dampedAlphaFactor: Float = 0.6f       // medium conf joints move 60% of normal
    )

    val INTERP_BALANCED = InterpolationConfig(
        highConfThreshold = 0.7f,
        mediumConfThreshold = 0.3f,
        dampedAlphaFactor = 0.6f
    )

    val INTERP_SMOOTH = InterpolationConfig(
        highConfThreshold = 0.75f,
        mediumConfThreshold = 0.4f,
        dampedAlphaFactor = 0.5f
    )

    val INTERP_RESPONSIVE = InterpolationConfig(
        highConfThreshold = 0.6f,
        mediumConfThreshold = 0.25f,
        dampedAlphaFactor = 0.8f
    )

    // ════════════════════════════════════════════════════════════════════════
    // Quick Selection: Match to Your Network
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Call this based on your measured Roboflow RTT (round-trip time in ms).
     * RTT = time from sending frame to receiving detection
     *
     * To measure:
     * - Look at logcat: "Sending frame to Roboflow… (X chars, WxH)"
     * - Wait for response log and note the time difference
     * - Average over 5-10 frames
     */
    fun selectByNetworkLatency(measuredRttMs: Long): Pair<List<KeypointKalmanFilter>, Long> {
        return when {
            measuredRttMs < 200 -> {
                // Very fast: use snappy config
                Pair(createSnappyFilters(17), SNAPPY_BUFFER_DELAY_MS)
            }
            measuredRttMs < 400 -> {
                // Good: use balanced config
                Pair(createBalancedFilters(17), BALANCED_BUFFER_DELAY_MS)
            }
            measuredRttMs < 600 -> {
                // Moderate: use buttery config
                Pair(createButteryFilters(17), BUTTERY_BUFFER_DELAY_MS)
            }
            else -> {
                // Slow: use buttery + longer buffer
                Pair(
                    createButteryFilters(17),
                    (BUTTERY_BUFFER_DELAY_MS + (measuredRttMs - 600) / 2).toLong()
                )
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HOW TO USE IN PracticeSession.kt
// ════════════════════════════════════════════════════════════════════════════

/*
In parseRoboflowResponse(), replace:

    val filters = if (prev != null) {
        prev.kalmanFilters
    } else {
        List(raw.keypoints.size) { KeypointKalmanFilter() }
    }

With ONE of these:

// ─────────────────────────────────────────────────────────────────────
// Option 1: Use preset based on your network
// ─────────────────────────────────────────────────────────────────────
    val filters = if (prev != null) {
        prev.kalmanFilters
    } else {
        KeypointSmootherConfig.createBalancedFilters(raw.keypoints.size)
    }

// ─────────────────────────────────────────────────────────────────────
// Option 2: Use per-joint customization
// ─────────────────────────────────────────────────────────────────────
    val filters = if (prev != null) {
        prev.kalmanFilters
    } else {
        KeypointSmootherConfig.createPerJointFilters()
    }

// ─────────────────────────────────────────────────────────────────────
// Option 3: Auto-detect based on measured RTT (advanced)
// ─────────────────────────────────────────────────────────────────────
    val filters = if (prev != null) {
        prev.kalmanFilters
    } else {
        // Measure your RTT first, then pass it in
        val (filtersForRtt, bufferDelay) = KeypointSmootherConfig.selectByNetworkLatency(350L)
        // Update LIVE_BUFFER_DELAY_MS = bufferDelay
        filtersForRtt
    }

Also update the buffer delay:

    private val LIVE_BUFFER_DELAY_MS = KeypointSmootherConfig.BALANCED_BUFFER_DELAY_MS

*/