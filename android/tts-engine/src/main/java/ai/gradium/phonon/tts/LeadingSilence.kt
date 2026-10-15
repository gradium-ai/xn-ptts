package ai.gradium.phonon.tts

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Drops the near-silence an utterance starts with, so the first sound is heard as soon as it is
 * made rather than after a stretch of quiet. Audio is held back until a 10 ms window is loud
 * enough to be speech, then played from a little before it. At most [maxMs] is dropped, so a
 * quiet start still plays. Only the start of an utterance is touched.
 */
class LeadingSilence(sampleRate: Int, maxMs: Int = 300) {
    private val window = sampleRate / 100
    private val preRoll = sampleRate * PRE_ROLL_MS / 1000
    private val limit = sampleRate * maxMs / 1000
    private var held = FloatArray(0)
    private var scanned = 0
    private var done = false

    /** The part of [pcm] to play now, possibly empty. */
    fun trim(pcm: FloatArray): FloatArray {
        if (done) return pcm
        held += pcm
        while (scanned + window <= held.size) {
            if (scanned >= limit || rms(held, scanned, window) >= THRESHOLD) {
                done = true
                val out = held.copyOfRange(max(0, scanned - preRoll), held.size)
                held = FloatArray(0)
                return out
            }
            scanned += window
        }
        return FloatArray(0)
    }

    /** What is still held back once the utterance ends: all of it was quiet, so it is short. */
    fun flush(): FloatArray = held.also { held = FloatArray(0) }

    private fun rms(x: FloatArray, from: Int, n: Int): Float {
        var sum = 0f
        for (i in from until from + n) sum += x[i] * x[i]
        return sqrt(sum / n)
    }

    private companion object {
        /** -40 dBFS. */
        const val THRESHOLD = 0.01f

        /** Kept before the first loud window so the onset is not clipped. */
        const val PRE_ROLL_MS = 20
    }
}
