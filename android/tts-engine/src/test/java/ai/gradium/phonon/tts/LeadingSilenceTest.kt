package ai.gradium.phonon.tts

import org.junit.Assert.assertEquals
import org.junit.Test

class LeadingSilenceTest {
    private val rate = 1000 // 10 samples per window, 20 of pre-roll, 300 at most dropped

    private fun quiet(n: Int) = FloatArray(n) { 0.001f }

    private fun loud(n: Int) = FloatArray(n) { 0.5f }

    @Test
    fun playsFromJustBeforeTheFirstLoudWindow() {
        val trim = LeadingSilence(rate)
        assertEquals(0, trim.trim(quiet(70)).size)
        // The onset is at 100, in the second chunk; play from 80.
        val out = trim.trim(quiet(30) + loud(50))
        assertEquals(70, out.size)
        assertEquals(0.5f, out[20])
        // Everything after the onset passes untouched.
        assertEquals(40, trim.trim(quiet(40)).size)
    }

    @Test
    fun dropsAtMostTheLimit() {
        val trim = LeadingSilence(rate)
        val out = trim.trim(quiet(500))
        assertEquals(500 - 300 + 20, out.size)
    }

    @Test
    fun aQuietUtteranceIsFlushedAtTheEnd() {
        val trim = LeadingSilence(rate)
        assertEquals(0, trim.trim(quiet(55)).size)
        assertEquals(55, trim.flush().size)
    }
}
