package ai.gradium.phonon

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class PhononTTSTest {
    private class Engine : NativeEngine {
        var stopped = false
        var closes = 0
        override fun sampleRate(handle: Long) = 24000
        override fun voices(handle: Long) = arrayOf("default", "test")
        override fun resetStop(handle: Long) { stopped = false }
        override fun stop(handle: Long) { stopped = true }
        override fun close(handle: Long) { closes++ }
        override fun speak(handle: Long, text: String, voice: String?, onAudio: AudioCallback): PhononTTS.Result {
            val keep = onAudio.onAudio(floatArrayOf(0.1f, 0.2f))
            return PhononTTS.Result(1, 2, 1.0, 2.0, stopped || !keep)
        }
    }
    @Test fun callbackCancellationAndReuse() {
        val engine = Engine()
        PhononTTS(1, engine).use { model ->
            assertTrue(model.speak("first") { false }.cancelled)
            assertFalse(model.speak("second") { true }.cancelled)
            assertEquals(24000, model.sampleRate)
            model.setVoice("test")
            assertThrows(IllegalArgumentException::class.java) { model.setVoice("missing") }
        }
        assertEquals(1, engine.closes)
    }
    @Test fun callbackErrorAndReentrantCallsLeaveModelReusable() {
        val model = PhononTTS(1, Engine())
        val failure = IllegalArgumentException("callback")
        assertSame(failure, assertThrows(IllegalArgumentException::class.java) {
            model.speak("first") { throw failure }
        })
        model.speak("second") {
            assertEquals(24000, model.sampleRate)
            assertEquals(listOf("default", "test"), model.voices)
            assertThrows(IllegalStateException::class.java) { model.close() }
            assertThrows(IllegalStateException::class.java) { model.speak("nested") { true } }
            model.stop()
            true
        }.also { assertTrue(it.cancelled) }
        assertFalse(model.speak("third") { true }.cancelled)
        model.close()
        model.close()
        assertThrows(IllegalStateException::class.java) { model.speak("closed") { true } }
        assertThrows(IllegalArgumentException::class.java) { model.speak("NUL\u0000") { true } }
    }
    @Test fun anotherThreadCanStopButCannotCloseAnActiveCall() {
        val model = PhononTTS(1, Engine())
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        var result: PhononTTS.Result? = null
        val worker = thread {
            result = model.speak("hello") { entered.countDown(); check(finish.await(5, TimeUnit.SECONDS)); true }
        }
        assertTrue(entered.await(5, TimeUnit.SECONDS))
        try {
            model.stop()
            assertThrows(IllegalStateException::class.java) { model.close() }
        } finally { finish.countDown(); worker.join(5000) }
        assertFalse(worker.isAlive)
        assertTrue(result!!.cancelled)
        model.close()
    }
}
