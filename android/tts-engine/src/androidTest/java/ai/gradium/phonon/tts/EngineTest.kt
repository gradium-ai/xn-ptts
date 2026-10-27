package ai.gradium.phonon.tts

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The engine as any app reaches it: through [TextToSpeech], bound to this package. Most tests
 * need a checkpoint installed in the app first (see README.md) and are skipped without one;
 * [withoutModel] is the one that runs only then. Each WAV a test writes is left in the app's
 * files folder, to listen to or transcribe.
 */
class EngineTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var tts: TextToSpeech
    private val events = ConcurrentHashMap<String, CountDownLatch>()

    private fun latch(event: String) = events.getOrPut(event) { CountDownLatch(1) }

    @Before
    fun bind() {
        val ready = CountDownLatch(1)
        var status = TextToSpeech.ERROR
        tts = TextToSpeech(context, { status = it; ready.countDown() }, context.packageName)
        assertTrue(ready.await(30, TimeUnit.SECONDS))
        assertEquals(TextToSpeech.SUCCESS, status)
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String) = latch("start:$id").countDown()
            override fun onDone(id: String) = latch("done:$id").countDown()
            override fun onStop(id: String, interrupted: Boolean) = latch("stop:$id").countDown()
            @Suppress("OVERRIDE_DEPRECATION")
            override fun onError(id: String) = latch("error:$id").countDown()
        })
    }

    @After
    fun unbind() {
        if (::tts.isInitialized) tts.shutdown()
    }

    @Test
    fun withoutModel() {
        assumeTrue("a checkpoint is installed", !Model.installed(context))
        assertEquals(TextToSpeech.LANG_MISSING_DATA, tts.isLanguageAvailable(Locale.ENGLISH))
        assertTrue(tts.voices.isNullOrEmpty())
        val file = File(context.filesDir, "test-none.wav")
        assertEquals(TextToSpeech.SUCCESS, tts.synthesizeToFile(SHORT, Bundle(), file, "none"))
        assertTrue("the error was not reported", latch("error:none").await(30, TimeUnit.SECONDS))
    }

    @Test
    fun listsVoicesForEachLanguage() {
        assumeInstalled()
        val voices = tts.voices.filter { it.locale.language in Model.SUPPORTED }
        assertTrue(voices.isNotEmpty())
        val spoken = Locale.forLanguageTag(Model.languages(context).first())
        assertEquals(TextToSpeech.LANG_AVAILABLE, tts.setLanguage(spoken))
        assertEquals(TextToSpeech.LANG_NOT_SUPPORTED, tts.isLanguageAvailable(Locale.JAPANESE))
    }

    @Test
    fun writesEnglish() {
        assumeSpeaks("en")
        write(Locale.ENGLISH, "en", ENGLISH)
    }

    @Test
    fun writesFrench() {
        assumeSpeaks("fr")
        write(Locale.FRENCH, "fr", FRENCH)
    }

    @Test
    fun writesEveryEnglishVoice() {
        assumeInstalled()
        for (voice in tts.voices.filter { it.locale.language == "en" }) {
            assertEquals(TextToSpeech.SUCCESS, tts.setVoice(voice))
            val file = File(context.filesDir, "test-${voice.name}.wav")
            synthesize(SHORT, file, voice.name)
        }
    }

    @Test
    fun flushStopsLongSpeech() {
        assumeSpeaks("en")
        tts.setLanguage(Locale.ENGLISH)
        tts.speak(LONG, TextToSpeech.QUEUE_FLUSH, Bundle(), "long")
        assertTrue("never started", latch("start:long").await(60, TimeUnit.SECONDS))
        Thread.sleep(1000)
        val flushed = System.nanoTime()
        tts.speak(SHORT, TextToSpeech.QUEUE_FLUSH, Bundle(), "short")
        assertTrue("long was not stopped", latch("stop:long").await(10, TimeUnit.SECONDS))
        assertTrue("short never started", latch("start:short").await(10, TimeUnit.SECONDS))
        val ms = (System.nanoTime() - flushed) / 1_000_000
        println("PHONON_TEST flush to next start: $ms ms")
        assertTrue("short never finished", latch("done:short").await(60, TimeUnit.SECONDS))
    }

    private fun assumeInstalled() = assumeTrue("no checkpoint installed", Model.installed(context))

    private fun assumeSpeaks(lang: String) {
        assumeInstalled()
        assumeTrue("the checkpoint does not speak $lang", lang in Model.languages(context))
    }

    private fun write(locale: Locale, tag: String, text: String) {
        assertEquals(TextToSpeech.LANG_AVAILABLE, tts.setLanguage(locale))
        synthesize(text, File(context.filesDir, "test-$tag.wav"), tag)
    }

    private fun synthesize(text: String, file: File, id: String) {
        file.delete()
        assertEquals(TextToSpeech.SUCCESS, tts.synthesizeToFile(text, Bundle(), file, id))
        assertTrue("$id did not finish", latch("done:$id").await(120, TimeUnit.SECONDS))
        assertEquals(1L, latch("error:$id").count)
        // A WAV header and at least half a second of 24 kHz 16-bit audio.
        assertTrue("${file.length()} bytes", file.length() > 44 + 24000)
    }

    private companion object {
        const val SHORT = "Hello. This is the system voice."
        const val ENGLISH = "The quick brown fox jumps over the lazy dog. It was 3:45 on May 2nd."
        const val FRENCH = "Bonjour à tous. Il fait beau aujourd'hui, et nous partons à la plage."
        val LONG = List(8) {
            "This is a long passage that keeps the engine busy, so that a flush has something to cut short."
        }.joinToString(" ")
    }
}
