package ai.gradium.phonon.smoke

import ai.gradium.phonon.PhononTTS
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class ModelTest {
    @Test fun nativeLibrariesAndJniLoadInsideAnInstalledApp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val libraries = File(context.applicationInfo.nativeLibraryDir)
        for (name in listOf("libptts_qnn.so", "libptts_text.so", "libQnnHtp.so", "libQnnSystem.so")) {
            assertTrue("Missing extracted $name", File(libraries, name).isFile)
        }
        val error = assertThrows(IllegalStateException::class.java) {
            PhononTTS.load(context, File(context.filesDir, "missing-model"), "none")
        }
        assertTrue(error.message!!.contains("metadata.json"))
    }
    @Test fun packagedRustFrontendLoadsAndDownloadedCodeIsIgnored() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bundle = File(context.cacheDir, "frontend-test").apply { mkdirs() }
        try {
            File(bundle, "bos.bin").writeBytes(byteArrayOf(0, 0))
            File(bundle, "tokenizer.json").writeText("""{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}""")
            File(bundle, "metadata.json").writeText("""{
              "generation":{"sample_rate":24000,"frame_rate":12.5,"cache_slots":32,
                "prefill_tokens":1,"max_tokens_per_chunk":1,"temperature":0.5,
                "eos_threshold":0,"min_frames_before_eos":0,
                "layout":{"layers":1,"heads":1,"head_dim":1},"bos":{"file":"bos.bin","shape":[1]}},
              "voices":[],"text":{"library":"downloaded-code-must-not-load.so","tokenizer":"tokenizer.json"},
              "runtime":{"context_binaries":{"test":{"file":"unused.bin","soc_models":[]}}}
            }""")
            val error = assertThrows(IllegalStateException::class.java) { PhononTTS.load(context, bundle, "none") }
            // Model selection follows successful packaged Rust frontend initialization.
            assertTrue(error.message, error.message!!.contains("no compiled QNN model"))
        } finally { bundle.deleteRecursively() }
    }
    @Test fun realNpuStreamingCancellationAndReuse() {
        val args = InstrumentationRegistry.getArguments()
        val path = args.getString("modelDir")
        assumeTrue("Pass -e modelDir with an exported QNN bundle on the phone", path != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        PhononTTS.load(context, File(path!!), args.getString("lang", "en")).use { model ->
            assertTrue(model.voices.isNotEmpty())
            var frames = 0
            val cancelled = model.speak("This sentence should stop after its first frame.") { pcm ->
                assertTrue(pcm.isNotEmpty()); assertTrue(pcm.all { it.isFinite() }); frames++; false
            }
            assertEquals(1, frames)
            assertTrue(cancelled.cancelled)
            val error = IllegalArgumentException("audio callback failed")
            assertSame(error, assertThrows(IllegalArgumentException::class.java) { model.speak("Hello.") { throw error } })
            val entered = CountDownLatch(1)
            val resume = CountDownLatch(1)
            val stopped = AtomicReference<PhononTTS.Result>()
            val failure = AtomicReference<Throwable>()
            val worker = Thread {
                try {
                    stopped.set(model.speak("This sentence is cancelled from another thread.") {
                        entered.countDown()
                        check(resume.await(10, TimeUnit.SECONDS)) { "Stop test timed out" }
                        true
                    })
                } catch (e: Throwable) { failure.set(e) }
            }
            worker.start()
            try {
                assertTrue("No audio callback", entered.await(30, TimeUnit.SECONDS))
                assertThrows(IllegalStateException::class.java) { model.close() }
                model.stop()
            } finally {
                resume.countDown()
                worker.join(30000)
            }
            assertFalse("Speak did not stop", worker.isAlive)
            failure.get()?.let { throw it }
            assertTrue(stopped.get().cancelled)
            val result = model.speak("Hello from Phonon.") { pcm -> assertTrue(pcm.all { it.isFinite() }); true }
            assertFalse(result.cancelled)
            assertTrue(result.samples > 0)
            assertTrue(result.ttfaMs >= 0)
            android.util.Log.i("PhononTest", "QNN HTP $result")
        }
    }
}
