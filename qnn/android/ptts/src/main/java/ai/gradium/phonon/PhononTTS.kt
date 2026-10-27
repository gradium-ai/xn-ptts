package ai.gradium.phonon

import android.content.Context
import android.os.Build
import java.io.File

/** A reusable Snapdragon NPU model. Load and speak off the main thread. */
class PhononTTS internal constructor(private var handle: Long, private val native: NativeEngine) : AutoCloseable {
    private val lock = Any()
    private var speaking = false
    private var voice: String? = null

    /** This model's output rate, in Hz. Audio is mono float PCM. */
    val sampleRate: Int = native.sampleRate(handle)
    val voices: List<String> = native.voices(handle).toList()

    fun setVoice(name: String) = synchronized(lock) {
        idle()
        require(name in voices) { "Unknown voice: $name" }
        voice = name
    }

    /**
     * Blocks while streaming PCM. Return false from [onAudio] to cancel.
     * Callback exceptions propagate to the caller. Callbacks may call [stop], but must not
     * call speak, setVoice or close. Reuse the model after completion or cancellation.
     */
    fun speak(text: String, onAudio: (FloatArray) -> Boolean): Result {
        require('\u0000' !in text) { "Text cannot contain NUL" }
        val selected = synchronized(lock) {
            idle()
            native.resetStop(handle)
            speaking = true
            voice
        }
        try {
            return native.speak(handle, text, selected, AudioCallback(onAudio))
        } finally {
            synchronized(lock) { speaking = false }
        }
    }

    /** May be called from another thread. Stops between inference calls. */
    fun stop() = synchronized(lock) {
        check(handle != 0L) { "PhononTTS is closed" }
        if (speaking) native.stop(handle)
    }

    /** Stop and join an active speak call before closing. */
    override fun close() = synchronized(lock) {
        if (handle != 0L) {
            check(!speaking) { "Stop and wait for speak to return before closing" }
            native.close(handle)
            handle = 0
        }
    }

    private fun idle() {
        check(handle != 0L) { "PhononTTS is closed" }
        check(!speaking) { "A speak call is already running" }
    }

    data class Result(val frames: Int, val samples: Long, val ttfaMs: Double, val totalMs: Double, val cancelled: Boolean)

    companion object {
        /**
         * Load an exported QNN bundle from [modelDir], using an explicit normalization [lang].
         * Supports en, fr, de, es, pt and none. This engine uses QNN HTP and has no CPU fallback.
         * The bundle must declare this phone's Build.SOC_MODEL in its soc_models metadata.
         */
        @JvmStatic
        fun load(context: Context, modelDir: File, lang: String): PhononTTS {
            require(lang in setOf("en", "fr", "de", "es", "pt", "none")) { "Unsupported language: $lang" }
            val libraries = File(context.applicationInfo.nativeLibraryDir)
            check(File(libraries, "libQnnHtp.so").isFile) {
                "QNN needs extracted native libraries. Set android.packaging.jniLibs.useLegacyPackaging = true in the app."
            }
            val engine = QnnNative()
            val handle = engine.create(modelDir.canonicalPath, lang, Build.SOC_MODEL.uppercase(java.util.Locale.ROOT), libraries.absolutePath)
            try {
                return PhononTTS(handle, engine)
            } catch (e: Throwable) {
                engine.close(handle)
                throw e
            }
        }
    }
}

internal fun interface AudioCallback { fun onAudio(pcm: FloatArray): Boolean }
internal interface NativeEngine {
    fun sampleRate(handle: Long): Int
    fun voices(handle: Long): Array<String>
    fun resetStop(handle: Long)
    fun stop(handle: Long)
    fun speak(handle: Long, text: String, voice: String?, onAudio: AudioCallback): PhononTTS.Result
    fun close(handle: Long)
}
internal class QnnNative : NativeEngine {
    init { System.loadLibrary("ptts_qnn") }
    external fun create(bundle: String, lang: String, soc: String, libraries: String): Long
    external override fun sampleRate(handle: Long): Int
    external override fun voices(handle: Long): Array<String>
    external override fun resetStop(handle: Long)
    external override fun stop(handle: Long)
    external override fun speak(handle: Long, text: String, voice: String?, onAudio: AudioCallback): PhononTTS.Result
    external override fun close(handle: Long)
}
