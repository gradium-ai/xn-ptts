package ai.gradium.phonon.tts

import ai.gradium.phonon.PhononTTS
import android.content.Context
import android.system.Os
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit

/**
 * The installed checkpoint and the model loaded from it, shared by the service and the settings
 * screen, which run in one process. The model is loaded once, off the main thread, and kept.
 */
object Model {
    private const val TAG = "PhononModel"

    /** The languages the runtime normalizes text for. */
    val SUPPORTED = listOf("en", "fr", "de", "es", "pt")

    /** Loaded models. Calls into one must hold its lock: the library takes one call at a time. */
    class Loaded(val tts: PhononTTS, val speakers: List<String>, val languages: List<String>) {
        /** The voice spoken when a request names none: `default` if the checkpoint has one. */
        val defaultSpeaker = if ("default" in speakers) "default" else speakers.firstOrNull()
    }

    sealed class State {
        object NotInstalled : State()
        object Loading : State()
        class Ready(val model: Loaded) : State()
        class Failed(val message: String) : State()
    }

    @Volatile var state: State = State.NotInstalled
        private set

    private val loader = Executors.newSingleThreadExecutor()
    private var pending: Future<*>? = null

    fun dir(context: Context) = File(context.filesDir, "model")

    /** Whether a checkpoint folder is in place. It may still fail to load. */
    fun installed(context: Context): Boolean {
        val dir = dir(context)
        return File(dir, "config.json").isFile && File(dir, "tokenizer.json").isFile &&
            (File(dir, "model.q8.gguf").isFile || File(dir, "model.safetensors").isFile)
    }

    /**
     * The languages the installed checkpoint speaks, read from its config without loading it.
     * A config that lists none is taken to speak them all.
     */
    fun languages(context: Context): List<String> {
        if (!installed(context)) return emptyList()
        val listed = try {
            JSONObject(File(dir(context), "config.json").readText()).optJSONArray("languages")
        } catch (e: Exception) {
            null
        }
        val names = (0 until (listed?.length() ?: 0)).map { listed!!.getString(it) }
        return if (names.isEmpty()) SUPPORTED else SUPPORTED.filter { it in names }
    }

    /** Start loading the installed checkpoint, unless it is loaded or loading already. */
    @Synchronized
    fun load(context: Context): Future<*>? {
        if (state is State.Ready || pending?.isDone == false) return pending
        if (!installed(context)) {
            state = State.NotInstalled
            return null
        }
        val dir = dir(context).path
        val languages = languages(context)
        state = State.Loading
        pending = loader.submit {
            state = try {
                // Before the first load: the library's thread pool reads it once. One thread per
                // core is too many on phones that mix fast and slow cores. A value already set wins.
                Os.setenv("RAYON_NUM_THREADS", THREADS.toString(), false)
                // Text is normalized per request with setLanguage; this is only the first one.
                val tts = PhononTTS(dir, languages.firstOrNull() ?: "none")
                State.Ready(Loaded(tts, tts.voices, languages))
            } catch (t: Throwable) {
                Log.e(TAG, "loading $dir failed", t)
                State.Failed(t.message ?: t.toString())
            }
        }
        return pending
    }

    /** The loaded model, waiting for a load in progress. Null when none could be loaded. */
    fun await(context: Context): Loaded? {
        try {
            load(context)?.get(LOAD_TIMEOUT_S, TimeUnit.SECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "waiting for the model failed", e)
        }
        return (state as? State.Ready)?.model
    }

    /**
     * Replace the installed checkpoint with what [fill] writes into an empty folder, then load
     * it. [fill] runs on the calling thread, which must not be the main one. This is the one way
     * in: a download from a model hub would be another [fill].
     */
    fun install(context: Context, fill: (File) -> Unit) {
        val staging = File(context.filesDir, "model.staging")
        staging.deleteRecursively()
        staging.mkdirs()
        try {
            fill(staging)
            for (name in listOf("config.json", "tokenizer.json")) {
                require(File(staging, name).isFile) { "the folder has no $name" }
            }
            require(File(staging, "model.q8.gguf").isFile || File(staging, "model.safetensors").isFile) {
                "the folder has no model.q8.gguf or model.safetensors"
            }
            // Held across the swap, so a load() from the service cannot start on the old folder
            // between unloading it and replacing it.
            synchronized(this) {
                unload()
                dir(context).deleteRecursively()
                check(staging.renameTo(dir(context))) { "could not move the checkpoint into place" }
            }
        } finally {
            staging.deleteRecursively()
        }
        load(context)
    }

    /** Free the loaded model, once whatever is speaking with it has finished. */
    @Synchronized
    private fun unload() {
        pending?.get()
        (state as? State.Ready)?.model?.let { synchronized(it) { it.tts.close() } }
        state = State.NotInstalled
        pending = null
    }

    /** A language as Android names it, two or three letters, to one of [SUPPORTED], or null. */
    fun code(language: String?): String? {
        if (language.isNullOrEmpty()) return null
        val iso3 = when (language.lowercase(Locale.ROOT)) {
            "fre" -> "fra"
            "ger" -> "deu"
            else -> try {
                Locale.forLanguageTag(language).isO3Language
            } catch (e: Exception) {
                return null
            }
        }
        return SUPPORTED.firstOrNull { Locale.forLanguageTag(it).isO3Language == iso3 }
    }

    private const val THREADS = 2
    private const val LOAD_TIMEOUT_S = 60L
}
