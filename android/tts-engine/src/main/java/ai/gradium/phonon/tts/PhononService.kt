package ai.gradium.phonon.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Phonon as a system text-to-speech engine. Android calls [onSynthesizeText] on one thread, one
 * request at a time, and [onStop] from another to cut the current one short.
 *
 * Android voices are named `<language>-<speaker>`: every checkpoint voice in every language the
 * checkpoint speaks.
 */
class PhononService : TextToSpeechService() {
    @Volatile private var stopped = false
    @Volatile private var language: String? = null

    override fun onCreate() {
        super.onCreate()
        Model.load(this)
    }

    // Answered from the installed files, without waiting for the model to load.
    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        val code = Model.code(lang) ?: return TextToSpeech.LANG_NOT_SUPPORTED
        if (!Model.installed(this)) return TextToSpeech.LANG_MISSING_DATA
        // The checkpoints have no regional variants, so a country never matches more closely.
        return if (code in Model.languages(this)) TextToSpeech.LANG_AVAILABLE else TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val result = onIsLanguageAvailable(lang, country, variant)
        if (result >= TextToSpeech.LANG_AVAILABLE) language = Model.code(lang)
        return result
    }

    override fun onGetLanguage(): Array<String> {
        val code = language ?: Model.languages(this).firstOrNull() ?: return arrayOf("", "", "")
        return arrayOf(Locale.forLanguageTag(code).isO3Language, "", "")
    }

    override fun onGetVoices(): List<Voice> {
        val model = Model.await(this) ?: return emptyList()
        return model.languages.flatMap { lang -> speakers(model).map { voice(lang, it) } }
    }

    override fun onIsValidVoiceName(voiceName: String?): Int =
        if (parse(voiceName) != null) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String?): Int = onIsValidVoiceName(voiceName)

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String? {
        val code = Model.code(lang) ?: return null
        val model = Model.await(this) ?: return null
        if (code !in model.languages) return null
        return name(code, model.defaultSpeaker ?: NO_SPEAKER)
    }

    override fun onStop() {
        stopped = true
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopped = false
        val model = Model.await(this)
        if (model == null) {
            fail(callback, TextToSpeech.ERROR_NOT_INSTALLED_YET)
            return
        }
        // A voice the request names wins; otherwise its language, spoken by the default voice.
        val (lang, speaker) = parse(request.voiceName)
            ?: Model.code(request.language)?.takeIf { it in model.languages }?.let { it to model.defaultSpeaker }
            ?: run {
                fail(callback, TextToSpeech.ERROR_INVALID_REQUEST)
                return
            }
        val text = request.charSequenceText?.toString().orEmpty()
        if (callback.start(SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return
        if (text.isBlank()) {
            callback.done()
            return
        }
        val max = callback.maxBufferSize
        // Whether Android still takes audio: it stops taking it once this request is stopped.
        fun send(pcm: FloatArray): Boolean {
            val bytes = pcm16(pcm)
            var at = 0
            while (at < bytes.size) {
                val n = minOf(max, bytes.size - at)
                if (callback.audioAvailable(bytes, at, n) != TextToSpeech.SUCCESS) return false
                at += n
            }
            return true
        }
        try {
            synchronized(model) {
                model.tts.setLanguage(lang)
                speaker?.let { model.tts.setVoice(it) }
                model.tts.speak(text) { pcm -> !stopped && send(pcm) }
            }
            callback.done()
        } catch (e: Exception) {
            Log.e(TAG, "synthesis failed", e)
            fail(callback, TextToSpeech.ERROR_SYNTHESIS)
        }
    }

    /** Android only tells the app about an error once the request is done. */
    private fun fail(callback: SynthesisCallback, code: Int) {
        callback.error(code)
        callback.done()
    }

    /** The speakers to offer: the checkpoint's voices, or one stand-in when it has none. */
    private fun speakers(model: Model.Loaded) = model.speakers.ifEmpty { listOf(NO_SPEAKER) }

    private fun name(lang: String, speaker: String) = "$lang-$speaker"

    private fun voice(lang: String, speaker: String) = Voice(
        name(lang, speaker), Locale.forLanguageTag(lang), Voice.QUALITY_HIGH, Voice.LATENCY_LOW, false, emptySet(),
    )

    /** A voice name as [name] writes it, to its language and speaker, or null if not ours. */
    private fun parse(voiceName: String?): Pair<String, String?>? {
        val (lang, speaker) = voiceName?.split('-', limit = 2)?.takeIf { it.size == 2 } ?: return null
        val model = Model.await(this) ?: return null
        if (lang !in model.languages || speaker !in speakers(model)) return null
        return lang to speaker.takeIf { it in model.speakers }
    }

    private companion object {
        const val TAG = "PhononService"

        /** What ptts.h hands out: 24 kHz mono. */
        const val SAMPLE_RATE = 24000

        /** The voice name's speaker part for a checkpoint without voices. */
        const val NO_SPEAKER = "default"

        /** Floats in [-1, 1] to little-endian 16-bit PCM. */
        fun pcm16(pcm: FloatArray): ByteArray {
            val out = ByteArray(pcm.size * 2)
            for (i in pcm.indices) {
                val s = (pcm[i].coerceIn(-1f, 1f) * 32767f).roundToInt()
                out[2 * i] = s.toByte()
                out[2 * i + 1] = (s shr 8).toByte()
            }
            return out
        }
    }
}
