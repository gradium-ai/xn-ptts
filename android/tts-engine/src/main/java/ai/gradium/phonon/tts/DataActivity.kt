package ai.gradium.phonon.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import java.util.Locale

/**
 * The two questions the system settings ask an engine without binding to it: which languages
 * have their data installed, and what to say when the user taps "Play".
 */
class DataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        when (intent.action) {
            TextToSpeech.Engine.ACTION_CHECK_TTS_DATA -> {
                val installed = Model.languages(this)
                val result = Intent()
                    .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, iso3(installed))
                    .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, iso3(Model.SUPPORTED - installed.toSet()))
                val pass = if (installed.isEmpty()) TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL else TextToSpeech.Engine.CHECK_VOICE_DATA_PASS
                setResult(pass, result)
            }
            TextToSpeech.Engine.ACTION_GET_SAMPLE_TEXT -> {
                val sample = SAMPLES[Model.code(intent.getStringExtra("language"))]
                if (sample == null) {
                    setResult(TextToSpeech.LANG_NOT_SUPPORTED)
                } else {
                    setResult(TextToSpeech.LANG_AVAILABLE, Intent().putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, sample))
                }
            }
        }
        finish()
    }

    private companion object {
        /**
         * Each language as the settings name it: on its own, and with every country Android knows
         * for it, since the settings look the phone's locale up with its country.
         */
        fun iso3(codes: List<String>): ArrayList<String> {
            val out = LinkedHashSet<String>()
            for (code in codes) {
                val lang = Locale.forLanguageTag(code).isO3Language
                out += lang
                for (locale in Locale.getAvailableLocales()) {
                    if (locale.language != code || locale.variant.isNotEmpty()) continue
                    val country = try {
                        locale.isO3Country
                    } catch (e: Exception) {
                        ""
                    }
                    if (country.isNotEmpty()) out += "$lang-$country"
                }
            }
            return ArrayList(out)
        }

        val SAMPLES = mapOf(
            "en" to "This is an example of speech synthesis in English.",
            "fr" to "Ceci est un exemple de synthèse vocale en français.",
            "de" to "Dies ist ein Beispiel für Sprachsynthese auf Deutsch.",
            "es" to "Este es un ejemplo de síntesis de voz en español.",
            "pt" to "Este é um exemplo de síntese de voz em português.",
        )
    }
}
