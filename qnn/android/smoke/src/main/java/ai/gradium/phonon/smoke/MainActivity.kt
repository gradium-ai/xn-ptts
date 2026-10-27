package ai.gradium.phonon.smoke

import android.app.Activity
import android.os.Bundle
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import ai.gradium.phonon.PhononTTS
import java.io.File

/** Installed-app check, including the minified release. Supply a bundle in externalFilesDir. */
class MainActivity : Activity() {
    @Volatile private var tts: PhononTTS? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this).apply { text = "Put the exported model in ${getExternalFilesDir(null)}/model, then speak." }
        val speak = Button(this).apply { text = "Speak" }
        val stop = Button(this).apply { text = "Stop"; setOnClickListener { tts?.stop() } }
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(status); addView(speak); addView(stop) }
        setContentView(layout)
        if (intent.getBooleanExtra("probe", false)) {
            Thread {
                try {
                    PhononTTS.load(this, File(filesDir, "missing-model"), "none").close()
                    error("Missing model unexpectedly loaded")
                } catch (e: IllegalStateException) {
                    if (e.message?.contains("metadata.json") == true) android.util.Log.i("PhononSmoke", "JNI_PROBE_OK")
                    else android.util.Log.e("PhononSmoke", "JNI_PROBE_FAILED", e)
                } catch (e: Throwable) { android.util.Log.e("PhononSmoke", "JNI_PROBE_FAILED", e) }
            }.start()
        }
        speak.setOnClickListener {
            speak.isEnabled = false
            Thread {
                try {
                    val model = PhononTTS.load(this, File(getExternalFilesDir(null), "model"), "en")
                    model.use {
                        tts = model
                        val format = AudioFormat.Builder().setSampleRate(model.sampleRate).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()
                        val track = AudioTrack.Builder().setAudioFormat(format).setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build()).setBufferSizeInBytes(AudioTrack.getMinBufferSize(model.sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)).build()
                        try {
                            track.play()
                            val result = model.speak("Hello from Phonon on the Snapdragon NPU.") { pcm ->
                                var offset = 0
                                while (offset < pcm.size) {
                                    val written = track.write(pcm, offset, pcm.size - offset, AudioTrack.WRITE_BLOCKING)
                                    check(written > 0) { "AudioTrack write failed: $written" }
                                    offset += written
                                }
                                true
                            }
                            runOnUiThread { status.text = "QNN HTP: $result" }
                            // Let the last queued frame play before releasing AudioTrack.
                            val deadline = android.os.SystemClock.elapsedRealtime() + 2000
                            while (!result.cancelled && track.playbackHeadPosition.toLong() < result.samples &&
                                !isFinishing && !isDestroyed && android.os.SystemClock.elapsedRealtime() < deadline) {
                                Thread.sleep(10)
                            }
                        } finally { track.release(); tts = null }
                    }
                } catch (e: Exception) {
                    runOnUiThread { status.text = e.toString() }
                } finally { runOnUiThread { speak.isEnabled = true } }
            }.start()
        }
    }
    override fun onDestroy() { tts?.stop(); super.onDestroy() }
}
