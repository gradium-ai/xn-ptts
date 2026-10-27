package ai.gradium.phonon.tts

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.provider.DocumentsContract.Document
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import kotlin.concurrent.thread

/**
 * The engine's settings, which the system text-to-speech settings open: what is installed, and a
 * way to install a checkpoint folder from the phone's storage.
 */
class SettingsActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var importButton: Button
    @Volatile private var importing: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val title = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 24f
            setPadding(0, 0, 0, pad)
        }
        status = TextView(this).apply { setPadding(0, 0, 0, pad) }
        importButton = Button(this).apply {
            text = getString(R.string.import_folder)
            setOnClickListener { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK) }
        }
        val systemSettings = Button(this).apply {
            text = getString(R.string.system_settings)
            setOnClickListener { startActivity(Intent("com.android.settings.TTS_SETTINGS")) }
        }
        val help = TextView(this).apply {
            text = getString(R.string.import_help)
            setPadding(0, pad, 0, 0)
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            for (v in listOf(title, status, importButton, systemSettings, help)) addView(v, MATCH_PARENT, WRAP_CONTENT)
        }
        // Keeps clear of the status and navigation bars, which the app draws under.
        setContentView(ScrollView(this).apply { fitsSystemWindows = true; addView(column) })
        Model.load(this)
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val tree = data?.data
        if (requestCode != PICK || resultCode != RESULT_OK || tree == null) return
        importing = getString(R.string.importing)
        refresh()
        thread {
            val error = try {
                Model.install(this) { into -> copyCheckpoint(tree, into) }
                Model.await(this)
                null
            } catch (e: Exception) {
                e.message ?: e.toString()
            }
            importing = null
            runOnUiThread {
                refresh()
                if (error != null) status.text = getString(R.string.import_failed, error)
            }
        }
    }

    private fun refresh() {
        importButton.isEnabled = importing == null
        status.text = importing ?: when (val s = Model.state) {
            is Model.State.NotInstalled -> getString(R.string.not_installed)
            is Model.State.Loading -> getString(R.string.loading)
            is Model.State.Failed -> getString(R.string.load_failed, s.message)
            is Model.State.Ready -> getString(
                R.string.ready,
                s.model.languages.joinToString(", "),
                s.model.speakers.joinToString(", ").ifEmpty { "-" },
            )
        }
        if (Model.state is Model.State.Loading) status.postDelayed(::refresh, 500)
    }

    /** Copy the files a checkpoint folder needs from the picked folder into [into]. */
    private fun copyCheckpoint(tree: Uri, into: File) {
        val root = children(tree, DocumentsContract.getTreeDocumentId(tree))
        // The library loads the GGUF when both are there, so the other is not worth the space.
        val skip = if ("model.q8.gguf" in root) "model.safetensors" else null
        for ((name, entry) in root) {
            val (id, isDir) = entry
            when {
                !isDir && name in FILES && name != skip -> copy(tree, id, File(into, name))
                isDir && name in VOICE_DIRS -> for ((voice, child) in children(tree, id)) {
                    // The name comes from the document provider: keep it inside the voices folder.
                    if (!child.second && voice.endsWith(".safetensors") && '/' !in voice) {
                        copy(tree, child.first, File(into, "$name/$voice"))
                    }
                }
            }
        }
    }

    /** A folder's entries by name: each one's document id and whether it is a folder. */
    private fun children(tree: Uri, id: String): Map<String, Pair<String, Boolean>> {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, id)
        val columns = arrayOf(Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME, Document.COLUMN_MIME_TYPE)
        val out = mutableMapOf<String, Pair<String, Boolean>>()
        contentResolver.query(uri, columns, null, null, null)?.use {
            while (it.moveToNext()) out[it.getString(1)] = it.getString(0) to (it.getString(2) == Document.MIME_TYPE_DIR)
        }
        return out
    }

    private fun copy(tree: Uri, id: String, to: File) {
        importing = getString(R.string.copying, to.name)
        runOnUiThread(::refresh)
        to.parentFile?.mkdirs()
        val uri = DocumentsContract.buildDocumentUriUsingTree(tree, id)
        val input = contentResolver.openInputStream(uri) ?: error("cannot read ${to.name}")
        input.use { from -> to.outputStream().use { from.copyTo(it, 1 shl 20) } }
    }

    private companion object {
        const val PICK = 1

        /** What a checkpoint folder holds, as android/README.md lists it. */
        val FILES = setOf("config.json", "tokenizer.json", "model.q8.gguf", "model.safetensors", "default-voice.safetensors")
        val VOICE_DIRS = setOf("voices", "embeddings")
    }
}
