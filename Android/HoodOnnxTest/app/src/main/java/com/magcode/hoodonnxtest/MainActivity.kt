package com.magcode.hoodonnxtest

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.TextView
import ai.onnxruntime.OrtEnvironment
import java.io.File

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val statusView = TextView(this)
        statusView.text = "Running ONNX Runtime pipeline..."
        statusView.movementMethod = ScrollingMovementMethod()
        statusView.setTextIsSelectable(true)
        setContentView(statusView)

        Thread {
            try {
                val result = runAll()
                runOnUiThread { statusView.text = result }
            } catch (t: Throwable) {
                Log.e("HoodOnnxTest", "Run failed", t)
                writeError(t)
                runOnUiThread { statusView.text = "FAILED: ${t.message}" }
            }
        }.start()
    }

    private fun runAll(): String {
        val env = OrtEnvironment.getEnvironment()
        val reportLines = mutableListOf<String>()

        val aggOk = CpuScatterSum.selfTest()
        val aggLine = "cpu_scatter_sum_selftest=$aggOk"
        reportLines.add(aggLine)
        Log.i("HoodOnnxTest", aggLine)
        if (!aggOk) {
            return "FAILED: cpu_scatter_sum_selftest=false"
        }

        val pipelineProfileDir = getExternalFilesDir(null) ?: filesDir
        val pipelineLine = HoodPipelineRunner.run(
            assets,
            filesDir,
            env,
            pipelineProfileDir
        )
        reportLines.add(pipelineLine)

        val reportText = reportLines.joinToString(separator = "\n")
        copyToClipboard(reportText)
        return reportText
    }

    private fun writeError(t: Throwable) {
        try {
            val errFile = File(getExternalFilesDir(null), "onnx_error.txt")
            errFile.writeText(t.stackTraceToString())
        } catch (_: Exception) {
            // ignore
        }
    }

    private fun copyToClipboard(text: String) {
        try {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("onnx_report", text)
            clipboard.setPrimaryClip(clip)
        } catch (_: Exception) {
            // ignore
        }
    }
}
