package com.magcode.hoodonnxtest

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.text.method.ScrollingMovementMethod

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val statusView = TextView(this)
        statusView.text = "Running ONNX Runtime NNAPI tests..."
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
        val opts = OrtSession.SessionOptions()

        // Fail fast if NNAPI is not available
        opts.addNnapi()
        opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")
        // Verbose logging not available in this ORT Java API; rely on exception output.

        val tests = readTests()
        var maxDiffAll = 0.0f
        val reportLines = mutableListOf<String>()

        val aggOk = CpuScatterSum.selfTest()
        val aggLine = "cpu_scatter_sum_selftest=$aggOk"
        reportLines.add(aggLine)
        Log.i("HoodOnnxTest", aggLine)
        if (!aggOk) {
            return "FAILED: cpu_scatter_sum_selftest=false"
        }

        val probeLines = runProbes()
        if (probeLines.isNotEmpty()) {
            reportLines.add("=== NNAPI probe ===")
            reportLines.addAll(probeLines)
            reportLines.add("")
        }

        val pipelineLine = HoodPipelineRunner.run(assets, filesDir, env, opts)
        reportLines.add(pipelineLine)
        reportLines.add("")

        for (i in 0 until tests.length()) {
            val t = tests.getJSONObject(i)
            val name = t.getString("name")

            val modelPath = copyAssetToFile(t.getString("model"))
            val inputShape = readShapeJson(t.getString("input_shape"))
            val outputShape = readShapeJson(t.getString("output_shape"))

            val inputData = readFloatBinary(t.getString("input_bin"))
            val expectedData = readFloatBinary(t.getString("output_bin"))

            try {
                val session = env.createSession(modelPath.absolutePath, opts)
                val inputName = session.inputNames.first()
                val inputTensor = OnnxTensor.createTensor(env, toFloatBuffer(inputData), inputShape)

                val results = session.run(mapOf(inputName to inputTensor))
                val outputTensor = results[0] as OnnxTensor

                val output = outputTensor.floatBuffer
                output.rewind()

                var maxDiff = 0.0f
                for (j in expectedData.indices) {
                    val v = output.get()
                    val diff = kotlin.math.abs(v - expectedData[j])
                    if (diff > maxDiff) maxDiff = diff
                }

                if (maxDiff > maxDiffAll) maxDiffAll = maxDiff
                val line = "${i + 1}/${tests.length()} $name max_abs_diff=$maxDiff"
                reportLines.add(line)
                Log.i("HoodOnnxTest", line)
            } catch (t: Throwable) {
                val diag = runDiagnostics(name, modelPath, inputData, inputShape)
                val probeHeader = if (probeLines.isNotEmpty()) {
                    "=== NNAPI probe ===\n" + probeLines.joinToString(separator = "\n") + "\n\n"
                } else {
                    ""
                }
                val msg = "${probeHeader}FAILED at $name. $diag"
                Log.e("HoodOnnxTest", msg, t)
                return msg
            }
        }

        val reportText = "OK NNAPI. models=${tests.length()} max_abs_diff=$maxDiffAll\n\n" +
                reportLines.joinToString(separator = "\n")
        copyToClipboard(reportText)

        return reportText
    }

    private fun runProbes(): List<String> {
        val probes = readOptionalJsonArray("probes.json") ?: return emptyList()
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        opts.addNnapi()
        opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")

        val lines = mutableListOf<String>()
        for (i in 0 until probes.length()) {
            val p = probes.getJSONObject(i)
            val name = p.getString("name")
            val modelPath = copyAssetToFile(p.getString("model"))
            val inputShape = readShapeJson(p.getString("input_shape"))
            val inputData = readFloatBinary(p.getString("input_bin"))
            try {
                val session = env.createSession(modelPath.absolutePath, opts)
                val inputName = session.inputNames.first()
                val inputTensor = OnnxTensor.createTensor(env, toFloatBuffer(inputData), inputShape)
                session.run(mapOf(inputName to inputTensor)).close()
                session.close()
                val line = "OK $name"
                lines.add(line)
                Log.i("HoodOnnxTest", line)
            } catch (t: Throwable) {
                val diag = runDiagnostics(name, modelPath, inputData, inputShape)
                val line = "FAIL $name. $diag"
                lines.add(line)
                Log.e("HoodOnnxTest", line, t)
            }
        }
        return lines
    }

    private fun runDiagnostics(
        name: String,
        modelPath: File,
        inputData: FloatArray,
        inputShape: LongArray
    ): String {
        return try {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            opts.addNnapi()
            opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            opts.setSessionLogVerbosityLevel(1)
            // allow CPU fallback for diagnostics

            val safeName = name.replace('/', '_')
            val profileFile = File(getExternalFilesDir(null), "ort_profile_${safeName}.json")
            opts.enableProfiling(profileFile.absolutePath)

            val session = env.createSession(modelPath.absolutePath, opts)
            val inputName = session.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(env, toFloatBuffer(inputData), inputShape)
            session.run(mapOf(inputName to inputTensor)).close()
            val profilePath = session.endProfiling()
            session.close()

            val summary = summarizeProfile(profilePath)
            val summaryFile = File(getExternalFilesDir(null), "ort_profile_${safeName}_summary.txt")
            summaryFile.writeText(summary)

            "diagnostic profile at $profilePath\nsummary: $summary\nsummary file: ${summaryFile.absolutePath}"
        } catch (e: Throwable) {
            "diagnostic failed: ${e.message}"
        }
    }

    private fun toFloatBuffer(data: FloatArray): java.nio.FloatBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        val fb = bb.asFloatBuffer()
        fb.put(data)
        fb.rewind()
        return fb
    }

    private fun readTests(): JSONArray {
        val text = assets.open("tests.json").bufferedReader().use { it.readText() }
        return JSONArray(text)
    }

    private fun readOptionalJsonArray(name: String): JSONArray? {
        return try {
            val text = assets.open(name).bufferedReader().use { it.readText() }
            JSONArray(text)
        } catch (_: Exception) {
            null
        }
    }

    private fun readShapeJson(name: String): LongArray {
        val text = assets.open(name).bufferedReader().use { it.readText() }
        val arr = JSONArray(text)
        val out = LongArray(arr.length())
        for (i in 0 until arr.length()) {
            out[i] = arr.getLong(i)
        }
        return out
    }

    private fun readFloatBinary(name: String): FloatArray {
        val bytes = assets.open(name).readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            out[i] = buf.float
        }
        return out
    }

    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(filesDir, assetName)
        if (outFile.exists()) return outFile
        outFile.parentFile?.mkdirs()
        assets.open(assetName).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        // also copy external data if present
        val dataAsset = assetName + ".data"
        try {
            assets.open(dataAsset).use { input ->
                val dataFile = File(filesDir, dataAsset)
                dataFile.parentFile?.mkdirs()
                dataFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } catch (_: Exception) {
            // no external data
        }

        return outFile
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

    private fun summarizeProfile(profilePath: String): String {
        return try {
            val text = File(profilePath).readText()
            val arr = JSONArray(text)
            val providerCounts = mutableMapOf<String, Int>()
            val cpuOps = mutableSetOf<String>()
            val nnapiOps = mutableSetOf<String>()

            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                if (obj.optString("cat") != "Node") continue
                val args = obj.optJSONObject("args") ?: continue
                val opName = args.optString("op_name", "")
                val provider = args.optString("provider", "")
                if (opName.isEmpty() || provider.isEmpty()) continue
                providerCounts[provider] = (providerCounts[provider] ?: 0) + 1
                if (provider.contains("CPU")) cpuOps.add(opName)
                if (provider.contains("Nnapi")) nnapiOps.add(opName)
            }

            val providerSummary = providerCounts.entries
                .sortedByDescending { it.value }
                .joinToString(", ") { "${it.key}=${it.value}" }
            val nnapiList = nnapiOps.toList().sorted().joinToString(", ")
            val cpuList = cpuOps.toList().sorted().joinToString(", ")

            "providers: $providerSummary; NNAPI ops: [$nnapiList]; CPU ops: [$cpuList]"
        } catch (e: Throwable) {
            "profile summary failed: ${e.message}"
        }
    }
}
