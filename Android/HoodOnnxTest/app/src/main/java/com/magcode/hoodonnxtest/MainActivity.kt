package com.magcode.hoodonnxtest

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.TextView
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : Activity() {
    private companion object {
        private const val RUN_STARTUP_PROBES = false
        private const val RUN_TARGETED_QNN_PROBES = false
    }

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

        val nnapiDevicesLine = "nnapi/devices: ${NativeQairtBridge.describeNnapiDevices()}"
        reportLines.add(nnapiDevicesLine)
        Log.i("HoodOnnxTest", nnapiDevicesLine)

        if (RUN_STARTUP_PROBES) {
            runCachedQnnNodeEncoderCase(reportLines, "qnn_node_encoder_case")
            runQnnCaseProbe("node_encoder_case", "qnn_node_encoder_case")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runQnnCaseProbe("block_0_0_node_case", "qnn_block_0_0_node_case")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }

            runLiteRtGpuProbe(
                label = "sample",
                assetName = "litert_probe/mix_precision_sample.tflite"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_linear",
                assetName = "litert_probe/dummy_linear.tflite"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3",
                assetName = "litert_probe/dummy_mlp3.tflite",
                useGpuDelegate = true
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3_cpu",
                assetName = "litert_probe/dummy_mlp3.tflite",
                useGpuDelegate = false
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3_large",
                assetName = "litert_probe/dummy_mlp3_large.tflite",
                useGpuDelegate = true
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3_large_cpu",
                assetName = "litert_probe/dummy_mlp3_large.tflite",
                useGpuDelegate = false
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3_xlarge",
                assetName = "litert_probe/dummy_mlp3_xlarge.tflite",
                useGpuDelegate = true,
                warmupRuns = 3,
                measuredRuns = 10
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "dummy_mlp3_xlarge_cpu",
                assetName = "litert_probe/dummy_mlp3_xlarge.tflite",
                useGpuDelegate = false,
                warmupRuns = 3,
                measuredRuns = 10
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "node_encoder_mlp",
                assetName = "litert_probe/node_encoder_mlp.tflite",
                useGpuDelegate = true,
                expectedAssetName = "split_artifacts/node_encoder/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/node_encoder/node_encoder_input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "node_encoder_mlp_cpu",
                assetName = "litert_probe/node_encoder_mlp.tflite",
                useGpuDelegate = false,
                expectedAssetName = "split_artifacts/node_encoder/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/node_encoder/node_encoder_input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "node_encoder_mlp_fp16",
                assetName = "litert_probe/node_encoder_mlp_fp16.tflite",
                useGpuDelegate = true,
                expectedAssetName = "split_artifacts/node_encoder/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/node_encoder/node_encoder_input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "node_encoder_mlp_fp16_cpu",
                assetName = "litert_probe/node_encoder_mlp_fp16.tflite",
                useGpuDelegate = false,
                expectedAssetName = "split_artifacts/node_encoder/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/node_encoder/node_encoder_input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "block_0_0_node_mlp",
                assetName = "litert_probe/block_0_0_node_mlp.tflite",
                useGpuDelegate = true,
                expectedAssetName = "split_artifacts/block_0_0_node/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/block_0_0_node/input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "block_0_0_node_mlp_cpu",
                assetName = "litert_probe/block_0_0_node_mlp.tflite",
                useGpuDelegate = false,
                expectedAssetName = "split_artifacts/block_0_0_node/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/block_0_0_node/input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "block_0_0_node_mlp_fp16",
                assetName = "litert_probe/block_0_0_node_mlp_fp16.tflite",
                useGpuDelegate = true,
                expectedAssetName = "split_artifacts/block_0_0_node/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/block_0_0_node/input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runLiteRtGpuProbe(
                label = "block_0_0_node_mlp_fp16_cpu",
                assetName = "litert_probe/block_0_0_node_mlp_fp16.tflite",
                useGpuDelegate = false,
                expectedAssetName = "split_artifacts/block_0_0_node/mlp_body_expected.raw",
                inputAssetName = "split_artifacts/block_0_0_node/input.raw"
            )?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }

            runBlockBodyProbe(env, "nnapi")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runBlockBodyProbe(env, "xnnpack")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
            runBlockBodyProbe(env, "cpu")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
        }

        if (RUN_TARGETED_QNN_PROBES) {
            runQnnCaseProbe("block_0_0_edge_mesh_mlp_body_case", "qnn_block_0_0_edge_mesh_mlp_body_case")?.let {
                reportLines.add(it)
                Log.i("HoodOnnxTest", it)
            }
        }

        val pipelineProfileDir = getExternalFilesDir(null) ?: filesDir
        val pipelineLine = HoodPipelineRunner.run(
            assets,
            filesDir,
            applicationInfo.nativeLibraryDir,
            env,
            pipelineProfileDir
        )
        reportLines.add(pipelineLine)

        val reportText = reportLines.joinToString(separator = "\n")
        copyToClipboard(reportText)
        return reportText
    }

    private fun runLiteRtGpuProbe(
        label: String,
        assetName: String,
        useGpuDelegate: Boolean = true,
        expectedAssetName: String? = null,
        inputAssetName: String? = null,
        warmupRuns: Int = 0,
        measuredRuns: Int = 1
    ): String? {
        if (!assetExists(assetName)) {
            return "litert/gpu_probe/$label: missing asset $assetName"
        }

        var compat: CompatibilityList? = null
        var delegate: GpuDelegate? = null
        var interpreter: Interpreter? = null
        return try {
            compat = CompatibilityList()
            val supported = compat.isDelegateSupportedOnThisDevice
            val options = Interpreter.Options()
            var delegateState = "not_created"
            if (useGpuDelegate && supported) {
                val delegateOptions = compat.bestOptionsForThisDevice
                delegate = GpuDelegate(delegateOptions)
                options.addDelegate(delegate)
                delegateState = "created"
            } else if (!useGpuDelegate) {
                delegateState = "disabled"
            }

            val modelFile = copyAssetToFile(assetName)
            val modelBuffer = FileInputStream(modelFile).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
            }

            interpreter = Interpreter(modelBuffer, options)
            val inputs = Array(interpreter.inputTensorCount) { index ->
                if (index == 0 && inputAssetName != null && assetExists(inputAssetName)) {
                    tensorBufferFromFloatArray(interpreter.getInputTensor(index), readFloatBinary(inputAssetName))
                } else {
                    zeroTensorBuffer(interpreter.getInputTensor(index))
                }
            }
            val outputs = HashMap<Int, Any>()
            for (index in 0 until interpreter.outputTensorCount) {
                outputs[index] = zeroTensorBuffer(interpreter.getOutputTensor(index))
            }

            fun rewindIoBuffers() {
                for (input in inputs) {
                    if (input is ByteBuffer) input.rewind()
                }
                for (value in outputs.values) {
                    if (value is ByteBuffer) value.rewind()
                }
            }

            repeat(warmupRuns) {
                rewindIoBuffers()
                interpreter.runForMultipleInputsOutputs(inputs, outputs)
            }
            val runTimesMs = ArrayList<Double>(measuredRuns)
            repeat(measuredRuns) {
                rewindIoBuffers()
                val startNs = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(inputs, outputs)
                runTimesMs.add((System.nanoTime() - startNs) / 1_000_000.0)
            }
            val elapsedMs = runTimesMs.average()
            val medianMs = runTimesMs.sorted().let { sorted ->
                if (sorted.isEmpty()) 0.0 else sorted[sorted.size / 2]
            }

            val inputSummary = (0 until interpreter.inputTensorCount).joinToString("; ") { index ->
                tensorSummary("in$index", interpreter.getInputTensor(index))
            }
            val outputSummary = (0 until interpreter.outputTensorCount).joinToString("; ") { index ->
                tensorSummary("out$index", interpreter.getOutputTensor(index))
            }
            val compareSummary = if (expectedAssetName != null && assetExists(expectedAssetName) && outputs.containsKey(0)) {
                val actual = floatArrayFromByteBuffer(outputs[0] as ByteBuffer)
                val expected = readFloatBinary(expectedAssetName)
                " compare={${compareFloatArrays(expected, actual)}}"
            } else {
                ""
            }
            "litert/gpu_probe/$label: support=$supported delegate=$delegateState run=OK " +
                "time=${"%.2f".format(elapsedMs)}ms median=${"%.2f".format(medianMs)}ms runs=$measuredRuns warmup=$warmupRuns$compareSummary inputs=[$inputSummary] outputs=[$outputSummary]"
        } catch (t: Throwable) {
            "litert/gpu_probe/$label: EXCEPTION ${t.javaClass.simpleName}: ${t.message}"
        } finally {
            interpreter?.close()
            delegate?.close()
            compat?.close()
        }
    }

    private fun runQnnCaseProbe(label: String, assetDir: String): String? {
        if (!assetDirExists(assetDir)) return null
        val qnnBundleDir = File(filesDir, assetDir)
        copyAssetTree(assetDir, qnnBundleDir)
        return "qnn/$label" + "_probe: ${NativeQairtBridge.probeQnnCase(qnnBundleDir.absolutePath)}"
    }

    private fun runCachedQnnNodeEncoderCase(reportLines: MutableList<String>, assetDir: String) {
        if (!assetDirExists(assetDir)) return
        val qnnBundleDir = File(filesDir, assetDir)
        copyAssetTree(assetDir, qnnBundleDir)
        val initLine = "qnn/node_encoder_case_cached_init: ${
            NativeQairtBridge.initCachedQnnNodeEncoderCase(qnnBundleDir.absolutePath)
        }"
        reportLines.add(initLine)
        Log.i("HoodOnnxTest", initLine)

        val runLine = "qnn/node_encoder_case_cached_run: ${
            NativeQairtBridge.runCachedQnnNodeEncoderCase(3, 20)
        }"
        reportLines.add(runLine)
        Log.i("HoodOnnxTest", runLine)

        val releaseLine = "qnn/node_encoder_case_cached_release: ${
            NativeQairtBridge.releaseCachedQnnNodeEncoderCase()
        }"
        reportLines.add(releaseLine)
        Log.i("HoodOnnxTest", releaseLine)
    }

    private fun runBlockBodyProbe(env: OrtEnvironment, provider: String): String? {
        val assetBase = "split_artifacts/block_0_0_node"
        if (!assetExists("$assetBase/manifest.json")) return null

        val manifest = JSONObject(
            assets.open("$assetBase/manifest.json").bufferedReader().use { it.readText() }
        )
        val rows = manifest.getInt("rows").toLong()
        val inputDim = manifest.getInt("input_dim").toLong()

        val opts = OrtSession.SessionOptions()
        when (provider) {
            "nnapi" -> opts.addNnapi()
            "xnnpack" -> opts.addXnnpack(emptyMap())
            "cpu" -> {}
            else -> return "split_probe/block_0_0_node_mlp_$provider: unsupported provider"
        }
        opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
        val profileFile = File(filesDir, "ort_profile_probe_block_0_0_node_mlp_${provider}.json")
        opts.enableProfiling(profileFile.absolutePath)

        val modelFile = copyAssetToFile("$assetBase/mlp_body.onnx")
        val input = readFloatBinary("$assetBase/input.raw")
        val expected = readFloatBinary("$assetBase/mlp_body_expected.raw")

        val startNs = System.nanoTime()
        val session = env.createSession(modelFile.absolutePath, opts)
        val inputName = session.inputNames.first()
        val inputTensor = OnnxTensor.createTensor(env, toFloatBuffer(input), longArrayOf(rows, inputDim))
        val result = session.run(mapOf(inputName to inputTensor))
        val output = flatten2dFloatArray(result[0].value as Array<*>)
        result.close()
        inputTensor.close()
        val profilePath = session.endProfiling()
        session.close()
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0

        return "split_probe/block_0_0_node_mlp_${provider}: " +
            "time=${"%.2f".format(elapsedMs)}ms " +
            "compare={${compareFloatArrays(expected, output)}} " +
            summarizeProfile(profilePath)
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

    private fun flatten2dFloatArray(output: Array<*>): FloatArray {
        val rows = output.size
        if (rows == 0) return FloatArray(0)
        val firstRow = output[0] as FloatArray
        val cols = firstRow.size
        val flat = FloatArray(rows * cols)
        for (r in 0 until rows) {
            val row = output[r] as FloatArray
            System.arraycopy(row, 0, flat, r * cols, cols)
        }
        return flat
    }

    private fun compareFloatArrays(expected: FloatArray, actual: FloatArray, atol: Float = 1e-5f, rtol: Float = 1e-4f): String {
        if (expected.size != actual.size) {
            return "size_mismatch expected=${expected.size} actual=${actual.size}"
        }
        var maxAbs = 0f
        var mismatch = 0
        for (i in expected.indices) {
            val abs = kotlin.math.abs(expected[i] - actual[i])
            if (abs > maxAbs) maxAbs = abs
            val tol = atol + rtol * kotlin.math.abs(expected[i])
            if (abs > tol) mismatch++
        }
        return "max_abs=${"%.6g".format(maxAbs)} mismatch=$mismatch"
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
        } catch (t: Throwable) {
            "profile summary failed: ${t.message}"
        }
    }

    private fun toFloatBuffer(data: FloatArray): java.nio.FloatBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        val fb = bb.asFloatBuffer()
        fb.put(data)
        fb.rewind()
        return fb
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

    private fun zeroTensorBuffer(tensor: Tensor): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        buffer.rewind()
        return buffer
    }

    private fun tensorBufferFromFloatArray(tensor: Tensor, data: FloatArray): ByteBuffer {
        val expectedFloats = tensor.numBytes() / 4
        require(data.size == expectedFloats) {
            "Input size mismatch for ${tensor.name()}: expected $expectedFloats floats, got ${data.size}"
        }
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(data)
        buffer.rewind()
        return buffer
    }

    private fun tensorSummary(name: String, tensor: Tensor): String {
        val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
        return "$name{type=${tensor.dataType()},shape=$shape,bytes=${tensor.numBytes()}}"
    }

    private fun floatArrayFromByteBuffer(buffer: ByteBuffer): FloatArray {
        val dup = buffer.duplicate().order(ByteOrder.nativeOrder())
        dup.rewind()
        val out = FloatArray(dup.remaining() / 4)
        dup.asFloatBuffer().get(out)
        return out
    }

    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(filesDir, assetName)
        outFile.parentFile?.mkdirs()
        assets.open(assetName).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
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

    private fun copyAssetTree(assetPath: String, outPath: File) {
        val children = assets.list(assetPath) ?: emptyArray()
        if (children.isEmpty()) {
            outPath.parentFile?.mkdirs()
            assets.open(assetPath).use { input ->
                outPath.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            return
        }
        outPath.mkdirs()
        for (child in children) {
            val childAssetPath = "$assetPath/$child"
            copyAssetTree(childAssetPath, File(outPath, child))
        }
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            assets.open(assetName).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun assetDirExists(assetPath: String): Boolean {
        return try {
            assets.list(assetPath) != null
        } catch (_: Exception) {
            false
        }
    }
}
