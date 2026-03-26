package com.magcode.hoodonnxtest

import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class SplitPartitionArtifacts(
    val assetBase: String,
    val manifest: JSONObject,
    val mlpBodyModelAsset: String,
    val gamma: FloatArray,
    val beta: FloatArray,
    val eps: Float,
    val rows: Int,
    val inputDim: Int,
    val hiddenDim: Int,
    val outputDim: Int
)

object SplitArtifactLoader {
    fun load(assets: AssetManager, assetBase: String): SplitPartitionArtifacts? {
        val manifestPath = "$assetBase/manifest.json"
        if (!assetExists(assets, manifestPath)) return null

        val manifest = JSONObject(assets.open(manifestPath).bufferedReader().use { it.readText() })
        val files = manifest.getJSONObject("files")
        val mlpBodyModelAsset = "$assetBase/${files.getString("mlp_body_onnx")}"
        val gammaAsset = "$assetBase/${files.getString("layernorm_gamma")}"
        val betaAsset = "$assetBase/${files.getString("layernorm_beta")}"

        val gamma = readFloatBinary(assets, gammaAsset)
        val beta = readFloatBinary(assets, betaAsset)
        return SplitPartitionArtifacts(
            assetBase = assetBase,
            manifest = manifest,
            mlpBodyModelAsset = mlpBodyModelAsset,
            gamma = gamma,
            beta = beta,
            eps = manifest.getDouble("layernorm_eps").toFloat(),
            rows = manifest.getInt("rows"),
            inputDim = manifest.getInt("input_dim"),
            hiddenDim = manifest.getInt("hidden_dim"),
            outputDim = manifest.getInt("output_dim")
        )
    }

    fun copyModelToFile(assets: AssetManager, filesDir: File, artifacts: SplitPartitionArtifacts): File {
        val assetName = artifacts.mlpBodyModelAsset
        val outFile = File(filesDir, assetName)
        if (outFile.exists()) return outFile
        outFile.parentFile?.mkdirs()
        assets.open(assetName).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        val externalDataAsset = assetName + ".data"
        if (assetExists(assets, externalDataAsset)) {
            val outData = File(filesDir, externalDataAsset)
            outData.parentFile?.mkdirs()
            assets.open(externalDataAsset).use { input ->
                outData.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile
    }

    private fun assetExists(assets: AssetManager, path: String): Boolean {
        return try {
            assets.open(path).close()
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun readFloatBinary(assets: AssetManager, path: String): FloatArray {
        val bytes = assets.open(path).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            out[i] = buffer.float
        }
        return out
    }
}
