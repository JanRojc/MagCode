package com.magcode.hoodonnxtest

import android.content.res.AssetManager
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object HoodPipelineRunner {
    fun run(
        assets: AssetManager,
        filesDir: File,
        env: OrtEnvironment,
        opts: OrtSession.SessionOptions
    ): String {
        val config = readJsonObject(assets, "pipeline/config.json")
        val nCloth = config.getInt("N_cloth")
        val nObs = config.getInt("N_obstacle")
        val eBlock = config.getInt("E_block")
        val eEncoder = config.getInt("E_encoder")
        val latent = config.getInt("latent_size")
        val outSize = config.getInt("output_size")
        val blocks = config.getJSONArray("blocks")

        val clothRaw = readFloatBinary(assets, "pipeline/cloth_raw.bin")
        val obstacleRaw = readFloatBinary(assets, "pipeline/obstacle_raw.bin")
        val meshRaw = readFloatBinary(assets, "pipeline/mesh_raw.bin")
        val coarse0Raw = readFloatBinary(assets, "pipeline/coarse0_raw.bin")
        val coarse1Raw = readFloatBinary(assets, "pipeline/coarse1_raw.bin")
        val coarse2Raw = readFloatBinary(assets, "pipeline/coarse2_raw.bin")
        val worldDirectRaw = readFloatBinary(assets, "pipeline/world_direct_raw.bin")
        val worldInverseRaw = readFloatBinary(assets, "pipeline/world_inverse_raw.bin")

        val edgeIndexMesh = readIntBinary(assets, "pipeline/edge_index_mesh.bin")
        val edgeIndexCoarse0 = readIntBinary(assets, "pipeline/edge_index_coarse0.bin")
        val edgeIndexCoarse1 = readIntBinary(assets, "pipeline/edge_index_coarse1.bin")
        val edgeIndexCoarse2 = readIntBinary(assets, "pipeline/edge_index_coarse2.bin")
        val edgeIndexWorldDirect = readIntBinary(assets, "pipeline/edge_index_world_direct.bin")
        val edgeIndexWorldInverse = readIntBinary(assets, "pipeline/edge_index_world_inverse.bin")

        val expected = readFloatBinary(assets, "pipeline/expected_output.bin")

        // Optional debug expected tensors
        val expNodeEncCloth = readFloatBinarySafe(assets, "pipeline/expected_node_encoder_cloth.bin")
        val expNodeEncObs = readFloatBinarySafe(assets, "pipeline/expected_node_encoder_obstacle.bin")
        val expEdgeMesh = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_mesh.bin")
        val expEdgeCoarse0 = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_coarse0.bin")
        val expEdgeCoarse1 = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_coarse1.bin")
        val expEdgeCoarse2 = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_coarse2.bin")
        val expEdgeWorldDirect = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_world_direct.bin")
        val expEdgeWorldInverse = readFloatBinarySafe(assets, "pipeline/expected_edge_encoder_world_inverse.bin")
        val expBlock0NodeIn = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_node_in_cloth.bin")
        val expBlock0NodeOut = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_node_out_cloth.bin")
        val expBlock0UpdWorldDirect = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_updated_world_direct.bin")
        val expBlock0UpdWorldInverse = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_updated_world_inverse.bin")
        val expBlock0UpdMesh = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_updated_mesh.bin")
        val expBlock0UpdCoarse0 = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_updated_coarse0.bin")
        val expBlock0AggWorld = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_agg_world_cloth.bin")
        val expBlock0AggMesh = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_agg_mesh.bin")
        val expBlock0AggCoarse0 = readFloatBinarySafe(assets, "pipeline/blocks/block_0_0_agg_coarse0.bin")

        val shapeClothRaw = readShape(assets, "pipeline/cloth_raw_shape.json")
        val shapeObsRaw = readShape(assets, "pipeline/obstacle_raw_shape.json")
        val shapeMeshRaw = readShape(assets, "pipeline/mesh_raw_shape.json")
        val shapeCoarse0Raw = readShape(assets, "pipeline/coarse0_raw_shape.json")
        val shapeCoarse1Raw = readShape(assets, "pipeline/coarse1_raw_shape.json")
        val shapeCoarse2Raw = readShape(assets, "pipeline/coarse2_raw_shape.json")
        val shapeWorldDirectRaw = readShape(assets, "pipeline/world_direct_raw_shape.json")
        val shapeWorldInverseRaw = readShape(assets, "pipeline/world_inverse_raw_shape.json")

        val shapeNodeEnc = shapeClothRaw
        val shapeNodeEncObs = shapeObsRaw
        val shapeEdgeMesh = shapeMeshRaw
        val shapeEdgeCoarse0 = shapeCoarse0Raw
        val shapeEdgeCoarse1 = shapeCoarse1Raw
        val shapeEdgeCoarse2 = shapeCoarse2Raw
        val shapeWorldCat = longArrayOf(eBlock * 2L, shapeWorldDirectRaw[1])

        val modelBase = "models_embedded"
        val nodeEnc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/node_encoder.onnx").absolutePath, opts)
        val edgeMeshEnc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/edge_encoder_mesh.onnx").absolutePath, opts)
        val edgeWorldEnc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/edge_encoder_world.onnx").absolutePath, opts)
        val edgeCoarse0Enc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/edge_encoder_coarse0.onnx").absolutePath, opts)
        val edgeCoarse1Enc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/edge_encoder_coarse1.onnx").absolutePath, opts)
        val edgeCoarse2Enc = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/edge_encoder_coarse2.onnx").absolutePath, opts)
        val decoder = env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/node_decoder.onnx").absolutePath, opts)

        val clothLatent = runOnnx(nodeEnc, clothRaw, shapeNodeEnc)
        val obsLatent = FloatArray(nObs * latent)
        logIfExpected("node_encoder_cloth", clothLatent, expNodeEncCloth)
        logIfExpected("node_encoder_obstacle", obsLatent, expNodeEncObs)

        val meshLatentFull = runOnnx(edgeMeshEnc, meshRaw, shapeEdgeMesh)
        val coarse0LatentFull = runOnnx(edgeCoarse0Enc, coarse0Raw, shapeEdgeCoarse0)
        val coarse1LatentFull = runOnnx(edgeCoarse1Enc, coarse1Raw, shapeEdgeCoarse1)
        val coarse2LatentFull = runOnnx(edgeCoarse2Enc, coarse2Raw, shapeEdgeCoarse2)

        val worldCat = FloatArray(worldDirectRaw.size + worldInverseRaw.size)
        System.arraycopy(worldDirectRaw, 0, worldCat, 0, worldDirectRaw.size)
        System.arraycopy(worldInverseRaw, 0, worldCat, worldDirectRaw.size, worldInverseRaw.size)
        val worldLatentCat = runOnnx(edgeWorldEnc, worldCat, shapeWorldCat)

        var clothNodes = clothLatent
        var obsNodes = obsLatent

        var meshEdges = meshLatentFull.copyOfRange(0, eBlock * latent)
        var coarse0Edges = coarse0LatentFull.copyOfRange(0, eBlock * latent)
        var coarse1Edges = coarse1LatentFull.copyOfRange(0, eBlock * latent)
        var coarse2Edges = coarse2LatentFull.copyOfRange(0, eBlock * latent)
        var worldDirectEdges = worldLatentCat.copyOfRange(0, eBlock * latent)
        var worldInverseEdges = worldLatentCat.copyOfRange(eBlock * latent, eBlock * latent * 2)

        logIfExpected("edge_encoder_mesh", meshEdges, expEdgeMesh)
        logIfExpected("edge_encoder_coarse0", coarse0Edges, expEdgeCoarse0)
        logIfExpected("edge_encoder_coarse1", coarse1Edges, expEdgeCoarse1)
        logIfExpected("edge_encoder_coarse2", coarse2Edges, expEdgeCoarse2)
        logIfExpected("edge_encoder_world_direct", worldDirectEdges, expEdgeWorldDirect)
        logIfExpected("edge_encoder_world_inverse", worldInverseEdges, expEdgeWorldInverse)

        val edgeIndexMap = mapOf(
            "mesh_edge" to edgeIndexMesh,
            "coarse_edge0" to edgeIndexCoarse0,
            "coarse_edge1" to edgeIndexCoarse1,
            "coarse_edge2" to edgeIndexCoarse2,
            "world_direct" to edgeIndexWorldDirect,
            "world_inverse" to edgeIndexWorldInverse
        )

        val zeroCloth = FloatArray(nCloth * latent)
        val zeroObs = FloatArray(nObs * latent)

        for (i in 0 until blocks.length()) {
            val blk = blocks.getJSONObject(i)
            val level = blk.getInt("level")
            val block = blk.getInt("block")
            val edgeKeys = blk.getJSONArray("edge_keys")

            val nodePath = "$modelBase/blocks/block_${level}_${block}_node.onnx"
            val nodeSess = env.createSession(copyAssetToFile(assets, filesDir, nodePath).absolutePath, opts)

            // Edge updates
            val worldEdgePath = "$modelBase/blocks/block_${level}_${block}_edge_world_edge.onnx"
            val worldEdgeSess = env.createSession(copyAssetToFile(assets, filesDir, worldEdgePath).absolutePath, opts)

            val meshEdgeSess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_mesh_edge.onnx")) {
                env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_mesh_edge.onnx").absolutePath, opts)
            } else null
            val coarse0Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge0.onnx")) {
                env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge0.onnx").absolutePath, opts)
            } else null
            val coarse1Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge1.onnx")) {
                env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge1.onnx").absolutePath, opts)
            } else null
            val coarse2Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge2.onnx")) {
                env.createSession(copyAssetToFile(assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge2.onnx").absolutePath, opts)
            } else null

            val worldDirectUpd = runEdgeMlp(worldEdgeSess, clothNodes, obsNodes, worldDirectEdges, edgeIndexWorldDirect, eBlock, latent)
            val worldInverseUpd = runEdgeMlp(worldEdgeSess, obsNodes, clothNodes, worldInverseEdges, edgeIndexWorldInverse, eBlock, latent)

            val meshUpd = if (meshEdgeSess != null) {
                runEdgeMlp(meshEdgeSess, clothNodes, clothNodes, meshEdges, edgeIndexMesh, eBlock, latent)
            } else null
            val coarse0Upd = if (coarse0Sess != null) {
                runEdgeMlp(coarse0Sess, clothNodes, clothNodes, coarse0Edges, edgeIndexCoarse0, eBlock, latent)
            } else null
            val coarse1Upd = if (coarse1Sess != null) {
                runEdgeMlp(coarse1Sess, clothNodes, clothNodes, coarse1Edges, edgeIndexCoarse1, eBlock, latent)
            } else null
            val coarse2Upd = if (coarse2Sess != null) {
                runEdgeMlp(coarse2Sess, clothNodes, clothNodes, coarse2Edges, edgeIndexCoarse2, eBlock, latent)
            } else null

            if (level == 0 && block == 0) {
                logIfExpected("block_0_0_updated_world_direct", worldDirectUpd, expBlock0UpdWorldDirect)
                logIfExpected("block_0_0_updated_world_inverse", worldInverseUpd, expBlock0UpdWorldInverse)
                if (meshUpd != null) logIfExpected("block_0_0_updated_mesh", meshUpd, expBlock0UpdMesh)
                if (coarse0Upd != null) logIfExpected("block_0_0_updated_coarse0", coarse0Upd, expBlock0UpdCoarse0)
            }

            addInPlace(worldDirectEdges, worldDirectUpd)
            addInPlace(worldInverseEdges, worldInverseUpd)
            if (meshUpd != null) addInPlace(meshEdges, meshUpd)
            if (coarse0Upd != null) addInPlace(coarse0Edges, coarse0Upd)
            if (coarse1Upd != null) addInPlace(coarse1Edges, coarse1Upd)
            if (coarse2Upd != null) addInPlace(coarse2Edges, coarse2Upd)

            val aggWorldCloth = CpuScatterSum.scatterSum(
                edgeIndexWorldDirect.copyOfRange(0, eBlock),
                edgeIndexWorldDirect.copyOfRange(eBlock, eBlock * 2),
                worldDirectEdges,
                nCloth,
                latent
            )
            val aggWorldObs = CpuScatterSum.scatterSum(
                edgeIndexWorldInverse.copyOfRange(0, eBlock),
                edgeIndexWorldInverse.copyOfRange(eBlock, eBlock * 2),
                worldInverseEdges,
                nObs,
                latent
            )

            val aggMesh = if (meshUpd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexMesh.copyOfRange(0, eBlock),
                    edgeIndexMesh.copyOfRange(eBlock, eBlock * 2),
                    meshEdges,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse0 = if (coarse0Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse0.copyOfRange(0, eBlock),
                    edgeIndexCoarse0.copyOfRange(eBlock, eBlock * 2),
                    coarse0Edges,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse1 = if (coarse1Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse1.copyOfRange(0, eBlock),
                    edgeIndexCoarse1.copyOfRange(eBlock, eBlock * 2),
                    coarse1Edges,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse2 = if (coarse2Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse2.copyOfRange(0, eBlock),
                    edgeIndexCoarse2.copyOfRange(eBlock, eBlock * 2),
                    coarse2Edges,
                    nCloth,
                    latent
                )
            } else null

            if (level == 0 && block == 0) {
                logIfExpected("block_0_0_agg_world_cloth", aggWorldCloth, expBlock0AggWorld)
                if (aggMesh != null) logIfExpected("block_0_0_agg_mesh", aggMesh, expBlock0AggMesh)
                if (aggCoarse0 != null) logIfExpected("block_0_0_agg_coarse0", aggCoarse0, expBlock0AggCoarse0)
            }

            val nodeInCloth = concatNodeInputs(clothNodes, edgeKeys, aggWorldCloth, aggMesh, aggCoarse0, aggCoarse1, aggCoarse2, nCloth, latent)
            val nodeInObs = concatNodeInputs(obsNodes, edgeKeys, aggWorldObs, null, null, null, null, nObs, latent)

            if (level == 0 && block == 0) {
                logIfExpected("block_0_0_node_in_cloth", nodeInCloth, expBlock0NodeIn)
                logNodeInSegments("block_0_0_node_in_cloth", nodeInCloth, expBlock0NodeIn, edgeKeys, nCloth, latent)
            }

            val nodeOutCloth = runOnnx(nodeSess, nodeInCloth, longArrayOf(nCloth.toLong(), (latent * (1 + edgeKeys.length())).toLong()))
            val nodeOutObs = runOnnx(nodeSess, nodeInObs, longArrayOf(nObs.toLong(), (latent * (1 + edgeKeys.length())).toLong()))

            if (level == 0 && block == 0) {
                logIfExpected("block_0_0_node_out_cloth", nodeOutCloth, expBlock0NodeOut)
            }

            addInPlace(clothNodes, nodeOutCloth)
            addInPlace(obsNodes, nodeOutObs)

            val blkKey = "pipeline/blocks/block_${level}_${block}_cloth_nodes.bin"
            val expBlk = readFloatBinarySafe(assets, blkKey)
            logIfExpected("block_${level}_${block}_cloth_nodes", clothNodes, expBlk)

            meshEdgeSess?.close()
            coarse0Sess?.close()
            coarse1Sess?.close()
            coarse2Sess?.close()
            worldEdgeSess.close()
            nodeSess.close()
        }

        val output = runOnnx(decoder, clothNodes, longArrayOf(nCloth.toLong(), latent.toLong()))
        val maxDiff = maxAbsDiff(output, expected)
        val line = "pipeline max_abs_diff=$maxDiff"
        Log.i("HoodOnnxTest", line)
        return line
    }

    private fun runEdgeMlp(
        session: OrtSession,
        tgtNodes: FloatArray,
        srcNodes: FloatArray,
        edgeFeat: FloatArray,
        edgeIndex: IntArray,
        eBlock: Int,
        latent: Int
    ): FloatArray {
        val input = FloatArray(eBlock * latent * 3)
        var e = 0
        while (e < eBlock) {
            val src = edgeIndex[e]
            val tgt = edgeIndex[e + eBlock]
            val inBase = e * latent * 3
            val tgtBase = tgt * latent
            val srcBase = src * latent
            val edgeBase = e * latent
            System.arraycopy(tgtNodes, tgtBase, input, inBase, latent)
            System.arraycopy(srcNodes, srcBase, input, inBase + latent, latent)
            System.arraycopy(edgeFeat, edgeBase, input, inBase + latent * 2, latent)
            e++
        }
        val shape = longArrayOf(eBlock.toLong(), (latent * 3).toLong())
        return runOnnx(session, input, shape)
    }

    private fun concatNodeInputs(
        nodes: FloatArray,
        edgeKeys: JSONArray,
        worldAgg: FloatArray,
        meshAgg: FloatArray?,
        coarse0Agg: FloatArray?,
        coarse1Agg: FloatArray?,
        coarse2Agg: FloatArray?,
        nNodes: Int,
        latent: Int
    ): FloatArray {
        val numKeys = edgeKeys.length()
        val rowWidth = latent * (1 + numKeys)
        val out = FloatArray(nNodes * rowWidth)
        for (node in 0 until nNodes) {
            var dst = node * rowWidth
            val srcNode = node * latent
            System.arraycopy(nodes, srcNode, out, dst, latent)
            dst += latent
            for (i in 0 until numKeys) {
                val key = edgeKeys.getString(i)
                val block = when (key) {
                    "world_edge" -> worldAgg
                    "mesh_edge" -> meshAgg
                    "coarse_edge0" -> coarse0Agg
                    "coarse_edge1" -> coarse1Agg
                    "coarse_edge2" -> coarse2Agg
                    else -> null
                }
                if (block != null) {
                    val src = node * latent
                    System.arraycopy(block, src, out, dst, latent)
                }
                // else leave zeros
                dst += latent
            }
        }
        return out
    }

    private fun addInPlace(dst: FloatArray, src: FloatArray) {
        for (i in dst.indices) {
            dst[i] += src[i]
        }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var max = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) {
            val d = kotlin.math.abs(a[i] - b[i])
            if (d > max) max = d
        }
        return max
    }

    private fun runOnnx(session: OrtSession, input: FloatArray, shape: LongArray): FloatArray {
        val inputName = session.inputNames.first()
        val inputTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), toFloatBuffer(input), shape)
        val results = session.run(mapOf(inputName to inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val fb = outputTensor.floatBuffer
        val out = FloatArray(fb.remaining())
        fb.get(out)
        outputTensor.close()
        inputTensor.close()
        results.close()
        return out
    }

    private fun toFloatBuffer(data: FloatArray): java.nio.FloatBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        val fb = bb.asFloatBuffer()
        fb.put(data)
        fb.rewind()
        return fb
    }

    private fun readJsonObject(assets: AssetManager, name: String): JSONObject {
        val text = assets.open(name).bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    private fun readShape(assets: AssetManager, name: String): LongArray {
        val text = assets.open(name).bufferedReader().use { it.readText() }
        val arr = JSONArray(text)
        val out = LongArray(arr.length())
        for (i in 0 until arr.length()) {
            out[i] = arr.getLong(i)
        }
        return out
    }

    private fun readFloatBinary(assets: AssetManager, name: String): FloatArray {
        val bytes = assets.open(name).readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            out[i] = buf.float
        }
        return out
    }

    private fun readFloatBinarySafe(assets: AssetManager, name: String): FloatArray? {
        return try {
            readFloatBinary(assets, name)
        } catch (_: Exception) {
            null
        }
    }

    private fun readIntBinary(assets: AssetManager, name: String): IntArray {
        val bytes = assets.open(name).readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = IntArray(bytes.size / 4)
        for (i in out.indices) {
            out[i] = buf.int
        }
        return out
    }

    private fun assetExists(assets: AssetManager, name: String): Boolean {
        return try {
            assets.open(name).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun logIfExpected(tag: String, actual: FloatArray, expected: FloatArray?) {
        if (expected == null) return
        val diff = maxAbsDiff(actual, expected)
        Log.i("HoodOnnxTest", "$tag max_abs_diff=$diff")
    }

    private fun logNodeInSegments(
        tag: String,
        actual: FloatArray,
        expected: FloatArray?,
        edgeKeys: JSONArray,
        nNodes: Int,
        latent: Int
    ) {
        if (expected == null) return
        val seg = nNodes * latent
        var offset = 0
        Log.i("HoodOnnxTest", "$tag segment nodes max_abs_diff=${maxAbsDiffSegment(actual, expected, offset, seg)}")
        offset += seg
        for (i in 0 until edgeKeys.length()) {
            val key = edgeKeys.getString(i)
            val d = maxAbsDiffSegment(actual, expected, offset, seg)
            Log.i("HoodOnnxTest", "$tag segment $key max_abs_diff=$d")
            offset += seg
        }
    }

    private fun maxAbsDiffSegment(a: FloatArray, b: FloatArray, start: Int, len: Int): Float {
        var max = 0f
        val end = minOf(start + len, minOf(a.size, b.size))
        var i = start
        while (i < end) {
            val d = kotlin.math.abs(a[i] - b[i])
            if (d > max) max = d
            i++
        }
        return max
    }

    private fun copyAssetToFile(assets: AssetManager, filesDir: File, assetName: String): File {
        val outFile = File(filesDir, assetName)
        if (outFile.exists()) return outFile
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
}
