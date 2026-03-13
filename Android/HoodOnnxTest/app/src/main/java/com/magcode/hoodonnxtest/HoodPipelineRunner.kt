package com.magcode.hoodonnxtest

import android.content.res.AssetManager
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object HoodPipelineRunner {
    private const val EDGE_MLP_CHUNK_EDGES = 4096

    private data class ProfiledSession(
        val label: String,
        val session: OrtSession
    )

    private data class SharedSessions(
        val nodeEnc: ProfiledSession,
        val edgeMeshEnc: ProfiledSession,
        val edgeWorldEnc: ProfiledSession,
        val edgeCoarse0Enc: ProfiledSession,
        val edgeCoarse1Enc: ProfiledSession,
        val edgeCoarse2Enc: ProfiledSession,
        val decoder: ProfiledSession
    )

    private data class EdgeStash(
        val oldIndex: IntArray,
        val oldFeatures: FloatArray,
        val mask: BooleanArray
    )

    private data class DownsampleStash(
        val worldDirect: EdgeStash,
        val worldInverse: EdgeStash
    )

    private data class FrameRunResult(
        val line: String,
        val maxDiff: Float
    )

    private data class PreparedInputs(
        val clothRaw: FloatArray,
        val obstacleRaw: FloatArray,
        val meshRaw: FloatArray,
        val coarse0Raw: FloatArray,
        val coarse1Raw: FloatArray,
        val coarse2Raw: FloatArray,
        val worldDirectRaw: FloatArray,
        val worldInverseRaw: FloatArray,
        val worldDirectIndex: IntArray,
        val worldInverseIndex: IntArray,
        val obstacleActiveMask: IntArray
    )

    fun run(
        assets: AssetManager,
        filesDir: File,
        env: OrtEnvironment,
        profileDir: File
    ): String {
        return when {
            assetExists(assets, "pipeline_real_sequence/config.json") -> runSequence(assets, filesDir, env, profileDir)
            assetExists(assets, "pipeline_real/config.json") -> runSingle(assets, filesDir, env, profileDir, "pipeline_real", logIntermediate = true)
            else -> runSingle(assets, filesDir, env, profileDir, "pipeline", logIntermediate = true)
        }
    }

    private fun runSequence(
        assets: AssetManager,
        filesDir: File,
        env: OrtEnvironment,
        profileDir: File
    ): String {
        val config = readJsonObject(assets, "pipeline_real_sequence/config.json")
        val frames = config.getJSONArray("frames")
        val modelBase = selectModelBase(assets)
        val sessions = createSharedSessions(env, assets, filesDir, profileDir, modelBase)
        val lines = mutableListOf<String>()
        var maxDiffAll = 0f
        try {
            for (i in 0 until frames.length()) {
                val frame = frames.getJSONObject(i)
                val frameIdx = frame.getInt("frame_idx")
                val assetBase = frame.getString("asset_base")
                val result = runFrame(assets, filesDir, env, assetBase, modelBase, sessions, logIntermediate = i == 0)
                if (result.maxDiff > maxDiffAll) maxDiffAll = result.maxDiff
                val line = "sequence frame=$frameIdx ${result.line}"
                lines.add(line)
                Log.i("HoodOnnxTest", line)
            }
        } finally {
            closeSharedSessions(sessions)
        }

        val summary = "sequence frames=${frames.length()} max_abs_diff=$maxDiffAll"
        Log.i("HoodOnnxTest", summary)
        return buildString {
            append(summary)
            if (lines.isNotEmpty()) {
                append("\n")
                append(lines.joinToString(separator = "\n"))
            }
        }
    }

    private fun runSingle(
        assets: AssetManager,
        filesDir: File,
        env: OrtEnvironment,
        profileDir: File,
        assetBase: String,
        logIntermediate: Boolean
    ): String {
        val modelBase = selectModelBase(assets)
        val sessions = createSharedSessions(env, assets, filesDir, profileDir, modelBase)
        return try {
            runFrame(assets, filesDir, env, assetBase, modelBase, sessions, logIntermediate).line
        } finally {
            closeSharedSessions(sessions)
        }
    }

    private fun runFrame(
        assets: AssetManager,
        filesDir: File,
        env: OrtEnvironment,
        assetBase: String,
        modelBase: String,
        shared: SharedSessions,
        logIntermediate: Boolean
    ): FrameRunResult {
        val config = readJsonObject(assets, "$assetBase/config.json")
        val nCloth = config.getInt("N_cloth")
        val nObs = config.getInt("N_obstacle")
        val latent = config.getInt("latent_size")
        val collisionRadius = config.optDouble("collision_radius", 3e-2).toFloat()
        val kWorldEdges = if (config.has("k_world_edges") && !config.isNull("k_world_edges")) {
            config.getInt("k_world_edges")
        } else {
            1
        }
        val blocks = config.getJSONArray("blocks")
        val processSteps = config.optJSONArray("process_steps")

        val obstacleActiveMaskRef = readIntBinarySafe(assets, "$assetBase/obstacle_active_mask.bin")

        val edgeIndexMesh = readIntBinary(assets, "$assetBase/edge_index_mesh.bin")
        val edgeIndexCoarse0 = readIntBinary(assets, "$assetBase/edge_index_coarse0.bin")
        val edgeIndexCoarse1 = readIntBinary(assets, "$assetBase/edge_index_coarse1.bin")
        val edgeIndexCoarse2 = readIntBinary(assets, "$assetBase/edge_index_coarse2.bin")
        var edgeIndexWorldDirect = readIntBinary(assets, "$assetBase/edge_index_world_direct.bin")
        var edgeIndexWorldInverse = readIntBinary(assets, "$assetBase/edge_index_world_inverse.bin")

        val usesLocalPreparedInputs = assetExists(assets, "$assetBase/node_norm_mean.bin")
        val preparedInputs = if (usesLocalPreparedInputs) {
            buildPreparedInputsLocally(
                assets,
                assetBase,
                nCloth,
                nObs,
                edgeIndexMesh,
                edgeIndexCoarse0,
                edgeIndexCoarse1,
                edgeIndexCoarse2,
                collisionRadius,
                kWorldEdges
            )
        } else {
            PreparedInputs(
                clothRaw = readFloatBinary(assets, "$assetBase/cloth_raw.bin"),
                obstacleRaw = readFloatBinary(assets, "$assetBase/obstacle_raw.bin"),
                meshRaw = readFloatBinary(assets, "$assetBase/mesh_raw.bin"),
                coarse0Raw = readFloatBinary(assets, "$assetBase/coarse0_raw.bin"),
                coarse1Raw = readFloatBinary(assets, "$assetBase/coarse1_raw.bin"),
                coarse2Raw = readFloatBinary(assets, "$assetBase/coarse2_raw.bin"),
                worldDirectRaw = readFloatBinary(assets, "$assetBase/world_direct_raw.bin"),
                worldInverseRaw = readFloatBinary(assets, "$assetBase/world_inverse_raw.bin"),
                worldDirectIndex = edgeIndexWorldDirect,
                worldInverseIndex = edgeIndexWorldInverse,
                obstacleActiveMask = obstacleActiveMaskRef ?: IntArray(nObs) { 0 }
            )
        }
        edgeIndexWorldDirect = preparedInputs.worldDirectIndex
        edgeIndexWorldInverse = preparedInputs.worldInverseIndex
        if (usesLocalPreparedInputs) {
            Log.i("HoodOnnxTest", "using local node/static-edge feature construction for $assetBase")
            if (logIntermediate) {
                logPreparedInputDiffs(
                    assets,
                    assetBase,
                    preparedInputs,
                    nCloth,
                    nObs,
                    edgeIndexMesh,
                    edgeIndexCoarse0,
                    edgeIndexCoarse1,
                    edgeIndexCoarse2,
                    readIntBinary(assets, "$assetBase/edge_index_world_direct.bin"),
                    readIntBinary(assets, "$assetBase/edge_index_world_inverse.bin"),
                    obstacleActiveMaskRef
                )
            }
        }

        val clothRaw = preparedInputs.clothRaw
        val obstacleRaw = preparedInputs.obstacleRaw
        val meshRaw = preparedInputs.meshRaw
        val coarse0Raw = preparedInputs.coarse0Raw
        val coarse1Raw = preparedInputs.coarse1Raw
        val coarse2Raw = preparedInputs.coarse2Raw
        val worldDirectRaw = preparedInputs.worldDirectRaw
        val worldInverseRaw = preparedInputs.worldInverseRaw

        val expected = readFloatBinary(assets, "$assetBase/expected_output.bin")

        // Optional debug expected tensors
        val expNodeEncCloth = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_node_encoder_cloth.bin") else null
        val expNodeEncObs = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_node_encoder_obstacle.bin") else null
        val expEdgeMesh = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_mesh.bin") else null
        val expEdgeCoarse0 = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_coarse0.bin") else null
        val expEdgeCoarse1 = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_coarse1.bin") else null
        val expEdgeCoarse2 = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_coarse2.bin") else null
        val expEdgeWorldDirect = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_world_direct.bin") else null
        val expEdgeWorldInverse = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/expected_edge_encoder_world_inverse.bin") else null
        val expBlock0NodeIn = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_node_in_cloth.bin") else null
        val expBlock0NodeOut = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_node_out_cloth.bin") else null
        val expBlock0UpdWorldDirect = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_updated_world_direct.bin") else null
        val expBlock0UpdWorldInverse = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_updated_world_inverse.bin") else null
        val expBlock0UpdMesh = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_updated_mesh.bin") else null
        val expBlock0UpdCoarse0 = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_updated_coarse0.bin") else null
        val expBlock0AggWorld = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_agg_world_cloth.bin") else null
        val expBlock0AggMesh = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_agg_mesh.bin") else null
        val expBlock0AggCoarse0 = if (logIntermediate) readFloatBinarySafe(assets, "$assetBase/blocks/block_0_0_agg_coarse0.bin") else null

        val shapeClothRaw = longArrayOf(nCloth.toLong(), (clothRaw.size / nCloth).toLong())
        val shapeEdgeMesh = longArrayOf((edgeIndexMesh.size / 2).toLong(), (meshRaw.size / (edgeIndexMesh.size / 2)).toLong())
        val shapeEdgeCoarse0 = longArrayOf((edgeIndexCoarse0.size / 2).toLong(), (coarse0Raw.size / (edgeIndexCoarse0.size / 2)).toLong())
        val shapeEdgeCoarse1 = longArrayOf((edgeIndexCoarse1.size / 2).toLong(), (coarse1Raw.size / (edgeIndexCoarse1.size / 2)).toLong())
        val shapeEdgeCoarse2 = longArrayOf((edgeIndexCoarse2.size / 2).toLong(), (coarse2Raw.size / (edgeIndexCoarse2.size / 2)).toLong())
        val shapeWorldDirectRaw = longArrayOf((edgeIndexWorldDirect.size / 2).toLong(), (worldDirectRaw.size / (edgeIndexWorldDirect.size / 2)).toLong())
        val shapeWorldInverseRaw = longArrayOf((edgeIndexWorldInverse.size / 2).toLong(), (worldInverseRaw.size / (edgeIndexWorldInverse.size / 2)).toLong())
        val shapeWorldCat = longArrayOf(shapeWorldDirectRaw[0] + shapeWorldInverseRaw[0], shapeWorldDirectRaw[1])

        val nodeFeatureDim = shapeClothRaw[1].toInt()
        val activeMask = preparedInputs.obstacleActiveMask
        val nActiveObs = activeMask.count { it != 0 }
        val combinedNodeRaw = FloatArray((nCloth + nActiveObs) * nodeFeatureDim)
        System.arraycopy(clothRaw, 0, combinedNodeRaw, 0, clothRaw.size)
        var activeWrite = nCloth * nodeFeatureDim
        for (obsIdx in 0 until nObs) {
            if (activeMask.getOrElse(obsIdx) { 0 } == 0) continue
            val src = obsIdx * nodeFeatureDim
            System.arraycopy(obstacleRaw, src, combinedNodeRaw, activeWrite, nodeFeatureDim)
            activeWrite += nodeFeatureDim
        }
        val combinedLatent = runOnnx(shared.nodeEnc.session, combinedNodeRaw, longArrayOf((nCloth + nActiveObs).toLong(), nodeFeatureDim.toLong()))
        val clothLatent = combinedLatent.copyOfRange(0, nCloth * latent)
        val obsLatent = FloatArray(nObs * latent)
        var activeRead = nCloth * latent
        for (obsIdx in 0 until nObs) {
            if (activeMask.getOrElse(obsIdx) { 0 } == 0) continue
            val dst = obsIdx * latent
            System.arraycopy(combinedLatent, activeRead, obsLatent, dst, latent)
            activeRead += latent
        }
        logIfExpected("node_encoder_cloth", clothLatent, expNodeEncCloth)
        logIfExpected("node_encoder_obstacle", obsLatent, expNodeEncObs)

        val meshLatentFull = runOnnx(shared.edgeMeshEnc.session, meshRaw, shapeEdgeMesh)
        val coarse0LatentFull = runOnnx(shared.edgeCoarse0Enc.session, coarse0Raw, shapeEdgeCoarse0)
        val coarse1LatentFull = runOnnx(shared.edgeCoarse1Enc.session, coarse1Raw, shapeEdgeCoarse1)
        val coarse2LatentFull = runOnnx(shared.edgeCoarse2Enc.session, coarse2Raw, shapeEdgeCoarse2)

        val worldCat = FloatArray(worldDirectRaw.size + worldInverseRaw.size)
        System.arraycopy(worldDirectRaw, 0, worldCat, 0, worldDirectRaw.size)
        System.arraycopy(worldInverseRaw, 0, worldCat, worldDirectRaw.size, worldInverseRaw.size)
        val worldLatentCat = runOnnx(shared.edgeWorldEnc.session, worldCat, shapeWorldCat)

        var clothNodes = clothLatent
        var obsNodes = obsLatent

        var meshEdges = meshLatentFull
        var coarse0Edges = coarse0LatentFull
        var coarse1Edges = coarse1LatentFull
        var coarse2Edges = coarse2LatentFull
        val nWorldDirect = shapeWorldDirectRaw[0].toInt()
        val nWorldInverse = shapeWorldInverseRaw[0].toInt()
        var worldDirectEdges = worldLatentCat.copyOfRange(0, nWorldDirect * latent)
        var worldInverseEdges = worldLatentCat.copyOfRange(nWorldDirect * latent, (nWorldDirect + nWorldInverse) * latent)

        logIfExpected("edge_encoder_mesh", meshEdges, expEdgeMesh)
        logIfExpected("edge_encoder_coarse0", coarse0Edges, expEdgeCoarse0)
        logIfExpected("edge_encoder_coarse1", coarse1Edges, expEdgeCoarse1)
        logIfExpected("edge_encoder_coarse2", coarse2Edges, expEdgeCoarse2)
        logIfExpected("edge_encoder_world_direct", worldDirectEdges, expEdgeWorldDirect)
        logIfExpected("edge_encoder_world_inverse", worldInverseEdges, expEdgeWorldInverse)

        val steps = processSteps ?: buildDefaultProcessSteps(blocks)
        val downsampleStack = ArrayDeque<DownsampleStash>()

        for (i in 0 until steps.length()) {
            val step = steps.getJSONObject(i)
            when (step.getString("type")) {
                "downsample" -> {
                    val remainingMask = computeRemainingNodeMask(step.getJSONArray("target_edge_keys"), nCloth, edgeIndexMesh, edgeIndexCoarse0, edgeIndexCoarse1, edgeIndexCoarse2)
                    val worldDirectStash = filterWorldEdges(edgeIndexWorldDirect, worldDirectEdges, remainingMask, latent, useSourceMask = false, useTargetMask = true)
                    edgeIndexWorldDirect = worldDirectStash.first
                    worldDirectEdges = worldDirectStash.second
                    val worldInverseStash = filterWorldEdges(edgeIndexWorldInverse, worldInverseEdges, remainingMask, latent, useSourceMask = true, useTargetMask = false)
                    edgeIndexWorldInverse = worldInverseStash.first
                    worldInverseEdges = worldInverseStash.second
                    downsampleStack.addLast(DownsampleStash(worldDirectStash.third, worldInverseStash.third))
                    continue
                }
                "upsample" -> {
                    val stashed = downsampleStack.removeLast()
                    val restoredWorldDirect = restoreWorldEdges(stashed.worldDirect, worldDirectEdges, latent)
                    edgeIndexWorldDirect = restoredWorldDirect.first
                    worldDirectEdges = restoredWorldDirect.second
                    val restoredWorldInverse = restoreWorldEdges(stashed.worldInverse, worldInverseEdges, latent)
                    edgeIndexWorldInverse = restoredWorldInverse.first
                    worldInverseEdges = restoredWorldInverse.second
                    continue
                }
            }

            val level = step.getInt("level")
            val block = step.getInt("block")
            val edgeKeys = step.getJSONArray("edge_keys")

            val nodePath = "$modelBase/blocks/block_${level}_${block}_node.onnx"
            val nodeSess = createRelaxedSession(env, assets, filesDir, nodePath)

            // Edge updates
            val worldEdgePath = "$modelBase/blocks/block_${level}_${block}_edge_world_edge.onnx"
            val worldEdgeSess = createRelaxedSession(env, assets, filesDir, worldEdgePath)

            val meshEdgeSess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_mesh_edge.onnx")) {
                createRelaxedSession(env, assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_mesh_edge.onnx")
            } else null
            val coarse0Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge0.onnx")) {
                createRelaxedSession(env, assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge0.onnx")
            } else null
            val coarse1Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge1.onnx")) {
                createRelaxedSession(env, assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge1.onnx")
            } else null
            val coarse2Sess = if (assetExists(assets, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge2.onnx")) {
                createRelaxedSession(env, assets, filesDir, "$modelBase/blocks/block_${level}_${block}_edge_coarse_edge2.onnx")
            } else null

            val worldDirectUpd = runEdgeMlp(worldEdgeSess, clothNodes, obsNodes, worldDirectEdges, edgeIndexWorldDirect, latent)
            val worldInverseUpd = runEdgeMlp(worldEdgeSess, obsNodes, clothNodes, worldInverseEdges, edgeIndexWorldInverse, latent)

            val meshUpd = if (meshEdgeSess != null) {
                runEdgeMlp(meshEdgeSess, clothNodes, clothNodes, meshEdges, edgeIndexMesh, latent)
            } else null
            val coarse0Upd = if (coarse0Sess != null) {
                runEdgeMlp(coarse0Sess, clothNodes, clothNodes, coarse0Edges, edgeIndexCoarse0, latent)
            } else null
            val coarse1Upd = if (coarse1Sess != null) {
                runEdgeMlp(coarse1Sess, clothNodes, clothNodes, coarse1Edges, edgeIndexCoarse1, latent)
            } else null
            val coarse2Upd = if (coarse2Sess != null) {
                runEdgeMlp(coarse2Sess, clothNodes, clothNodes, coarse2Edges, edgeIndexCoarse2, latent)
            } else null

            if (level == 0 && block == 0) {
                logIfExpected("block_0_0_updated_world_direct", worldDirectUpd, expBlock0UpdWorldDirect)
                logIfExpected("block_0_0_updated_world_inverse", worldInverseUpd, expBlock0UpdWorldInverse)
                if (meshUpd != null) logIfExpected("block_0_0_updated_mesh", meshUpd, expBlock0UpdMesh)
                if (coarse0Upd != null) logIfExpected("block_0_0_updated_coarse0", coarse0Upd, expBlock0UpdCoarse0)
            }

            val aggWorldCloth = CpuScatterSum.scatterSum(
                edgeIndexWorldDirect.copyOfRange(0, edgeIndexWorldDirect.size / 2),
                edgeIndexWorldDirect.copyOfRange(edgeIndexWorldDirect.size / 2, edgeIndexWorldDirect.size),
                worldDirectUpd,
                nCloth,
                latent
            )
            val aggWorldObs = CpuScatterSum.scatterSum(
                edgeIndexWorldInverse.copyOfRange(0, edgeIndexWorldInverse.size / 2),
                edgeIndexWorldInverse.copyOfRange(edgeIndexWorldInverse.size / 2, edgeIndexWorldInverse.size),
                worldInverseUpd,
                nObs,
                latent
            )

            val aggMesh = if (meshUpd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexMesh.copyOfRange(0, edgeIndexMesh.size / 2),
                    edgeIndexMesh.copyOfRange(edgeIndexMesh.size / 2, edgeIndexMesh.size),
                    meshUpd,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse0 = if (coarse0Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse0.copyOfRange(0, edgeIndexCoarse0.size / 2),
                    edgeIndexCoarse0.copyOfRange(edgeIndexCoarse0.size / 2, edgeIndexCoarse0.size),
                    coarse0Upd,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse1 = if (coarse1Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse1.copyOfRange(0, edgeIndexCoarse1.size / 2),
                    edgeIndexCoarse1.copyOfRange(edgeIndexCoarse1.size / 2, edgeIndexCoarse1.size),
                    coarse1Upd,
                    nCloth,
                    latent
                )
            } else null
            val aggCoarse2 = if (coarse2Upd != null) {
                CpuScatterSum.scatterSum(
                    edgeIndexCoarse2.copyOfRange(0, edgeIndexCoarse2.size / 2),
                    edgeIndexCoarse2.copyOfRange(edgeIndexCoarse2.size / 2, edgeIndexCoarse2.size),
                    coarse2Upd,
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

            addInPlace(worldDirectEdges, worldDirectUpd)
            addInPlace(worldInverseEdges, worldInverseUpd)
            if (meshUpd != null) addInPlace(meshEdges, meshUpd)
            if (coarse0Upd != null) addInPlace(coarse0Edges, coarse0Upd)
            if (coarse1Upd != null) addInPlace(coarse1Edges, coarse1Upd)
            if (coarse2Upd != null) addInPlace(coarse2Edges, coarse2Upd)

            addInPlace(clothNodes, nodeOutCloth)
            addInPlace(obsNodes, nodeOutObs)

            val blkKey = "$assetBase/blocks/block_${level}_${block}_cloth_nodes.bin"
            val expBlk = if (logIntermediate) readFloatBinarySafe(assets, blkKey) else null
            logIfExpected("block_${level}_${block}_cloth_nodes", clothNodes, expBlk)

            meshEdgeSess?.close()
            coarse0Sess?.close()
            coarse1Sess?.close()
            coarse2Sess?.close()
            worldEdgeSess.close()
            nodeSess.close()
        }

        val output = runOnnx(shared.decoder.session, clothNodes, longArrayOf(nCloth.toLong(), latent.toLong()))
        val maxDiff = maxAbsDiff(output, expected)
        val line = "pipeline max_abs_diff=$maxDiff"
        Log.i("HoodOnnxTest", line)
        return FrameRunResult(line, maxDiff)
    }

    private fun selectModelBase(assets: AssetManager): String {
        return if (assetExists(assets, "models_dynamic/node_encoder.onnx")) "models_dynamic" else "models_embedded"
    }

    private fun createSharedSessions(
        env: OrtEnvironment,
        assets: AssetManager,
        filesDir: File,
        profileDir: File,
        modelBase: String
    ): SharedSessions {
        return SharedSessions(
            nodeEnc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/node_encoder.onnx", "real/node_encoder"),
            edgeMeshEnc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/edge_encoder_mesh.onnx", "real/edge_encoder_mesh"),
            edgeWorldEnc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/edge_encoder_world.onnx", "real/edge_encoder_world"),
            edgeCoarse0Enc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/edge_encoder_coarse0.onnx", "real/edge_encoder_coarse0"),
            edgeCoarse1Enc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/edge_encoder_coarse1.onnx", "real/edge_encoder_coarse1"),
            edgeCoarse2Enc = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/edge_encoder_coarse2.onnx", "real/edge_encoder_coarse2"),
            decoder = createProfiledSession(env, assets, filesDir, profileDir, "$modelBase/node_decoder.onnx", "real/node_decoder")
        )
    }

    private fun closeSharedSessions(shared: SharedSessions) {
        closeProfiledSession(shared.nodeEnc)
        closeProfiledSession(shared.edgeMeshEnc)
        closeProfiledSession(shared.edgeWorldEnc)
        closeProfiledSession(shared.edgeCoarse0Enc)
        closeProfiledSession(shared.edgeCoarse1Enc)
        closeProfiledSession(shared.edgeCoarse2Enc)
        closeProfiledSession(shared.decoder)
    }

    private fun createRelaxedSession(
        env: OrtEnvironment,
        assets: AssetManager,
        filesDir: File,
        assetName: String
    ): OrtSession {
        val opts = OrtSession.SessionOptions()
        opts.addNnapi()
        return env.createSession(copyAssetToFile(assets, filesDir, assetName).absolutePath, opts)
    }

    private fun buildPreparedInputsLocally(
        assets: AssetManager,
        assetBase: String,
        nCloth: Int,
        nObs: Int,
        edgeIndexMesh: IntArray,
        edgeIndexCoarse0: IntArray,
        edgeIndexCoarse1: IntArray,
        edgeIndexCoarse2: IntArray,
        collisionRadius: Float,
        kWorldEdges: Int?
    ): PreparedInputs {
        val clothPos = readFloatBinary(assets, "$assetBase/cloth_pos.bin")
        val clothPrevPos = readFloatBinary(assets, "$assetBase/cloth_prev_pos.bin")
        val clothTargetPos = readFloatBinary(assets, "$assetBase/cloth_target_pos.bin")
        val clothRestPos = readFloatBinary(assets, "$assetBase/cloth_rest_pos.bin")
        val clothVertexType = readIntBinary(assets, "$assetBase/cloth_vertex_type.bin")
        val clothVertexLevel = readIntBinary(assets, "$assetBase/cloth_vertex_level.bin")
        val clothFaces = readIntBinary(assets, "$assetBase/cloth_faces.bin")
        val clothLogVMass = readFloatBinary(assets, "$assetBase/cloth_log_v_mass.bin")
        val clothBending = readFloatBinary(assets, "$assetBase/cloth_bending_coeff_input.bin").first()
        val clothLameMu = readFloatBinary(assets, "$assetBase/cloth_lame_mu_input.bin").first()
        val clothLameLambda = readFloatBinary(assets, "$assetBase/cloth_lame_lambda_input.bin").first()
        val timestep = readFloatBinary(assets, "$assetBase/timestep.bin").first()

        val obstaclePos = readFloatBinary(assets, "$assetBase/obstacle_pos.bin")
        val obstaclePrevPos = readFloatBinary(assets, "$assetBase/obstacle_prev_pos.bin")
        val obstacleTargetPos = readFloatBinary(assets, "$assetBase/obstacle_target_pos.bin")
        val obstacleVertexType = readIntBinary(assets, "$assetBase/obstacle_vertex_type.bin")
        val obstacleVertexLevel = readIntBinary(assets, "$assetBase/obstacle_vertex_level.bin")
        val obstacleFaces = readIntBinary(assets, "$assetBase/obstacle_faces.bin")

        val nodeTypeEmbedding = readFloatBinary(assets, "$assetBase/node_type_embedding.bin")
        val nodeTypeEmbeddingShape = readShape(assets, "$assetBase/node_type_embedding_shape.json")
        val vertexLevelEmbedding = readFloatBinary(assets, "$assetBase/vertex_level_embedding.bin")
        val vertexLevelEmbeddingShape = readShape(assets, "$assetBase/vertex_level_embedding_shape.json")
        val nodeNormMean = readFloatBinary(assets, "$assetBase/node_norm_mean.bin")
        val nodeNormStd = readFloatBinary(assets, "$assetBase/node_norm_std.bin")
        val meshNormMean = readFloatBinary(assets, "$assetBase/mesh_norm_mean.bin")
        val meshNormStd = readFloatBinary(assets, "$assetBase/mesh_norm_std.bin")
        val worldNormMean = readFloatBinary(assets, "$assetBase/world_norm_mean.bin")
        val worldNormStd = readFloatBinary(assets, "$assetBase/world_norm_std.bin")

        val clothVelocity = subtractVec3(clothPos, clothPrevPos)
        applyPinnedVertexUpdate(clothPos, clothPrevPos, clothTargetPos, clothVertexType)
        val worldConnectivity = computeWorldEdgeConnectivity(clothPos, obstaclePos, obstacleVertexType, collisionRadius, kWorldEdges)
        val edgeIndexWorldInverse = worldConnectivity.first
        val edgeIndexWorldDirect = worldConnectivity.second
        val activeMask = worldConnectivity.third

        val clothNormals = computeVertexNormals(clothPos, clothFaces, nCloth)
        val obstacleNormals = computeVertexNormals(obstaclePos, obstacleFaces, nObs)
        val obstacleVelocity = subtractVec3(obstaclePos, obstaclePrevPos)

        val clothRaw = buildNodeFeatures(
            numNodes = nCloth,
            velocity = clothVelocity,
            vertexType = clothVertexType,
            vertexLevel = clothVertexLevel,
            normals = clothNormals,
            timestep = timestep,
            logVMass = clothLogVMass,
            bending = clothBending,
            lameMu = clothLameMu,
            lameLambda = clothLameLambda,
            nodeTypeEmbedding = nodeTypeEmbedding,
            nodeTypeEmbeddingShape = nodeTypeEmbeddingShape,
            vertexLevelEmbedding = vertexLevelEmbedding,
            vertexLevelEmbeddingShape = vertexLevelEmbeddingShape,
            nodeNormMean = nodeNormMean,
            nodeNormStd = nodeNormStd,
            activeMask = null,
            isObstacle = false
        )
        val obstacleRaw = buildNodeFeatures(
            numNodes = nObs,
            velocity = obstacleVelocity,
            vertexType = obstacleVertexType,
            vertexLevel = obstacleVertexLevel,
            normals = obstacleNormals,
            timestep = timestep,
            logVMass = null,
            bending = -1f,
            lameMu = -1f,
            lameLambda = -1f,
            nodeTypeEmbedding = nodeTypeEmbedding,
            nodeTypeEmbeddingShape = nodeTypeEmbeddingShape,
            vertexLevelEmbedding = vertexLevelEmbedding,
            vertexLevelEmbeddingShape = vertexLevelEmbeddingShape,
            nodeNormMean = nodeNormMean,
            nodeNormStd = nodeNormStd,
            activeMask = activeMask,
            isObstacle = true
        )

        val meshRaw = buildStaticEdgeFeatures(clothPos, clothRestPos, edgeIndexMesh, timestep, clothBending, clothLameMu, clothLameLambda, meshNormMean, meshNormStd)
        val coarse0Raw = buildStaticEdgeFeatures(clothPos, clothRestPos, edgeIndexCoarse0, timestep, clothBending, clothLameMu, clothLameLambda, meshNormMean, meshNormStd)
        val coarse1Raw = buildStaticEdgeFeatures(clothPos, clothRestPos, edgeIndexCoarse1, timestep, clothBending, clothLameMu, clothLameLambda, meshNormMean, meshNormStd)
        val coarse2Raw = buildStaticEdgeFeatures(clothPos, clothRestPos, edgeIndexCoarse2, timestep, clothBending, clothLameMu, clothLameLambda, meshNormMean, meshNormStd)
        val worldDirectRaw = buildWorldEdgeFeaturesObstacleToCloth(obstaclePos, obstacleTargetPos, clothPos, edgeIndexWorldDirect, timestep, worldNormMean, worldNormStd)
        val worldInverseRaw = buildWorldEdgeFeaturesClothToObstacle(clothPos, obstaclePos, obstacleTargetPos, edgeIndexWorldInverse, timestep, worldNormMean, worldNormStd)

        return PreparedInputs(
            clothRaw = clothRaw,
            obstacleRaw = obstacleRaw,
            meshRaw = meshRaw,
            coarse0Raw = coarse0Raw,
            coarse1Raw = coarse1Raw,
            coarse2Raw = coarse2Raw,
            worldDirectRaw = worldDirectRaw,
            worldInverseRaw = worldInverseRaw,
            worldDirectIndex = edgeIndexWorldDirect,
            worldInverseIndex = edgeIndexWorldInverse,
            obstacleActiveMask = activeMask
        )
    }

    private fun logPreparedInputDiffs(
        assets: AssetManager,
        assetBase: String,
        local: PreparedInputs,
        nCloth: Int,
        nObs: Int,
        edgeIndexMesh: IntArray,
        edgeIndexCoarse0: IntArray,
        edgeIndexCoarse1: IntArray,
        edgeIndexCoarse2: IntArray,
        edgeIndexWorldDirect: IntArray,
        edgeIndexWorldInverse: IntArray,
        obstacleActiveMaskRef: IntArray?
    ) {
        val expClothRaw = readFloatBinary(assets, "$assetBase/cloth_raw.bin")
        val expObstacleRaw = readFloatBinary(assets, "$assetBase/obstacle_raw.bin")
        val expMeshRaw = readFloatBinary(assets, "$assetBase/mesh_raw.bin")
        val expCoarse0Raw = readFloatBinary(assets, "$assetBase/coarse0_raw.bin")
        val expCoarse1Raw = readFloatBinary(assets, "$assetBase/coarse1_raw.bin")
        val expCoarse2Raw = readFloatBinary(assets, "$assetBase/coarse2_raw.bin")
        val expWorldDirectRaw = readFloatBinary(assets, "$assetBase/world_direct_raw.bin")
        val expWorldInverseRaw = readFloatBinary(assets, "$assetBase/world_inverse_raw.bin")

        Log.i("HoodOnnxTest", "local cloth_raw max_abs_diff=${maxAbsDiff(local.clothRaw, expClothRaw)}")
        logNodeRawSegments("local cloth_raw", local.clothRaw, expClothRaw, nCloth)
        Log.i("HoodOnnxTest", "local obstacle_raw max_abs_diff=${maxAbsDiff(local.obstacleRaw, expObstacleRaw)}")
        logNodeRawSegments("local obstacle_raw", local.obstacleRaw, expObstacleRaw, nObs)

        val meshEdges = edgeIndexMesh.size / 2
        val coarse0Edges = edgeIndexCoarse0.size / 2
        val coarse1Edges = edgeIndexCoarse1.size / 2
        val coarse2Edges = edgeIndexCoarse2.size / 2

        Log.i("HoodOnnxTest", "local mesh_raw max_abs_diff=${maxAbsDiff(local.meshRaw, expMeshRaw)}")
        logEdgeRawSegments("local mesh_raw", local.meshRaw, expMeshRaw, meshEdges)
        Log.i("HoodOnnxTest", "local coarse0_raw max_abs_diff=${maxAbsDiff(local.coarse0Raw, expCoarse0Raw)}")
        logEdgeRawSegments("local coarse0_raw", local.coarse0Raw, expCoarse0Raw, coarse0Edges)
        Log.i("HoodOnnxTest", "local coarse1_raw max_abs_diff=${maxAbsDiff(local.coarse1Raw, expCoarse1Raw)}")
        logEdgeRawSegments("local coarse1_raw", local.coarse1Raw, expCoarse1Raw, coarse1Edges)
        Log.i("HoodOnnxTest", "local coarse2_raw max_abs_diff=${maxAbsDiff(local.coarse2Raw, expCoarse2Raw)}")
        logEdgeRawSegments("local coarse2_raw", local.coarse2Raw, expCoarse2Raw, coarse2Edges)

        val worldDirectEdges = edgeIndexWorldDirect.size / 2
        val worldInverseEdges = edgeIndexWorldInverse.size / 2
        Log.i("HoodOnnxTest", "local edge_index_world_direct max_abs_diff=${maxAbsDiff(local.worldDirectIndex.map { it.toFloat() }.toFloatArray(), edgeIndexWorldDirect.map { it.toFloat() }.toFloatArray())}")
        Log.i("HoodOnnxTest", "local edge_index_world_inverse max_abs_diff=${maxAbsDiff(local.worldInverseIndex.map { it.toFloat() }.toFloatArray(), edgeIndexWorldInverse.map { it.toFloat() }.toFloatArray())}")
        if (obstacleActiveMaskRef != null) {
            Log.i("HoodOnnxTest", "local obstacle_active_mask max_abs_diff=${maxAbsDiff(local.obstacleActiveMask.map { it.toFloat() }.toFloatArray(), obstacleActiveMaskRef.map { it.toFloat() }.toFloatArray())}")
        }
        Log.i("HoodOnnxTest", "local world_direct_raw max_abs_diff=${maxAbsDiff(local.worldDirectRaw, expWorldDirectRaw)}")
        logWorldRawSegments("local world_direct_raw", local.worldDirectRaw, expWorldDirectRaw, worldDirectEdges)
        Log.i("HoodOnnxTest", "local world_inverse_raw max_abs_diff=${maxAbsDiff(local.worldInverseRaw, expWorldInverseRaw)}")
        logWorldRawSegments("local world_inverse_raw", local.worldInverseRaw, expWorldInverseRaw, worldInverseEdges)
    }

    private fun logNodeRawSegments(tag: String, actual: FloatArray, expected: FloatArray, nRows: Int) {
        Log.i("HoodOnnxTest", "$tag segment velocity max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 0, 3)}")
        Log.i("HoodOnnxTest", "$tag segment node_type_emb max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 3, 12)}")
        Log.i("HoodOnnxTest", "$tag segment level_emb max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 12, 16)}")
        Log.i("HoodOnnxTest", "$tag segment normals max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 16, 19)}")
        Log.i("HoodOnnxTest", "$tag segment timestep max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 19, 20)}")
        Log.i("HoodOnnxTest", "$tag segment mass max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 20, 21)}")
        Log.i("HoodOnnxTest", "$tag segment material max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 24, 21, 24)}")
    }

    private fun logEdgeRawSegments(tag: String, actual: FloatArray, expected: FloatArray, nRows: Int) {
        Log.i("HoodOnnxTest", "$tag segment rel_pos max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 0, 3)}")
        Log.i("HoodOnnxTest", "$tag segment rel_norm max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 3, 4)}")
        Log.i("HoodOnnxTest", "$tag segment rel_rest_pos max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 4, 7)}")
        Log.i("HoodOnnxTest", "$tag segment rel_rest_norm max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 7, 8)}")
        Log.i("HoodOnnxTest", "$tag segment timestep max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 8, 9)}")
        Log.i("HoodOnnxTest", "$tag segment material max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 12, 9, 12)}")
    }

    private fun logWorldRawSegments(tag: String, actual: FloatArray, expected: FloatArray, nRows: Int) {
        Log.i("HoodOnnxTest", "$tag segment rel_pos max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 9, 0, 3)}")
        Log.i("HoodOnnxTest", "$tag segment rel_norm max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 9, 3, 4)}")
        Log.i("HoodOnnxTest", "$tag segment rel_next_pos max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 9, 4, 7)}")
        Log.i("HoodOnnxTest", "$tag segment rel_next_norm max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 9, 7, 8)}")
        Log.i("HoodOnnxTest", "$tag segment timestep max_abs_diff=${maxAbsDiffColumns(actual, expected, nRows, 9, 8, 9)}")
    }

    private fun maxAbsDiffColumns(
        a: FloatArray,
        b: FloatArray,
        nRows: Int,
        rowWidth: Int,
        colStart: Int,
        colEnd: Int
    ): Float {
        var max = 0f
        for (row in 0 until nRows) {
            val base = row * rowWidth
            for (c in colStart until colEnd) {
                val d = kotlin.math.abs(a[base + c] - b[base + c])
                if (d > max) max = d
            }
        }
        return max
    }

    private fun buildNodeFeatures(
        numNodes: Int,
        velocity: FloatArray,
        vertexType: IntArray,
        vertexLevel: IntArray,
        normals: FloatArray,
        timestep: Float,
        logVMass: FloatArray?,
        bending: Float,
        lameMu: Float,
        lameLambda: Float,
        nodeTypeEmbedding: FloatArray,
        nodeTypeEmbeddingShape: LongArray,
        vertexLevelEmbedding: FloatArray,
        vertexLevelEmbeddingShape: LongArray,
        nodeNormMean: FloatArray,
        nodeNormStd: FloatArray,
        activeMask: IntArray?,
        isObstacle: Boolean
    ): FloatArray {
        val typeCols = nodeTypeEmbeddingShape[1].toInt()
        val levelRows = vertexLevelEmbeddingShape[0].toInt()
        val levelCols = vertexLevelEmbeddingShape[1].toInt()
        val raw = FloatArray(numNodes * 24)
        for (node in 0 until numNodes) {
            val dst = node * 24
            val v3 = node * 3
            raw[dst] = velocity[v3]
            raw[dst + 1] = velocity[v3 + 1]
            raw[dst + 2] = velocity[v3 + 2]

            val typeId = vertexType[node]
            for (c in 0 until typeCols) {
                raw[dst + 3 + c] = nodeTypeEmbedding[typeId * typeCols + c]
            }

            val levelId = vertexLevel[node].coerceIn(0, levelRows - 1)
            for (c in 0 until levelCols) {
                raw[dst + 12 + c] = vertexLevelEmbedding[levelId * levelCols + c]
            }

            raw[dst + 16] = normals[v3]
            raw[dst + 17] = normals[v3 + 1]
            raw[dst + 18] = normals[v3 + 2]
            raw[dst + 19] = timestep
            raw[dst + 20] = if (isObstacle) -1f else logVMass!![node]
            raw[dst + 21] = bending
            raw[dst + 22] = lameMu
            raw[dst + 23] = lameLambda
        }

        val normalized = raw.copyOf()
        for (node in 0 until numNodes) {
            val isActive = activeMask == null || activeMask.getOrElse(node) { 0 } != 0
            if (!isActive) continue
            val base = node * 24
            for (c in 0 until 21) {
                normalized[base + c] = (raw[base + c] - nodeNormMean[c]) / nodeNormStd[c]
            }
        }
        return normalized
    }

    private fun buildStaticEdgeFeatures(
        pos: FloatArray,
        restPos: FloatArray,
        edgeIndex: IntArray,
        timestep: Float,
        bending: Float,
        lameMu: Float,
        lameLambda: Float,
        normMean: FloatArray,
        normStd: FloatArray
    ): FloatArray {
        val eCount = edgeIndex.size / 2
        val out = FloatArray(eCount * 12)
        for (e in 0 until eCount) {
            val src = edgeIndex[e]
            val tgt = edgeIndex[e + eCount]
            val src3 = src * 3
            val tgt3 = tgt * 3
            val dst = e * 12

            val rx = pos[src3] - pos[tgt3]
            val ry = pos[src3 + 1] - pos[tgt3 + 1]
            val rz = pos[src3 + 2] - pos[tgt3 + 2]
            val rr = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz)

            val rrx = restPos[src3] - restPos[tgt3]
            val rry = restPos[src3 + 1] - restPos[tgt3 + 1]
            val rrz = restPos[src3 + 2] - restPos[tgt3 + 2]
            val rrr = kotlin.math.sqrt(rrx * rrx + rry * rry + rrz * rrz)

            val pre = floatArrayOf(rx, ry, rz, rr, rrx, rry, rrz, rrr, timestep)
            for (c in 0 until 9) {
                out[dst + c] = (pre[c] - normMean[c]) / normStd[c]
            }
            out[dst + 9] = bending
            out[dst + 10] = lameMu
            out[dst + 11] = lameLambda
        }
        return out
    }

    private fun buildWorldEdgeFeaturesClothToObstacle(
        clothPos: FloatArray,
        obstaclePos: FloatArray,
        obstacleTargetPos: FloatArray,
        edgeIndex: IntArray,
        timestep: Float,
        normMean: FloatArray,
        normStd: FloatArray
    ): FloatArray {
        val eCount = edgeIndex.size / 2
        val out = FloatArray(eCount * 9)
        for (e in 0 until eCount) {
            val src = edgeIndex[e]
            val tgt = edgeIndex[e + eCount]
            val clothBase = src * 3
            val obsBase = tgt * 3
            val dst = e * 9

            val rx = clothPos[clothBase] - obstaclePos[obsBase]
            val ry = clothPos[clothBase + 1] - obstaclePos[obsBase + 1]
            val rz = clothPos[clothBase + 2] - obstaclePos[obsBase + 2]
            val rr = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz)

            val nx = clothPos[clothBase] - obstacleTargetPos[obsBase]
            val ny = clothPos[clothBase + 1] - obstacleTargetPos[obsBase + 1]
            val nz = clothPos[clothBase + 2] - obstacleTargetPos[obsBase + 2]
            val nr = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz)

            val pre = floatArrayOf(rx, ry, rz, rr, nx, ny, nz, nr, timestep)
            for (c in 0 until 9) {
                out[dst + c] = (pre[c] - normMean[c]) / normStd[c]
            }
        }
        return out
    }

    private fun buildWorldEdgeFeaturesObstacleToCloth(
        obstaclePos: FloatArray,
        obstacleTargetPos: FloatArray,
        clothPos: FloatArray,
        edgeIndex: IntArray,
        timestep: Float,
        normMean: FloatArray,
        normStd: FloatArray
    ): FloatArray {
        val eCount = edgeIndex.size / 2
        val out = FloatArray(eCount * 9)
        for (e in 0 until eCount) {
            val src = edgeIndex[e]
            val tgt = edgeIndex[e + eCount]
            val obsBase = src * 3
            val clothBase = tgt * 3
            val dst = e * 9

            val rx = obstaclePos[obsBase] - clothPos[clothBase]
            val ry = obstaclePos[obsBase + 1] - clothPos[clothBase + 1]
            val rz = obstaclePos[obsBase + 2] - clothPos[clothBase + 2]
            val rr = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz)

            val nx = obstacleTargetPos[obsBase] - clothPos[clothBase]
            val ny = obstacleTargetPos[obsBase + 1] - clothPos[clothBase + 1]
            val nz = obstacleTargetPos[obsBase + 2] - clothPos[clothBase + 2]
            val nr = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz)

            val pre = floatArrayOf(rx, ry, rz, rr, nx, ny, nz, nr, timestep)
            for (c in 0 until 9) {
                out[dst + c] = (pre[c] - normMean[c]) / normStd[c]
            }
        }
        return out
    }

    private fun computeWorldEdgeConnectivity(
        clothPos: FloatArray,
        obstaclePos: FloatArray,
        obstacleVertexType: IntArray,
        collisionRadius: Float,
        kWorldEdges: Int?
    ): Triple<IntArray, IntArray, IntArray> {
        val nCloth = clothPos.size / 3
        val nObstacle = obstaclePos.size / 3
        val radiusSq = collisionRadius * collisionRadius
        val limit = kWorldEdges ?: Int.MAX_VALUE

        val clothToObstacleSrc = ArrayList<Int>()
        val clothToObstacleTgt = ArrayList<Int>()
        val activeMask = IntArray(nObstacle)

        for (clothIdx in 0 until nCloth) {
            val cx = clothPos[clothIdx * 3]
            val cy = clothPos[clothIdx * 3 + 1]
            val cz = clothPos[clothIdx * 3 + 2]

            if (limit <= 1) {
                var bestObs = -1
                var bestDist = Float.POSITIVE_INFINITY
                for (obsIdx in 0 until nObstacle) {
                    val ox = obstaclePos[obsIdx * 3]
                    val oy = obstaclePos[obsIdx * 3 + 1]
                    val oz = obstaclePos[obsIdx * 3 + 2]
                    val dx = cx - ox
                    val dy = cy - oy
                    val dz = cz - oz
                    val dist = dx * dx + dy * dy + dz * dz
                    if (dist < bestDist) {
                        bestDist = dist
                        bestObs = obsIdx
                    }
                }
                if (bestObs >= 0 && bestDist <= radiusSq && obstacleVertexType[bestObs] != 2) {
                    clothToObstacleSrc.add(clothIdx)
                    clothToObstacleTgt.add(bestObs)
                    activeMask[bestObs] = 1
                }
                continue
            }

            val bestIdx = IntArray(limit) { -1 }
            val bestDist = FloatArray(limit) { Float.POSITIVE_INFINITY }
            for (obsIdx in 0 until nObstacle) {
                val ox = obstaclePos[obsIdx * 3]
                val oy = obstaclePos[obsIdx * 3 + 1]
                val oz = obstaclePos[obsIdx * 3 + 2]
                val dx = cx - ox
                val dy = cy - oy
                val dz = cz - oz
                val dist = dx * dx + dy * dy + dz * dz
                for (slot in 0 until limit) {
                    if (dist < bestDist[slot]) {
                        for (move in limit - 1 downTo slot + 1) {
                            bestDist[move] = bestDist[move - 1]
                            bestIdx[move] = bestIdx[move - 1]
                        }
                        bestDist[slot] = dist
                        bestIdx[slot] = obsIdx
                        break
                    }
                }
            }
            for (slot in 0 until limit) {
                val obsIdx = bestIdx[slot]
                if (obsIdx < 0) continue
                if (bestDist[slot] > radiusSq) continue
                if (obstacleVertexType[obsIdx] == 2) continue
                clothToObstacleSrc.add(clothIdx)
                clothToObstacleTgt.add(obsIdx)
                activeMask[obsIdx] = 1
            }
        }

        val eCount = clothToObstacleSrc.size
        val inverse = IntArray(eCount * 2)
        val direct = IntArray(eCount * 2)
        for (e in 0 until eCount) {
            val clothIdx = clothToObstacleSrc[e]
            val obsIdx = clothToObstacleTgt[e]
            inverse[e] = clothIdx
            inverse[e + eCount] = obsIdx
            direct[e] = obsIdx
            direct[e + eCount] = clothIdx
        }
        return Triple(inverse, direct, activeMask)
    }

    private fun subtractVec3(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(a.size)
        for (i in a.indices) out[i] = a[i] - b[i]
        return out
    }

    private fun applyPinnedVertexUpdate(
        pos: FloatArray,
        prevPos: FloatArray,
        targetPos: FloatArray,
        vertexType: IntArray
    ) {
        for (node in vertexType.indices) {
            if (vertexType[node] != 3) continue
            val base = node * 3
            prevPos[base] = pos[base]
            prevPos[base + 1] = pos[base + 1]
            prevPos[base + 2] = pos[base + 2]
            pos[base] = targetPos[base]
            pos[base + 1] = targetPos[base + 1]
            pos[base + 2] = targetPos[base + 2]
        }
    }

    private fun computeVertexNormals(pos: FloatArray, faces: IntArray, numVerts: Int): FloatArray {
        val out = FloatArray(numVerts * 3)
        val fCount = faces.size / 3
        for (f in 0 until fCount) {
            val i0 = faces[f * 3]
            val i1 = faces[f * 3 + 1]
            val i2 = faces[f * 3 + 2]

            val v0x = pos[i0 * 3]
            val v0y = pos[i0 * 3 + 1]
            val v0z = pos[i0 * 3 + 2]
            val v1x = pos[i1 * 3]
            val v1y = pos[i1 * 3 + 1]
            val v1z = pos[i1 * 3 + 2]
            val v2x = pos[i2 * 3]
            val v2y = pos[i2 * 3 + 1]
            val v2z = pos[i2 * 3 + 2]

            val e0x = v1x - v0x
            val e0y = v1y - v0y
            val e0z = v1z - v0z
            val e1x = v2x - v1x
            val e1y = v2y - v1y
            val e1z = v2z - v1z
            val e2x = v0x - v2x
            val e2y = v0y - v2y
            val e2z = v0z - v2z

            val nx = crossX(e0x, e0y, e0z, e1x, e1y, e1z) +
                crossX(e1x, e1y, e1z, e2x, e2y, e2z) +
                crossX(e2x, e2y, e2z, e0x, e0y, e0z)
            val ny = crossY(e0x, e0y, e0z, e1x, e1y, e1z) +
                crossY(e1x, e1y, e1z, e2x, e2y, e2z) +
                crossY(e2x, e2y, e2z, e0x, e0y, e0z)
            val nz = crossZ(e0x, e0y, e0z, e1x, e1y, e1z) +
                crossZ(e1x, e1y, e1z, e2x, e2y, e2z) +
                crossZ(e2x, e2y, e2z, e0x, e0y, e0z)

            accumulate3(out, i0, nx, ny, nz)
            accumulate3(out, i1, nx, ny, nz)
            accumulate3(out, i2, nx, ny, nz)
        }
        for (v in 0 until numVerts) {
            val base = v * 3
            val nx = out[base]
            val ny = out[base + 1]
            val nz = out[base + 2]
            val norm = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz)
            if (norm > 1e-12f) {
                out[base] = nx / norm
                out[base + 1] = ny / norm
                out[base + 2] = nz / norm
            }
        }
        return out
    }

    private fun accumulate3(dst: FloatArray, idx: Int, x: Float, y: Float, z: Float) {
        val base = idx * 3
        dst[base] += x
        dst[base + 1] += y
        dst[base + 2] += z
    }

    private fun crossX(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float = ay * bz - az * by
    private fun crossY(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float = az * bx - ax * bz
    private fun crossZ(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float = ax * by - ay * bx

    private fun createProfiledSession(
        env: OrtEnvironment,
        assets: AssetManager,
        filesDir: File,
        profileDir: File,
        assetName: String,
        label: String
    ): ProfiledSession {
        val opts = OrtSession.SessionOptions()
        opts.addNnapi()
        opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
        val profileFile = File(profileDir, "ort_profile_${label.replace('/', '_')}.json")
        opts.enableProfiling(profileFile.absolutePath)
        val session = env.createSession(copyAssetToFile(assets, filesDir, assetName).absolutePath, opts)
        return ProfiledSession(label, session)
    }

    private fun closeProfiledSession(profiled: ProfiledSession) {
        try {
            val profilePath = profiled.session.endProfiling()
            Log.i("HoodOnnxTest", "${profiled.label} ${summarizeProfile(profilePath)}")
        } catch (t: Throwable) {
            Log.w("HoodOnnxTest", "${profiled.label} profile summary failed", t)
        } finally {
            profiled.session.close()
        }
    }

    private fun buildDefaultProcessSteps(blocks: JSONArray): JSONArray {
        val arr = JSONArray()
        for (i in 0 until blocks.length()) {
            val blk = blocks.getJSONObject(i)
            val step = JSONObject()
            step.put("type", "block")
            step.put("level", blk.getInt("level"))
            step.put("block", blk.getInt("block"))
            step.put("edge_keys", blk.getJSONArray("edge_keys"))
            arr.put(step)
        }
        return arr
    }

    private fun computeRemainingNodeMask(
        targetEdgeKeys: JSONArray,
        nCloth: Int,
        edgeIndexMesh: IntArray,
        edgeIndexCoarse0: IntArray,
        edgeIndexCoarse1: IntArray,
        edgeIndexCoarse2: IntArray
    ): BooleanArray {
        val mask = BooleanArray(nCloth)
        for (i in 0 until targetEdgeKeys.length()) {
            val edgeIndex = when (targetEdgeKeys.getString(i)) {
                "mesh_edge" -> edgeIndexMesh
                "coarse_edge0" -> edgeIndexCoarse0
                "coarse_edge1" -> edgeIndexCoarse1
                "coarse_edge2" -> edgeIndexCoarse2
                else -> null
            } ?: continue
            val eCount = edgeIndex.size / 2
            for (e in 0 until eCount) {
                mask[edgeIndex[e]] = true
                mask[edgeIndex[e + eCount]] = true
            }
        }
        return mask
    }

    private fun filterWorldEdges(
        edgeIndex: IntArray,
        edgeFeatures: FloatArray,
        remainingMask: BooleanArray,
        latent: Int,
        useSourceMask: Boolean,
        useTargetMask: Boolean
    ): Triple<IntArray, FloatArray, EdgeStash> {
        val eCount = edgeIndex.size / 2
        val mask = BooleanArray(eCount)
        var kept = 0
        for (e in 0 until eCount) {
            val src = edgeIndex[e]
            val tgt = edgeIndex[e + eCount]
            var keep = true
            if (useSourceMask) keep = keep && remainingMask[src]
            if (useTargetMask) keep = keep && remainingMask[tgt]
            mask[e] = keep
            if (keep) kept++
        }

        val newIndex = IntArray(kept * 2)
        val newFeatures = FloatArray(kept * latent)
        var j = 0
        for (e in 0 until eCount) {
            if (!mask[e]) continue
            newIndex[j] = edgeIndex[e]
            newIndex[j + kept] = edgeIndex[e + eCount]
            System.arraycopy(edgeFeatures, e * latent, newFeatures, j * latent, latent)
            j++
        }

        val stash = EdgeStash(edgeIndex.copyOf(), edgeFeatures.copyOf(), mask)
        return Triple(newIndex, newFeatures, stash)
    }

    private fun restoreWorldEdges(
        stash: EdgeStash,
        currentFeatures: FloatArray,
        latent: Int
    ): Pair<IntArray, FloatArray> {
        val restored = stash.oldFeatures.copyOf()
        val eCount = stash.oldIndex.size / 2
        var currentEdge = 0
        for (e in 0 until eCount) {
            if (!stash.mask[e]) continue
            System.arraycopy(currentFeatures, currentEdge * latent, restored, e * latent, latent)
            currentEdge++
        }
        return Pair(stash.oldIndex.copyOf(), restored)
    }

    private fun runEdgeMlp(
        session: OrtSession,
        tgtNodes: FloatArray,
        srcNodes: FloatArray,
        edgeFeat: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray {
        val eCount = edgeIndex.size / 2
        val output = FloatArray(eCount * latent)
        var edgeStart = 0
        while (edgeStart < eCount) {
            val chunkEdges = minOf(EDGE_MLP_CHUNK_EDGES, eCount - edgeStart)
            val input = FloatArray(chunkEdges * latent * 3)
            var localEdge = 0
            while (localEdge < chunkEdges) {
                val edge = edgeStart + localEdge
                val src = edgeIndex[edge]
                val tgt = edgeIndex[edge + eCount]
                val inBase = localEdge * latent * 3
                val tgtBase = tgt * latent
                val srcBase = src * latent
                val edgeBase = edge * latent
                System.arraycopy(tgtNodes, tgtBase, input, inBase, latent)
                System.arraycopy(srcNodes, srcBase, input, inBase + latent, latent)
                System.arraycopy(edgeFeat, edgeBase, input, inBase + latent * 2, latent)
                localEdge++
            }
            val shape = longArrayOf(chunkEdges.toLong(), (latent * 3).toLong())
            val chunkOut = runOnnx(session, input, shape)
            System.arraycopy(chunkOut, 0, output, edgeStart * latent, chunkOut.size)
            edgeStart += chunkEdges
        }
        return output
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

    private fun readIntBinarySafe(assets: AssetManager, name: String): IntArray? {
        return try {
            readIntBinary(assets, name)
        } catch (_: Exception) {
            null
        }
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
}
