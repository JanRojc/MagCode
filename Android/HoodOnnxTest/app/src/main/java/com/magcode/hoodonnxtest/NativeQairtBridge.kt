package com.magcode.hoodonnxtest

object NativeQairtBridge {
    init {
        try {
            System.loadLibrary("hood_gpu")
        } catch (_: Throwable) {
            // layer norm calls will fail naturally if the library is unavailable
        }
    }

    external fun applyLayerNorm(
        input: FloatArray,
        rows: Int,
        cols: Int,
        gamma: FloatArray,
        beta: FloatArray,
        eps: Float
    ): FloatArray

    external fun describeNnapiDevices(): String

    external fun probeQnnCase(bundleDir: String): String

    external fun initCachedQnnNodeEncoderCase(bundleDir: String): String

    external fun runCachedQnnNodeEncoderCase(warmupRuns: Int, measuredRuns: Int): String

    external fun runCachedQnnNodeEncoderCaseInput(input: FloatArray): FloatArray

    external fun releaseCachedQnnNodeEncoderCase(): String

    external fun initCachedQnnBlock000EdgeMeshMlpBodyCase(bundleDir: String): String
    external fun initCachedQnnBlock001EdgeMeshMlpBodyCase(bundleDir: String): String
    external fun initCachedQnnBlock002EdgeMeshMlpBodyCase(bundleDir: String): String

    external fun runCachedQnnBlock000EdgeMeshMlpBodyCaseInput(input: FloatArray): FloatArray

    external fun runCachedQnnBlock000EdgeMeshMlpBodyCasePacked(
        tgtNodes: FloatArray,
        srcNodes: FloatArray,
        edgeFeat: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray

    external fun runCachedQnnBlock000EdgeMeshMlpBodyCasePackedAgg(
        tgtNodes: FloatArray,
        srcNodes: FloatArray,
        edgeFeat: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray

    external fun primeCachedQnnMeshEdgeState(edgeFeat: FloatArray, latent: Int): String

    external fun runCachedQnnBlock000EdgeMeshMlpBodyCaseStateAgg(
        clothNodes: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray

    external fun runCachedQnnBlock001EdgeMeshMlpBodyCaseStateAgg(
        clothNodes: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray

    external fun runCachedQnnBlock002EdgeMeshMlpBodyCaseStateAgg(
        clothNodes: FloatArray,
        edgeIndex: IntArray,
        latent: Int
    ): FloatArray

    external fun exportCachedQnnMeshEdgeState(): FloatArray

    external fun releaseCachedQnnBlock000EdgeMeshMlpBodyCase(): String
    external fun releaseCachedQnnBlock001EdgeMeshMlpBodyCase(): String
    external fun releaseCachedQnnBlock002EdgeMeshMlpBodyCase(): String
}
