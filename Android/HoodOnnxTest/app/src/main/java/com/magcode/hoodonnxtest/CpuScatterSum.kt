package com.magcode.hoodonnxtest

import kotlin.math.abs

object CpuScatterSum {
    /**
     * Scatter-sum edge values into target nodes.
     *
     * edgeSrc/edgeTgt: length E arrays
     * edgeValues: length E * featureDim (row-major, per-edge features)
     * returns: length numTargets * featureDim (row-major)
     */
    fun scatterSum(
        edgeSrc: IntArray,
        edgeTgt: IntArray,
        edgeValues: FloatArray,
        numTargets: Int,
        featureDim: Int
    ): FloatArray {
        require(edgeSrc.size == edgeTgt.size) { "edgeSrc/edgeTgt size mismatch" }
        val eCount = edgeSrc.size
        require(edgeValues.size == eCount * featureDim) {
            "edgeValues size ${edgeValues.size} != E*F ${eCount * featureDim}"
        }
        val out = FloatArray(numTargets * featureDim)
        var e = 0
        while (e < eCount) {
            val tgt = edgeTgt[e]
            val inBase = e * featureDim
            val outBase = tgt * featureDim
            var f = 0
            while (f < featureDim) {
                out[outBase + f] += edgeValues[inBase + f]
                f++
            }
            e++
        }
        return out
    }

    /**
     * Minimal correctness check against a hand-computed example.
     */
    fun selfTest(): Boolean {
        val edgeSrc = intArrayOf(0, 1, 2)
        val edgeTgt = intArrayOf(1, 1, 2)
        val featureDim = 2
        val edgeValues = floatArrayOf(
            1f, 2f,   // edge 0
            3f, 4f,   // edge 1
            5f, 6f    // edge 2
        )
        val out = scatterSum(edgeSrc, edgeTgt, edgeValues, numTargets = 3, featureDim = featureDim)
        val expected = floatArrayOf(
            0f, 0f,   // node 0
            4f, 6f,   // node 1 = (1,2) + (3,4)
            5f, 6f    // node 2 = (5,6)
        )
        var maxDiff = 0f
        for (i in expected.indices) {
            val d = abs(out[i] - expected[i])
            if (d > maxDiff) maxDiff = d
        }
        return maxDiff < 1e-6f
    }
}
