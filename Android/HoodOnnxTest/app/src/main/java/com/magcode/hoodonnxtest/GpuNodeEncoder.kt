package com.magcode.hoodonnxtest

import android.content.res.AssetManager
import android.util.Log
import org.json.JSONObject

internal class GpuNodeEncoder private constructor(
    private val gl: GlComputeContext,
    private val inputDim: Int,
    private val hiddenDim: Int,
    private val outputDim: Int,
    private val epsilon: Float,
    private val dense0Program: Int,
    private val dense1Program: Int,
    private val dense2Program: Int,
    private val normProgram: Int,
    private val dense0Weight: Int,
    private val dense0Bias: Int,
    private val dense1Weight: Int,
    private val dense1Bias: Int,
    private val dense2Weight: Int,
    private val dense2Bias: Int,
    private val gamma: Int,
    private val beta: Int,
    private var inputBuffer: Int = 0,
    private var hidden0Buffer: Int = 0,
    private var hidden1Buffer: Int = 0,
    private var hidden2Buffer: Int = 0,
    private var outputBuffer: Int = 0,
    private var bufferCapacityRows: Int = 0
) : AutoCloseable {
    fun run(input: FloatArray, rows: Int): FloatArray {
        require(rows >= 0) { "rows must be non-negative" }
        require(input.size == rows * inputDim) { "Expected ${rows * inputDim} floats, got ${input.size}" }

        ensureCapacity(rows)
        gl.updateFloatSsbo(inputBuffer, input)
        runDense(dense0Program, inputBuffer, dense0Weight, dense0Bias, hidden0Buffer, rows, inputDim, hiddenDim, relu = true)
        runDense(dense1Program, hidden0Buffer, dense1Weight, dense1Bias, hidden1Buffer, rows, hiddenDim, hiddenDim, relu = true)
        runDense(dense2Program, hidden1Buffer, dense2Weight, dense2Bias, hidden2Buffer, rows, hiddenDim, outputDim, relu = false)
        runLayerNorm(hidden2Buffer, gamma, beta, outputBuffer, rows, outputDim)
        return gl.readFloatSsbo(outputBuffer, rows * outputDim)
    }

    private fun ensureCapacity(rows: Int) {
        if (rows <= bufferCapacityRows) return
        releaseScratch()
        inputBuffer = gl.createFloatSsbo(capacityFloats = rows * inputDim)
        hidden0Buffer = gl.createFloatSsbo(capacityFloats = rows * hiddenDim)
        hidden1Buffer = gl.createFloatSsbo(capacityFloats = rows * hiddenDim)
        hidden2Buffer = gl.createFloatSsbo(capacityFloats = rows * outputDim)
        outputBuffer = gl.createFloatSsbo(capacityFloats = rows * outputDim)
        bufferCapacityRows = rows
    }

    private fun releaseScratch() {
        gl.deleteBuffer(inputBuffer)
        gl.deleteBuffer(hidden0Buffer)
        gl.deleteBuffer(hidden1Buffer)
        gl.deleteBuffer(hidden2Buffer)
        gl.deleteBuffer(outputBuffer)
        inputBuffer = 0
        hidden0Buffer = 0
        hidden1Buffer = 0
        hidden2Buffer = 0
        outputBuffer = 0
        bufferCapacityRows = 0
    }

    private fun runDense(
        program: Int,
        inputBuffer: Int,
        weightBuffer: Int,
        biasBuffer: Int,
        outputBuffer: Int,
        rows: Int,
        inDim: Int,
        outDim: Int,
        relu: Boolean
    ) {
        gl.bindSsbo(0, inputBuffer)
        gl.bindSsbo(1, weightBuffer)
        gl.bindSsbo(2, biasBuffer)
        gl.bindSsbo(3, outputBuffer)
        gl.setInt(program, "rows", rows)
        gl.setInt(program, "inDim", inDim)
        gl.setInt(program, "outDim", outDim)
        gl.setInt(program, "applyRelu", if (relu) 1 else 0)
        val groupsX = (outDim + 15) / 16
        val groupsY = (rows + 15) / 16
        gl.dispatch(program, groupsX, groupsY)
    }

    private fun runLayerNorm(
        inputBuffer: Int,
        gammaBuffer: Int,
        betaBuffer: Int,
        outputBuffer: Int,
        rows: Int,
        dim: Int
    ) {
        require(dim == 128) { "LayerNorm shader currently expects dim=128, got $dim" }
        gl.bindSsbo(0, inputBuffer)
        gl.bindSsbo(1, gammaBuffer)
        gl.bindSsbo(2, betaBuffer)
        gl.bindSsbo(3, outputBuffer)
        gl.setInt(normProgram, "rows", rows)
        gl.setInt(normProgram, "dim", dim)
        gl.setFloat(normProgram, "epsilon", epsilon)
        gl.dispatch(normProgram, rows)
    }

    override fun close() {
        releaseScratch()
        gl.deleteBuffer(dense0Weight)
        gl.deleteBuffer(dense0Bias)
        gl.deleteBuffer(dense1Weight)
        gl.deleteBuffer(dense1Bias)
        gl.deleteBuffer(dense2Weight)
        gl.deleteBuffer(dense2Bias)
        gl.deleteBuffer(gamma)
        gl.deleteBuffer(beta)
        gl.deleteProgram(dense0Program)
        gl.deleteProgram(dense1Program)
        gl.deleteProgram(dense2Program)
        gl.deleteProgram(normProgram)
        gl.close()
    }

    companion object {
        fun tryCreate(assets: AssetManager): GpuNodeEncoder? {
            if (!assetExists(assets, "gpu/node_encoder/manifest.json")) return null
            return try {
                val manifest = JSONObject(assets.open("gpu/node_encoder/manifest.json").bufferedReader().use { it.readText() })
                val gl = GlComputeContext()
                val denseProgram0 = gl.createProgram(DENSE_SHADER)
                val denseProgram1 = gl.createProgram(DENSE_SHADER)
                val denseProgram2 = gl.createProgram(DENSE_SHADER)
                val normProgram = gl.createProgram(LAYER_NORM_SHADER)
                val base = "gpu/node_encoder"
                val encoder = GpuNodeEncoder(
                    gl = gl,
                    inputDim = manifest.getInt("input_dim"),
                    hiddenDim = manifest.getInt("hidden_dim"),
                    outputDim = manifest.getInt("output_dim"),
                    epsilon = manifest.optDouble("epsilon", 1e-6).toFloat(),
                    dense0Program = denseProgram0,
                    dense1Program = denseProgram1,
                    dense2Program = denseProgram2,
                    normProgram = normProgram,
                    dense0Weight = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear0_weight.bin")),
                    dense0Bias = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear0_bias.bin")),
                    dense1Weight = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear1_weight.bin")),
                    dense1Bias = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear1_bias.bin")),
                    dense2Weight = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear2_weight.bin")),
                    dense2Bias = gl.createFloatSsbo(readFloatBinary(assets, "$base/linear2_bias.bin")),
                    gamma = gl.createFloatSsbo(readFloatBinary(assets, "$base/layernorm_gamma.bin")),
                    beta = gl.createFloatSsbo(readFloatBinary(assets, "$base/layernorm_beta.bin"))
                )
                Log.i("HoodOnnxTest", "Enabled GPU node encoder")
                encoder
            } catch (t: Throwable) {
                Log.w("HoodOnnxTest", "GPU node encoder unavailable: ${t.message}")
                null
            }
        }

        private fun assetExists(assets: AssetManager, assetName: String): Boolean {
            return try {
                assets.open(assetName).close()
                true
            } catch (_: Exception) {
                false
            }
        }

        private fun readFloatBinary(assets: AssetManager, assetName: String): FloatArray {
            val bytes = assets.open(assetName).readBytes()
            val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            return FloatArray(bytes.size / 4) { buf.float }
        }

        private const val DENSE_SHADER = """
#version 310 es
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer { float inputData[]; };
layout(std430, binding = 1) readonly buffer WeightBuffer { float weightData[]; };
layout(std430, binding = 2) readonly buffer BiasBuffer { float biasData[]; };
layout(std430, binding = 3) writeonly buffer OutputBuffer { float outputData[]; };

uniform int rows;
uniform int inDim;
uniform int outDim;
uniform int applyRelu;

void main() {
    uint outCol = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    if (int(outCol) >= outDim || int(row) >= rows) {
        return;
    }

    int inputBase = int(row) * inDim;
    int weightBase = int(outCol) * inDim;
    float sum = biasData[int(outCol)];
    for (int k = 0; k < inDim; ++k) {
        sum += inputData[inputBase + k] * weightData[weightBase + k];
    }
    if (applyRelu != 0 && sum < 0.0) {
        sum = 0.0;
    }
    outputData[int(row) * outDim + int(outCol)] = sum;
}
"""

        private const val LAYER_NORM_SHADER = """
#version 310 es
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer { float inputData[]; };
layout(std430, binding = 1) readonly buffer GammaBuffer { float gammaData[]; };
layout(std430, binding = 2) readonly buffer BetaBuffer { float betaData[]; };
layout(std430, binding = 3) writeonly buffer OutputBuffer { float outputData[]; };

uniform int rows;
uniform int dim;
uniform float epsilon;

shared float scratch[128];

void main() {
    uint row = gl_WorkGroupID.x;
    uint col = gl_LocalInvocationID.x;
    if (int(row) >= rows || int(col) >= dim) {
        return;
    }

    int idx = int(row) * dim + int(col);
    float value = inputData[idx];
    scratch[int(col)] = value;
    barrier();

    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (col < stride) {
            scratch[int(col)] += scratch[int(col + stride)];
        }
        barrier();
    }
    float mean = scratch[0] / float(dim);

    float centered = value - mean;
    scratch[int(col)] = centered * centered;
    barrier();

    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (col < stride) {
            scratch[int(col)] += scratch[int(col + stride)];
        }
        barrier();
    }
    float invStd = inversesqrt((scratch[0] / float(dim)) + epsilon);
    outputData[idx] = centered * invStd * gammaData[int(col)] + betaData[int(col)];
}
"""
    }
}
