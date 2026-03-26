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
}
