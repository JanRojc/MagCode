package com.magcode.hoodonnxtest

import android.util.Log

internal object NativeGpuBridge {
    private var loaded = false

    init {
        try {
            System.loadLibrary("hood_gpu")
            loaded = true
        } catch (t: Throwable) {
            Log.w("HoodOnnxTest", "Native GPU bridge unavailable: ${t.message}")
        }
    }

    fun isLoaded(): Boolean = loaded

    external fun initBlockNodeMlp(assetDir: String): Boolean

    external fun runBlockNodeMlp(input: FloatArray, rows: Int, cols: Int): FloatArray?

    external fun closeBlockNodeMlp()
}
