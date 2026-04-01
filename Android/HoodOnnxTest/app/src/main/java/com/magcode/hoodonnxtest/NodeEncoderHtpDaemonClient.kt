package com.magcode.hoodonnxtest

import android.net.LocalSocket
import android.net.LocalSocketAddress
import android.util.Log
import java.io.EOFException
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

object NodeEncoderHtpDaemonClient {
    private const val TAG = "HoodOnnxTest"
    private const val MAGIC = 0x48445450.toInt()
    private const val VERSION = 1
    private const val OP_RUN_NODE_ENCODER_STAGE = 1
    private const val STATUS_OK = 0
    private const val STATUS_BAD_REQUEST = 1
    private const val STATUS_RUNTIME_ERROR = 2
    private const val SOCKET_WAIT_MS = 5_000L
    private const val CONNECT_RETRY_MS = 50L
    private const val EXECUTABLE_NAME = "libhood_htp_daemon.so"

    data class RunResult(
        val output: FloatArray,
        val qnnMs: Double,
        val layerNormMs: Double,
        val daemonTotalMs: Double,
        val roundTripMs: Double
    )

    private var process: Process? = null
    private var socket: LocalSocket? = null
    private var input: InputStream? = null
    private var output: OutputStream? = null
    private var socketPath: String? = null
    private var bundleDir: String? = null
    private var nativeLibDir: String? = null

    @Synchronized
    fun init(filesDir: File, nativeLibraryDir: String, bundleDir: File): String {
        this.socketPath = File(filesDir, "qnn_node_encoder_htp.sock").absolutePath
        this.bundleDir = bundleDir.absolutePath
        this.nativeLibDir = nativeLibraryDir
        return tryInitWithRestart(allowRestart = true)
    }

    @Synchronized
    fun runNodeEncoderStage(inputFloats: FloatArray): RunResult {
        ensureConnected()
        return runOnce(inputFloats, allowRestart = true)
    }

    @Synchronized
    fun close(): String {
        closeSocket()
        val proc = process
        process = null
        if (proc != null) {
            proc.destroy()
            try {
                proc.waitFor()
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        socketPath?.let { File(it).delete() }
        return "closed"
    }

    private fun tryInitWithRestart(allowRestart: Boolean): String {
        return try {
            ensureConnected()
            "connected socket=${socketPath}"
        } catch (t: Throwable) {
            if (!allowRestart) {
                "init_failed: ${t.message}"
            } else {
                Log.w(TAG, "HTP daemon init failed, restarting once: ${t.message}")
                hardReset()
                try {
                    ensureConnected()
                    "connected_after_restart socket=${socketPath}"
                } catch (second: Throwable) {
                    "init_failed_after_restart: ${second.message}"
                }
            }
        }
    }

    private fun ensureConnected() {
        if (socket?.isConnected == true && input != null && output != null) return
        val exePath = File(nativeLibDir ?: error("nativeLibraryDir not set"), EXECUTABLE_NAME)
        check(exePath.exists()) { "daemon executable missing: ${exePath.absolutePath}" }
        val socketFile = File(socketPath ?: error("socketPath not set"))
        socketFile.delete()

        if (process == null || !(process?.isAlive ?: false)) {
            val bundle = bundleDir ?: error("bundleDir not set")
            process = ProcessBuilder(exePath.absolutePath, bundle, socketFile.absolutePath)
                .apply {
                    environment()["LD_LIBRARY_PATH"] =
                        "${exePath.parentFile.absolutePath}:/vendor/lib64:/system/lib64:/vendor/lib:/system/lib"
                    redirectErrorStream(true)
                }
                .start()
        }

        val deadline = System.currentTimeMillis() + SOCKET_WAIT_MS
        var lastError: Throwable? = null
        while (System.currentTimeMillis() < deadline) {
            try {
                val newSocket = LocalSocket()
                newSocket.connect(LocalSocketAddress(socketFile.absolutePath, LocalSocketAddress.Namespace.FILESYSTEM))
                socket = newSocket
                input = newSocket.inputStream
                output = newSocket.outputStream
                return
            } catch (t: Throwable) {
                lastError = t
                Thread.sleep(CONNECT_RETRY_MS)
            }
        }
        throw IllegalStateException("socket connect timeout: ${lastError?.message}")
    }

    private fun runOnce(inputFloats: FloatArray, allowRestart: Boolean): RunResult {
        return try {
            val out = output ?: error("output stream missing")
            val input = input ?: error("input stream missing")
            val requestHeader = ByteBuffer.allocate(16)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(MAGIC)
                .putInt(VERSION)
                .putInt(OP_RUN_NODE_ENCODER_STAGE)
                .putInt(inputFloats.size)
                .array()
            val payload = ByteBuffer.allocate(inputFloats.size * 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .also { buffer -> inputFloats.forEach { buffer.putFloat(it) } }
                .array()
            val startNs = System.nanoTime()
            out.write(requestHeader)
            out.write(payload)
            out.flush()

            val responseHeaderBytes = readExact(input, 40)
            val responseHeader = ByteBuffer.wrap(responseHeaderBytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = responseHeader.int
            val version = responseHeader.int
            val status = responseHeader.int
            val outputFloatCount = responseHeader.int
            val qnnNs = responseHeader.long
            val layerNormNs = responseHeader.long
            val totalNs = responseHeader.long
            check(magic == MAGIC) { "bad response magic: $magic" }
            check(version == VERSION) { "bad response version: $version" }
            check(status == STATUS_OK) { "daemon status=$status" }
            val outputPayload = readExact(input, outputFloatCount * 4)
            val endNs = System.nanoTime()
            val outputFloatsArray = FloatArray(outputFloatCount)
            ByteBuffer.wrap(outputPayload).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(outputFloatsArray)
            RunResult(
                output = outputFloatsArray,
                qnnMs = qnnNs / 1_000_000.0,
                layerNormMs = layerNormNs / 1_000_000.0,
                daemonTotalMs = totalNs / 1_000_000.0,
                roundTripMs = (endNs - startNs) / 1_000_000.0
            )
        } catch (t: Throwable) {
            if (!allowRestart) throw t
            Log.w(TAG, "HTP daemon run failed, restarting once: ${t.message}")
            hardReset()
            ensureConnected()
            runOnce(inputFloats, allowRestart = false)
        }
    }

    private fun hardReset() {
        closeSocket()
        process?.destroy()
        process = null
        socketPath?.let { File(it).delete() }
    }

    private fun closeSocket() {
        try {
            input?.close()
        } catch (_: Throwable) {
        }
        try {
            output?.close()
        } catch (_: Throwable) {
        }
        try {
            socket?.close()
        } catch (_: Throwable) {
        }
        input = null
        output = null
        socket = null
    }

    private fun readExact(input: InputStream, size: Int): ByteArray {
        val buffer = ByteArray(size)
        var offset = 0
        while (offset < size) {
            val read = input.read(buffer, offset, size - offset)
            if (read < 0) throw EOFException("expected $size bytes, got $offset")
            offset += read
        }
        return buffer
    }
}
