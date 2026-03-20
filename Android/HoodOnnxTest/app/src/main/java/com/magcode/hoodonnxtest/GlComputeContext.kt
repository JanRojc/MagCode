package com.magcode.hoodonnxtest

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES31

internal class GlComputeContext : AutoCloseable {
    private val display: EGLDisplay
    private val context: EGLContext
    private val surface: EGLSurface

    init {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        require(display != EGL14.EGL_NO_DISPLAY) { "eglGetDisplay failed" }
        val version = IntArray(2)
        require(EGL14.eglInitialize(display, version, 0, version, 1)) { "eglInitialize failed" }

        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        val configAttrs = intArrayOf(
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RENDERABLE_TYPE, 0x40,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )
        require(EGL14.eglChooseConfig(display, configAttrs, 0, configs, 0, 1, numConfigs, 0) && numConfigs[0] > 0) {
            "eglChooseConfig failed"
        }
        val config = configs[0] ?: error("No EGL config")

        val contextAttrs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL14.EGL_NONE
        )
        context = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT, contextAttrs, 0)
        require(context != null && context != EGL14.EGL_NO_CONTEXT) { "eglCreateContext failed" }

        val surfaceAttrs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
        surface = EGL14.eglCreatePbufferSurface(display, config, surfaceAttrs, 0)
        require(surface != null && surface != EGL14.EGL_NO_SURFACE) { "eglCreatePbufferSurface failed" }

        require(EGL14.eglMakeCurrent(display, surface, surface, context)) { "eglMakeCurrent failed" }
    }

    fun createProgram(shaderSource: String): Int {
        val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        require(shader != 0) { "glCreateShader failed" }
        GLES31.glShaderSource(shader, shaderSource)
        GLES31.glCompileShader(shader)
        val compileStatus = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compileStatus, 0)
        if (compileStatus[0] == 0) {
            val info = GLES31.glGetShaderInfoLog(shader)
            GLES31.glDeleteShader(shader)
            error("Compute shader compile failed: $info")
        }

        val program = GLES31.glCreateProgram()
        require(program != 0) { "glCreateProgram failed" }
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)
        GLES31.glDeleteShader(shader)

        val linkStatus = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] == 0) {
            val info = GLES31.glGetProgramInfoLog(program)
            GLES31.glDeleteProgram(program)
            error("Compute program link failed: $info")
        }
        return program
    }

    fun createFloatSsbo(data: FloatArray? = null, capacityFloats: Int = data?.size ?: 0): Int {
        require(capacityFloats >= 0) { "capacityFloats must be non-negative" }
        val ids = IntArray(1)
        GLES31.glGenBuffers(1, ids, 0)
        val id = ids[0]
        require(id != 0) { "glGenBuffers failed" }
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, id)
        val byteCount = capacityFloats * 4
        val buffer = if (data != null) {
            java.nio.ByteBuffer.allocateDirect(byteCount)
                .order(java.nio.ByteOrder.nativeOrder())
                .asFloatBuffer()
                .apply {
                    put(data)
                    rewind()
                }
        } else {
            null
        }
        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, byteCount, buffer, GLES31.GL_DYNAMIC_COPY)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
        return id
    }

    fun updateFloatSsbo(bufferId: Int, data: FloatArray) {
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, bufferId)
        val fb = java.nio.ByteBuffer.allocateDirect(data.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
            .asFloatBuffer()
        fb.put(data)
        fb.rewind()
        GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, data.size * 4, fb)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
    }

    fun readFloatSsbo(bufferId: Int, sizeFloats: Int): FloatArray {
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, bufferId)
        val mapped = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            sizeFloats * 4,
            GLES31.GL_MAP_READ_BIT
        ) ?: error("glMapBufferRange returned null")
        val out = FloatArray(sizeFloats)
        (mapped as java.nio.ByteBuffer).order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer().get(out)
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
        return out
    }

    fun bindSsbo(binding: Int, bufferId: Int) {
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, binding, bufferId)
    }

    fun dispatch(program: Int, groupsX: Int, groupsY: Int = 1, groupsZ: Int = 1) {
        GLES31.glUseProgram(program)
        GLES31.glDispatchCompute(groupsX, groupsY, groupsZ)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT or GLES31.GL_BUFFER_UPDATE_BARRIER_BIT)
        GLES31.glUseProgram(0)
    }

    fun setInt(program: Int, name: String, value: Int) {
        val location = GLES31.glGetUniformLocation(program, name)
        require(location >= 0) { "Uniform $name missing" }
        GLES31.glUseProgram(program)
        GLES31.glUniform1i(location, value)
        GLES31.glUseProgram(0)
    }

    fun setFloat(program: Int, name: String, value: Float) {
        val location = GLES31.glGetUniformLocation(program, name)
        require(location >= 0) { "Uniform $name missing" }
        GLES31.glUseProgram(program)
        GLES31.glUniform1f(location, value)
        GLES31.glUseProgram(0)
    }

    fun deleteBuffer(bufferId: Int) {
        if (bufferId == 0) return
        GLES31.glDeleteBuffers(1, intArrayOf(bufferId), 0)
    }

    fun deleteProgram(programId: Int) {
        if (programId == 0) return
        GLES31.glDeleteProgram(programId)
    }

    override fun close() {
        EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(display, surface)
        EGL14.eglDestroyContext(display, context)
        EGL14.eglTerminate(display)
    }
}
