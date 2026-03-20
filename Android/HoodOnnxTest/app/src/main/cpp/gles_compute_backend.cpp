#include <jni.h>

#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kTag = "HoodOnnxTest";
constexpr EGLint EGL_OPENGL_ES3_BIT_KHR = 0x00000040;

void LogInfo(const std::string& msg) {
  __android_log_print(ANDROID_LOG_INFO, kTag, "%s", msg.c_str());
}

void LogWarn(const std::string& msg) {
  __android_log_print(ANDROID_LOG_WARN, kTag, "%s", msg.c_str());
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<float> ReadFloatFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<float> out(size / sizeof(float));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
  return out;
}

int ParseJsonInt(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) return 0;
  const auto colon = json.find(':', key_pos);
  const auto start = json.find_first_of("-0123456789", colon);
  const auto end = json.find_first_not_of("0123456789", start);
  return std::stoi(json.substr(start, end - start));
}

float ParseJsonFloat(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) return 0.0f;
  const auto colon = json.find(':', key_pos);
  const auto start = json.find_first_of("-0123456789.eE", colon);
  const auto end = json.find_first_not_of("0123456789.eE+-", start);
  return std::stof(json.substr(start, end - start));
}

GLuint CompileProgram(const char* source) {
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    GLint len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    std::string info(static_cast<size_t>(len), '\0');
    glGetShaderInfoLog(shader, len, nullptr, info.data());
    glDeleteShader(shader);
    throw std::runtime_error("Compute shader compile failed: " + info);
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    GLint len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
    std::string info(static_cast<size_t>(len), '\0');
    glGetProgramInfoLog(program, len, nullptr, info.data());
    glDeleteProgram(program);
    throw std::runtime_error("Compute program link failed: " + info);
  }
  return program;
}

GLuint CreateSsbo(const std::vector<float>& data) {
  GLuint id = 0;
  glGenBuffers(1, &id);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
  glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return id;
}

GLuint CreateSsbo(size_t float_count) {
  GLuint id = 0;
  glGenBuffers(1, &id);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
  glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(float_count * sizeof(float)), nullptr, GL_DYNAMIC_COPY);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return id;
}

void UpdateSsbo(GLuint id, const float* data, size_t float_count) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(float_count * sizeof(float)), data);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

std::vector<float> ReadSsbo(GLuint id, size_t float_count) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
  auto* ptr = static_cast<float*>(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(float_count * sizeof(float)), GL_MAP_READ_BIT));
  if (!ptr) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    throw std::runtime_error("glMapBufferRange failed");
  }
  std::vector<float> out(float_count);
  std::copy(ptr, ptr + float_count, out.begin());
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return out;
}

struct EglState {
  EGLDisplay display = EGL_NO_DISPLAY;
  EGLContext context = EGL_NO_CONTEXT;
  EGLSurface surface = EGL_NO_SURFACE;

  void Init() {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) throw std::runtime_error("eglGetDisplay failed");
    EGLint major = 0, minor = 0;
    if (!eglInitialize(display, &major, &minor)) throw std::runtime_error("eglInitialize failed");

    const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE};
    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
      throw std::runtime_error("eglChooseConfig failed");
    }

    const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext failed");

    const EGLint surface_attribs[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE};
    surface = eglCreatePbufferSurface(display, config, surface_attribs);
    if (surface == EGL_NO_SURFACE) throw std::runtime_error("eglCreatePbufferSurface failed");

    if (!eglMakeCurrent(display, surface, surface, context)) throw std::runtime_error("eglMakeCurrent failed");
  }

  void Close() {
    if (display != EGL_NO_DISPLAY) {
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
      if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
      eglTerminate(display);
    }
    display = EGL_NO_DISPLAY;
    context = EGL_NO_CONTEXT;
    surface = EGL_NO_SURFACE;
  }
};

struct BlockNodeMlpState {
  bool initialized = false;
  int input_dim = 0;
  int hidden_dim = 0;
  int output_dim = 0;
  float epsilon = 1e-6f;
  int capacity_rows = 0;
  EglState egl;
  GLuint dense_program = 0;
  GLuint norm_program = 0;
  GLuint w0 = 0, b0 = 0, w1 = 0, b1 = 0, w2 = 0, b2 = 0, gamma = 0, beta = 0;
  GLuint input = 0, hidden0 = 0, hidden1 = 0, hidden2 = 0, output = 0;

  void ReleaseScratch() {
    GLuint ids[] = {input, hidden0, hidden1, hidden2, output};
    glDeleteBuffers(5, ids);
    input = hidden0 = hidden1 = hidden2 = output = 0;
    capacity_rows = 0;
  }

  void Close() {
    if (initialized) {
      ReleaseScratch();
      GLuint buffers[] = {w0, b0, w1, b1, w2, b2, gamma, beta};
      glDeleteBuffers(8, buffers);
      w0 = b0 = w1 = b1 = w2 = b2 = gamma = beta = 0;
      if (dense_program) glDeleteProgram(dense_program);
      if (norm_program) glDeleteProgram(norm_program);
      dense_program = norm_program = 0;
      egl.Close();
    }
    *this = {};
  }

  void EnsureCapacity(int rows) {
    if (rows <= capacity_rows) return;
    ReleaseScratch();
    input = CreateSsbo(static_cast<size_t>(rows) * input_dim);
    hidden0 = CreateSsbo(static_cast<size_t>(rows) * hidden_dim);
    hidden1 = CreateSsbo(static_cast<size_t>(rows) * hidden_dim);
    hidden2 = CreateSsbo(static_cast<size_t>(rows) * output_dim);
    output = CreateSsbo(static_cast<size_t>(rows) * output_dim);
    capacity_rows = rows;
  }
};

std::mutex g_mutex;
BlockNodeMlpState g_block_node_state;

const char* kDenseShader = R"(#version 310 es
precision highp float;
precision highp int;
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
    if (int(outCol) >= outDim || int(row) >= rows) return;
    int inputBase = int(row) * inDim;
    int weightBase = int(outCol) * inDim;
    float sum = biasData[int(outCol)];
    for (int k = 0; k < inDim; ++k) {
        sum += inputData[inputBase + k] * weightData[weightBase + k];
    }
    if (applyRelu != 0 && sum < 0.0) sum = 0.0;
    outputData[int(row) * outDim + int(outCol)] = sum;
})";

const char* kLayerNormShader = R"(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer InputBuffer { float inputData[]; };
layout(std430, binding = 1) readonly buffer GammaBuffer { float gammaData[]; };
layout(std430, binding = 2) readonly buffer BetaBuffer { float betaData[]; };
layout(std430, binding = 3) writeonly buffer OutputBuffer { float outputData[]; };
uniform int rows;
uniform int dim;
uniform float epsilon;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (int(row) >= rows) return;
    int base = int(row) * dim;
    float mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += inputData[base + i];
    mean /= float(dim);
    float var = 0.0;
    for (int i = 0; i < dim; ++i) {
        float c = inputData[base + i] - mean;
        var += c * c;
    }
    float invStd = inversesqrt(var / float(dim) + epsilon);
    for (int i = 0; i < dim; ++i) {
        float c = inputData[base + i] - mean;
        outputData[base + i] = c * invStd * gammaData[i] + betaData[i];
    }
})";

void SetInt(GLuint program, const char* name, int value) {
  const GLint location = glGetUniformLocation(program, name);
  if (location >= 0) glUniform1i(location, value);
}

void SetFloat(GLuint program, const char* name, float value) {
  const GLint location = glGetUniformLocation(program, name);
  if (location >= 0) glUniform1f(location, value);
}

void RunDense(BlockNodeMlpState& state, GLuint input_buf, GLuint weight_buf, GLuint bias_buf, GLuint output_buf,
              int rows, int in_dim, int out_dim, bool relu) {
  glUseProgram(state.dense_program);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buf);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, weight_buf);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bias_buf);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, output_buf);
  SetInt(state.dense_program, "rows", rows);
  SetInt(state.dense_program, "inDim", in_dim);
  SetInt(state.dense_program, "outDim", out_dim);
  SetInt(state.dense_program, "applyRelu", relu ? 1 : 0);
  glDispatchCompute((out_dim + 15) / 16, (rows + 15) / 16, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void RunLayerNorm(BlockNodeMlpState& state, GLuint input_buf, GLuint output_buf, int rows, int dim) {
  glUseProgram(state.norm_program);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buf);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.gamma);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, state.beta);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, output_buf);
  SetInt(state.norm_program, "rows", rows);
  SetInt(state.norm_program, "dim", dim);
  SetFloat(state.norm_program, "epsilon", state.epsilon);
  glDispatchCompute(rows, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

std::vector<float> RunLayerNormCpu(BlockNodeMlpState& state, const std::vector<float>& input, int rows, int dim) {
  const auto gamma = ReadSsbo(state.gamma, static_cast<size_t>(dim));
  const auto beta = ReadSsbo(state.beta, static_cast<size_t>(dim));
  std::vector<float> output(static_cast<size_t>(rows) * dim);
  for (int row = 0; row < rows; ++row) {
    const int base = row * dim;
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) mean += input[base + i];
    mean /= static_cast<float>(dim);
    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
      const float centered = input[base + i] - mean;
      var += centered * centered;
    }
    const float inv_std = 1.0f / std::sqrt(var / static_cast<float>(dim) + state.epsilon);
    for (int i = 0; i < dim; ++i) {
      const float centered = input[base + i] - mean;
      output[base + i] = centered * inv_std * gamma[i] + beta[i];
    }
  }
  return output;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_magcode_hoodonnxtest_NativeGpuBridge_initBlockNodeMlp(
    JNIEnv* env,
    jobject /* thiz */,
    jstring asset_dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_block_node_state.Close();

  if (asset_dir == nullptr) {
    LogWarn("initBlockNodeMlp called with null assetDir");
    return JNI_FALSE;
  }

  const char* chars = env->GetStringUTFChars(asset_dir, nullptr);
  if (chars == nullptr) return JNI_FALSE;
  const std::string base_dir(chars);
  env->ReleaseStringUTFChars(asset_dir, chars);

  try {
    const std::string manifest = ReadTextFile(base_dir + "/manifest.json");
    g_block_node_state.input_dim = ParseJsonInt(manifest, "input_dim");
    g_block_node_state.hidden_dim = ParseJsonInt(manifest, "hidden_dim");
    g_block_node_state.output_dim = ParseJsonInt(manifest, "output_dim");
    g_block_node_state.epsilon = ParseJsonFloat(manifest, "epsilon");

    g_block_node_state.egl.Init();
    g_block_node_state.dense_program = CompileProgram(kDenseShader);
    g_block_node_state.norm_program = CompileProgram(kLayerNormShader);

    g_block_node_state.w0 = CreateSsbo(ReadFloatFile(base_dir + "/linear0_weight.bin"));
    g_block_node_state.b0 = CreateSsbo(ReadFloatFile(base_dir + "/linear0_bias.bin"));
    g_block_node_state.w1 = CreateSsbo(ReadFloatFile(base_dir + "/linear1_weight.bin"));
    g_block_node_state.b1 = CreateSsbo(ReadFloatFile(base_dir + "/linear1_bias.bin"));
    g_block_node_state.w2 = CreateSsbo(ReadFloatFile(base_dir + "/linear2_weight.bin"));
    g_block_node_state.b2 = CreateSsbo(ReadFloatFile(base_dir + "/linear2_bias.bin"));
    g_block_node_state.gamma = CreateSsbo(ReadFloatFile(base_dir + "/layernorm_gamma.bin"));
    g_block_node_state.beta = CreateSsbo(ReadFloatFile(base_dir + "/layernorm_beta.bin"));
    g_block_node_state.initialized = true;
    LogInfo("Native block-node MLP initialized");
    return JNI_TRUE;
  } catch (const std::exception& ex) {
    LogWarn(std::string("Native block-node MLP init failed: ") + ex.what());
    g_block_node_state.Close();
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeGpuBridge_runBlockNodeMlp(
    JNIEnv* env,
    jobject /* thiz */,
    jfloatArray input,
    jint rows,
    jint cols) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_block_node_state.initialized || input == nullptr) return nullptr;
  if (rows <= 0 || cols != g_block_node_state.input_dim) return nullptr;

  const jsize length = env->GetArrayLength(input);
  if (length != rows * cols) return nullptr;

  std::vector<float> host_input(static_cast<size_t>(length));
  env->GetFloatArrayRegion(input, 0, length, host_input.data());

  try {
    g_block_node_state.EnsureCapacity(rows);
    UpdateSsbo(g_block_node_state.input, host_input.data(), host_input.size());
    RunDense(g_block_node_state, g_block_node_state.input, g_block_node_state.w0, g_block_node_state.b0, g_block_node_state.hidden0,
             rows, g_block_node_state.input_dim, g_block_node_state.hidden_dim, true);
    RunDense(g_block_node_state, g_block_node_state.hidden0, g_block_node_state.w1, g_block_node_state.b1, g_block_node_state.hidden1,
             rows, g_block_node_state.hidden_dim, g_block_node_state.hidden_dim, true);
    RunDense(g_block_node_state, g_block_node_state.hidden1, g_block_node_state.w2, g_block_node_state.b2, g_block_node_state.hidden2,
             rows, g_block_node_state.hidden_dim, g_block_node_state.output_dim, false);
    const auto hidden = ReadSsbo(g_block_node_state.hidden2, static_cast<size_t>(rows) * g_block_node_state.output_dim);
    const auto result = RunLayerNormCpu(g_block_node_state, hidden, rows, g_block_node_state.output_dim);

    jfloatArray out = env->NewFloatArray(static_cast<jsize>(result.size()));
    if (!out) return nullptr;
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(result.size()), result.data());
    return out;
  } catch (const std::exception& ex) {
    LogWarn(std::string("Native block-node MLP run failed: ") + ex.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_magcode_hoodonnxtest_NativeGpuBridge_closeBlockNodeMlp(
    JNIEnv* /* env */,
    jobject /* thiz */) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_block_node_state.Close();
}
