#include <jni.h>

#include <android/NeuralNetworks.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnModel.hpp"
#include "System/QnnSystemInterface.h"

namespace {

using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);
using QnnSystemInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);
using ComposeGraphsFn_t = qnn_wrapper_api::ModelError_t (*)(Qnn_BackendHandle_t,
                                                            QNN_INTERFACE_VER_TYPE,
                                                            Qnn_ContextHandle_t,
                                                            const qnn_wrapper_api::GraphConfigInfo_t**,
                                                            uint32_t,
                                                            qnn_wrapper_api::GraphInfo_t***,
                                                            uint32_t*,
                                                            bool,
                                                            QnnLog_Callback_t,
                                                            QnnLog_Level_t);
using FreeGraphsInfoFn_t =
    qnn_wrapper_api::ModelError_t (*)(qnn_wrapper_api::GraphInfo_t***, uint32_t);

struct QnnCasePaths {
  std::filesystem::path bundle;
  std::filesystem::path libcxx;
  std::filesystem::path system;
  std::filesystem::path gpuExt;
  std::filesystem::path backend;
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path expected;
  std::filesystem::path probeOutput;
};

struct CachedQnnCaseState {
  bool initialized = false;
  std::string bundleDir;
  std::string buildId;
  std::string graphName;
  void* libcxxHandle = nullptr;
  void* systemHandle = nullptr;
  void* gpuExtHandle = nullptr;
  void* backendLibHandle = nullptr;
  void* modelLibHandle = nullptr;
  const QnnInterface_t* provider = nullptr;
  Qnn_BackendHandle_t backendHandle = nullptr;
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  Qnn_ContextHandle_t contextHandle = nullptr;
  Qnn_GraphHandle_t graphHandle = nullptr;
  Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
  std::vector<uint8_t> inputBytes;
  std::vector<uint8_t> expectedBytes;
  std::vector<uint8_t> inputBuffer;
  std::vector<uint8_t> outputBuffer;
};

CachedQnnCaseState g_cachedNodeEncoderCase;

const QnnInterface_t* SelectQnnInterface(const QnnInterface_t** providers, uint32_t count);
const QnnSystemInterface_t* SelectQnnSystemInterface(const QnnSystemInterface_t** providers,
                                                     uint32_t count);

template <typename T>
T ResolveSymbol(void* handle, const char* symbol) {
  return reinterpret_cast<T>(dlsym(handle, symbol));
}

std::string DlErrorString() {
  const char* err = dlerror();
  return err == nullptr ? "<none>" : err;
}

std::string QnnErrorString(Qnn_ErrorHandle_t code) {
  std::ostringstream oss;
  oss << code;
  return oss.str();
}

std::string ModelErrorString(qnn_wrapper_api::ModelError_t code) {
  return qnn_wrapper_api::getModelErrorName(code);
}

std::filesystem::path JoinPath(const std::string& base, const std::string& child) {
  return std::filesystem::path(base) / child;
}

bool FileExists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec) && std::filesystem::is_regular_file(path, ec);
}

std::filesystem::path FindFirstMatchingFile(const std::filesystem::path& dir,
                                            const std::vector<std::string>& candidates) {
  for (const auto& candidate : candidates) {
    const auto path = dir / candidate;
    if (FileExists(path)) return path;
  }
  return {};
}

std::filesystem::path FindProbeModelLib(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::path firstGenericModelLib;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    const auto name = entry.path().filename().string();
    if (name.rfind("lib", 0) == 0 && name.find("_probe.so") != std::string::npos) {
      return entry.path();
    }
    const bool isSharedLib = name.size() >= 3 && name.compare(name.size() - 3, 3, ".so") == 0;
    if (name.rfind("lib", 0) == 0 &&
        isSharedLib &&
        name != "libc++_shared.so" &&
        name != "libQnnSystem.so" &&
        name != "libQnnGpuNetRunExtensions.so" &&
        name != "libQnnGpu.so" &&
        name != "libQnnCpu.so" &&
        name != "libQnnHtp.so" &&
        firstGenericModelLib.empty()) {
      firstGenericModelLib = entry.path();
    }
  }
  return firstGenericModelLib;
}

QnnCasePaths DiscoverCasePaths(const std::string& bundleDir) {
  const std::filesystem::path bundle(bundleDir);
  return {
      bundle,
      bundle / "libc++_shared.so",
      bundle / "libQnnSystem.so",
      bundle / "libQnnGpuNetRunExtensions.so",
      bundle / "libQnnGpu.so",
      FindProbeModelLib(bundle),
      FindFirstMatchingFile(bundle, {"node_encoder_input.raw", "input.raw"}),
      FindFirstMatchingFile(bundle, {"expected_node_encoder_output.raw", "node_encoder_probe_expected.raw", "expected_output.raw"}),
      FindFirstMatchingFile(bundle, {"output_probe/output/Result_0/output.raw", "hybrid_output.raw"}),
  };
}

std::vector<uint8_t> ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  input.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(input.tellg());
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (size > 0) {
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  }
  return bytes;
}

size_t DataTypeByteSize(Qnn_DataType_t dataType) {
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32:
      return 4;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 0;
  }
}

size_t TensorByteSize(const Qnn_Tensor_t& tensor) {
  const auto rank = QNN_TENSOR_GET_RANK(tensor);
  auto* dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
  const auto typeSize = DataTypeByteSize(QNN_TENSOR_GET_DATA_TYPE(tensor));
  if (dims == nullptr || rank == 0 || typeSize == 0) return 0;
  size_t elements = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    elements *= dims[i];
  }
  return elements * typeSize;
}

bool SetupTensorFromTemplate(const Qnn_Tensor_t& src,
                             Qnn_Tensor_t& dst,
                             std::vector<uint8_t>& ownedBuffer,
                             const uint8_t* initialData,
                             size_t initialSize) {
  if (qnn_wrapper_api::deepCopyQnnTensors(const_cast<Qnn_Tensor_t&>(src), dst) !=
      qnn_wrapper_api::MODEL_NO_ERROR) {
    return false;
  }
  const auto bytes = TensorByteSize(dst);
  if (bytes == 0) return false;
  ownedBuffer.assign(bytes, 0);
  if (initialData != nullptr) {
    if (initialSize != bytes) return false;
    std::memcpy(ownedBuffer.data(), initialData, bytes);
  }
  QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSORMEMTYPE_RAW);
  Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
  clientBuffer.data = ownedBuffer.data();
  clientBuffer.dataSize = ownedBuffer.size();
  QNN_TENSOR_SET_CLIENT_BUF(dst, clientBuffer);
  return true;
}

std::string CompareFloatBuffers(const std::vector<uint8_t>& expectedBytes,
                                const std::vector<uint8_t>& actualBytes,
                                float atol = 1e-5f,
                                float rtol = 1e-4f) {
  if (expectedBytes.size() != actualBytes.size()) {
    std::ostringstream oss;
    oss << "size_mismatch expected=" << expectedBytes.size() << " actual=" << actualBytes.size();
    return oss.str();
  }
  if ((expectedBytes.size() % sizeof(float)) != 0) {
    return "non_float_output_size";
  }
  const auto* expected = reinterpret_cast<const float*>(expectedBytes.data());
  const auto* actual = reinterpret_cast<const float*>(actualBytes.data());
  const size_t count = expectedBytes.size() / sizeof(float);
  float maxAbs = 0.0f;
  size_t mismatch = 0;
  for (size_t i = 0; i < count; ++i) {
    const float absDiff = std::fabs(expected[i] - actual[i]);
    maxAbs = std::max(maxAbs, absDiff);
    const float tol = atol + rtol * std::fabs(expected[i]);
    if (absDiff > tol) mismatch++;
  }
  std::ostringstream oss;
  oss << "max_abs=" << maxAbs << " mismatch=" << mismatch;
  return oss.str();
}

std::string FormatMs(double ms) {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << ms;
  return oss.str();
}

void DlCloseQuiet(void*& handle) {
  if (handle != nullptr) {
    dlclose(handle);
    handle = nullptr;
  }
}

void ReleaseCachedCase(CachedQnnCaseState& state) {
  if (state.initialized) {
    qnn_wrapper_api::freeQnnTensor(state.inputTensor);
    qnn_wrapper_api::freeQnnTensor(state.outputTensor);
    if (state.contextHandle != nullptr && state.provider != nullptr &&
        state.provider->QNN_INTERFACE_VER_NAME.contextFree != nullptr) {
      state.provider->QNN_INTERFACE_VER_NAME.contextFree(state.contextHandle, nullptr);
    }
    if (state.deviceHandle != nullptr && state.provider != nullptr &&
        state.provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      state.provider->QNN_INTERFACE_VER_NAME.deviceFree(state.deviceHandle);
    }
    if (state.backendHandle != nullptr && state.provider != nullptr &&
        state.provider->QNN_INTERFACE_VER_NAME.backendFree != nullptr) {
      state.provider->QNN_INTERFACE_VER_NAME.backendFree(state.backendHandle);
    }
  }
  DlCloseQuiet(state.modelLibHandle);
  DlCloseQuiet(state.backendLibHandle);
  DlCloseQuiet(state.gpuExtHandle);
  DlCloseQuiet(state.systemHandle);
  DlCloseQuiet(state.libcxxHandle);
  state = CachedQnnCaseState{};
}

std::string InitializeCachedNodeEncoderCase(const std::string& bundleDir, CachedQnnCaseState& state) {
  ReleaseCachedCase(state);
  const auto paths = DiscoverCasePaths(bundleDir);
  std::ostringstream oss;
  oss << "bundle=" << bundleDir;
  if (!FileExists(paths.libcxx) || !FileExists(paths.system) || !FileExists(paths.gpuExt) ||
      !FileExists(paths.backend) || paths.model.empty() || paths.input.empty() || paths.expected.empty()) {
    oss << " | missing_required_files";
    return oss.str();
  }

  state.bundleDir = bundleDir;
  state.inputBytes = ReadBinaryFile(paths.input);
  state.expectedBytes = ReadBinaryFile(paths.expected);

  dlerror();
  state.libcxxHandle = dlopen(paths.libcxx.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (state.libcxxHandle == nullptr) {
    oss << " | libcxx_dlopen_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }
  state.systemHandle = dlopen(paths.system.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (state.systemHandle == nullptr) {
    oss << " | system_dlopen_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }
  state.gpuExtHandle = dlopen(paths.gpuExt.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (state.gpuExtHandle == nullptr) {
    oss << " | gpu_ext_dlopen_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }
  state.backendLibHandle = dlopen(paths.backend.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (state.backendLibHandle == nullptr) {
    oss << " | backend_dlopen_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }
  state.modelLibHandle = dlopen(paths.model.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (state.modelLibHandle == nullptr) {
    oss << " | model_dlopen_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }

  const auto getProviders =
      ResolveSymbol<QnnInterfaceGetProvidersFn_t>(state.backendLibHandle, "QnnInterface_getProviders");
  if (getProviders == nullptr) {
    oss << " | providers_symbol_fail=" << DlErrorString();
    ReleaseCachedCase(state);
    return oss.str();
  }
  const QnnInterface_t** providers = nullptr;
  uint32_t providerCount = 0;
  const auto providersStatus = getProviders(&providers, &providerCount);
  state.provider = SelectQnnInterface(providers, providerCount);
  oss << " | providers=" << providerCount << "@" << QnnErrorString(providersStatus);
  if (providersStatus != QNN_SUCCESS || state.provider == nullptr) {
    oss << " | provider_select=FAIL";
    ReleaseCachedCase(state);
    return oss.str();
  }

  const auto composeGraphs = ResolveSymbol<ComposeGraphsFn_t>(state.modelLibHandle, "QnnModel_composeGraphs");
  const auto freeGraphsInfo = ResolveSymbol<FreeGraphsInfoFn_t>(state.modelLibHandle, "QnnModel_freeGraphsInfo");
  if (composeGraphs == nullptr || freeGraphsInfo == nullptr) {
    oss << " | model_symbols=FAIL";
    ReleaseCachedCase(state);
    return oss.str();
  }

  const auto backendCreateStatus =
      state.provider->QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &state.backendHandle);
  oss << " | backendCreate=" << QnnErrorString(backendCreateStatus);
  if (backendCreateStatus != QNN_SUCCESS || state.backendHandle == nullptr) {
    ReleaseCachedCase(state);
    return oss.str();
  }

  if (state.provider->QNN_INTERFACE_VER_NAME.deviceCreate != nullptr) {
    const auto deviceCreateStatus =
        state.provider->QNN_INTERFACE_VER_NAME.deviceCreate(nullptr, nullptr, &state.deviceHandle);
    oss << " | deviceCreate=" << QnnErrorString(deviceCreateStatus);
    if (deviceCreateStatus != QNN_SUCCESS &&
        deviceCreateStatus != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      ReleaseCachedCase(state);
      return oss.str();
    }
  }

  const auto contextCreateStatus = state.provider->QNN_INTERFACE_VER_NAME.contextCreate(
      state.backendHandle, state.deviceHandle, nullptr, &state.contextHandle);
  oss << " | contextCreate=" << QnnErrorString(contextCreateStatus);
  if (contextCreateStatus != QNN_SUCCESS || state.contextHandle == nullptr) {
    ReleaseCachedCase(state);
    return oss.str();
  }

  qnn_wrapper_api::GraphInfo_t** graphsInfo = nullptr;
  uint32_t graphCount = 0;
  const auto composeStatus = composeGraphs(state.backendHandle,
                                           state.provider->QNN_INTERFACE_VER_NAME,
                                           state.contextHandle,
                                           nullptr,
                                           0,
                                           &graphsInfo,
                                           &graphCount,
                                           false,
                                           nullptr,
                                           QNN_LOG_LEVEL_ERROR);
  oss << " | composeGraphs=" << ModelErrorString(composeStatus) << " graphs=" << graphCount;
  if (composeStatus != qnn_wrapper_api::MODEL_NO_ERROR || graphsInfo == nullptr || graphCount == 0 ||
      graphsInfo[0] == nullptr) {
    ReleaseCachedCase(state);
    return oss.str();
  }

  auto* graphInfo = graphsInfo[0];
  const auto finalizeStatus =
      state.provider->QNN_INTERFACE_VER_NAME.graphFinalize(graphInfo->graph, nullptr, nullptr);
  oss << " | graphFinalize=" << QnnErrorString(finalizeStatus);
  if (finalizeStatus != QNN_SUCCESS) {
    freeGraphsInfo(&graphsInfo, graphCount);
    ReleaseCachedCase(state);
    return oss.str();
  }

  if (graphInfo->numInputTensors != 1 || graphInfo->numOutputTensors != 1) {
    oss << " | graphShape=unexpected";
    freeGraphsInfo(&graphsInfo, graphCount);
    ReleaseCachedCase(state);
    return oss.str();
  }

  state.graphHandle = graphInfo->graph;
  state.graphName = graphInfo->graphName == nullptr ? "" : graphInfo->graphName;
  const bool inputOk = SetupTensorFromTemplate(
      graphInfo->inputTensors[0], state.inputTensor, state.inputBuffer, state.inputBytes.data(), state.inputBytes.size());
  const bool outputOk = SetupTensorFromTemplate(
      graphInfo->outputTensors[0], state.outputTensor, state.outputBuffer, nullptr, 0);
  oss << " | setupInput=" << (inputOk ? "OK" : "FAIL") << " setupOutput=" << (outputOk ? "OK" : "FAIL");
  const auto freeGraphsStatus = freeGraphsInfo(&graphsInfo, graphCount);
  oss << " | freeGraphsInfo=" << ModelErrorString(freeGraphsStatus);
  if (!inputOk || !outputOk || freeGraphsStatus != qnn_wrapper_api::MODEL_NO_ERROR) {
    ReleaseCachedCase(state);
    return oss.str();
  }

  if (state.provider->QNN_INTERFACE_VER_NAME.backendGetBuildId != nullptr) {
    const char* buildId = nullptr;
    if (state.provider->QNN_INTERFACE_VER_NAME.backendGetBuildId(&buildId) == QNN_SUCCESS && buildId != nullptr) {
      state.buildId = buildId;
      oss << " | buildId=" << state.buildId;
    }
  }

  state.initialized = true;
  return oss.str();
}

const QnnInterface_t* SelectQnnInterface(const QnnInterface_t** providers, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const auto* provider = providers[i];
    if (provider == nullptr) continue;
    if (provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      return provider;
    }
  }
  return nullptr;
}

const QnnSystemInterface_t* SelectQnnSystemInterface(const QnnSystemInterface_t** providers,
                                                     uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const auto* provider = providers[i];
    if (provider == nullptr) continue;
    if (provider->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        provider->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      return provider;
    }
  }
  return nullptr;
}

std::string ValidateLayerNormArgs(jsize inputSize,
                                  jint rows,
                                  jint cols,
                                  jsize gammaSize,
                                  jsize betaSize) {
  if (rows <= 0 || cols <= 0) {
    return "rows and cols must be > 0";
  }
  if (inputSize != rows * cols) {
    return "input size mismatch";
  }
  if (gammaSize != cols) {
    return "gamma size mismatch";
  }
  if (betaSize != cols) {
    return "beta size mismatch";
  }
  return "";
}

}  // namespace

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_applyLayerNorm(
    JNIEnv* env,
    jclass,
    jfloatArray input,
    jint rows,
    jint cols,
    jfloatArray gamma,
    jfloatArray beta,
    jfloat eps) {
  const jsize inputSize = env->GetArrayLength(input);
  const jsize gammaSize = env->GetArrayLength(gamma);
  const jsize betaSize = env->GetArrayLength(beta);
  const std::string validation = ValidateLayerNormArgs(inputSize, rows, cols, gammaSize, betaSize);
  if (!validation.empty()) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, validation.c_str());
    return nullptr;
  }

  const auto* inputData = env->GetFloatArrayElements(input, nullptr);
  const auto* gammaData = env->GetFloatArrayElements(gamma, nullptr);
  const auto* betaData = env->GetFloatArrayElements(beta, nullptr);
  if (inputData == nullptr || gammaData == nullptr || betaData == nullptr) {
    if (inputData != nullptr) env->ReleaseFloatArrayElements(input, const_cast<jfloat*>(inputData), JNI_ABORT);
    if (gammaData != nullptr) env->ReleaseFloatArrayElements(gamma, const_cast<jfloat*>(gammaData), JNI_ABORT);
    if (betaData != nullptr) env->ReleaseFloatArrayElements(beta, const_cast<jfloat*>(betaData), JNI_ABORT);
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exClass, "failed to access array elements");
    return nullptr;
  }

  jfloatArray output = env->NewFloatArray(inputSize);
  if (output == nullptr) {
    env->ReleaseFloatArrayElements(input, const_cast<jfloat*>(inputData), JNI_ABORT);
    env->ReleaseFloatArrayElements(gamma, const_cast<jfloat*>(gammaData), JNI_ABORT);
    env->ReleaseFloatArrayElements(beta, const_cast<jfloat*>(betaData), JNI_ABORT);
    return nullptr;
  }

  std::vector<jfloat> outputData(static_cast<size_t>(inputSize));
  for (int r = 0; r < rows; ++r) {
    const int rowOffset = r * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) {
      mean += inputData[rowOffset + c];
    }
    mean /= static_cast<double>(cols);

    double variance = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double centered = static_cast<double>(inputData[rowOffset + c]) - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(cols);

    const double invStd = 1.0 / std::sqrt(variance + static_cast<double>(eps));
    for (int c = 0; c < cols; ++c) {
      const double normalized =
          (static_cast<double>(inputData[rowOffset + c]) - mean) * invStd;
      outputData[static_cast<size_t>(rowOffset + c)] =
          static_cast<jfloat>(normalized * static_cast<double>(gammaData[c]) +
                              static_cast<double>(betaData[c]));
    }
  }

  env->SetFloatArrayRegion(output, 0, inputSize, outputData.data());
  env->ReleaseFloatArrayElements(input, const_cast<jfloat*>(inputData), JNI_ABORT);
  env->ReleaseFloatArrayElements(gamma, const_cast<jfloat*>(gammaData), JNI_ABORT);
  env->ReleaseFloatArrayElements(beta, const_cast<jfloat*>(betaData), JNI_ABORT);
  return output;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_describeNnapiDevices(
    JNIEnv* env,
    jclass) {
  uint32_t deviceCount = 0;
  const int countResult = ANeuralNetworks_getDeviceCount(&deviceCount);
  if (countResult != ANEURALNETWORKS_NO_ERROR) {
    std::ostringstream oss;
    oss << "getDeviceCount_failed:" << countResult;
    return env->NewStringUTF(oss.str().c_str());
  }

  std::ostringstream oss;
  oss << "count=" << deviceCount;
  for (uint32_t i = 0; i < deviceCount; ++i) {
    ANeuralNetworksDevice* device = nullptr;
    const int deviceResult = ANeuralNetworks_getDevice(i, &device);
    oss << " | device[" << i << "]=";
    if (deviceResult != ANEURALNETWORKS_NO_ERROR || device == nullptr) {
      oss << "get_failed:" << deviceResult;
      continue;
    }

    const char* name = nullptr;
    const char* version = nullptr;
    int64_t featureLevel = 0;
    int32_t type = 0;
    const int nameResult = ANeuralNetworksDevice_getName(device, &name);
    const int versionResult = ANeuralNetworksDevice_getVersion(device, &version);
    const int featureResult = ANeuralNetworksDevice_getFeatureLevel(device, &featureLevel);
    const int typeResult = ANeuralNetworksDevice_getType(device, &type);

    oss << "name=" << ((nameResult == ANEURALNETWORKS_NO_ERROR && name != nullptr) ? name : "<unavailable>");
    oss << ",type=";
    switch (type) {
      case ANEURALNETWORKS_DEVICE_ACCELERATOR:
        oss << "ACCELERATOR";
        break;
      case ANEURALNETWORKS_DEVICE_CPU:
        oss << "CPU";
        break;
      case ANEURALNETWORKS_DEVICE_OTHER:
        oss << "OTHER";
        break;
      default:
        oss << "UNKNOWN(" << type << ")";
        break;
    }
    oss << ",version=" << ((versionResult == ANEURALNETWORKS_NO_ERROR && version != nullptr) ? version : "<unavailable>");
    if (featureResult == ANEURALNETWORKS_NO_ERROR) {
      oss << ",featureLevel=" << featureLevel;
    } else {
      oss << ",featureLevelErr=" << featureResult;
    }
    if (typeResult != ANEURALNETWORKS_NO_ERROR) {
      oss << ",typeErr=" << typeResult;
    }
  }

  return env->NewStringUTF(oss.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_probeQnnCase(
    JNIEnv* env,
    jclass,
    jstring bundleDirJ) {
  const char* bundleDirChars = env->GetStringUTFChars(bundleDirJ, nullptr);
  if (bundleDirChars == nullptr) {
    return env->NewStringUTF("failed_to_read_bundle_dir");
  }
  const std::string bundleDir(bundleDirChars);
  env->ReleaseStringUTFChars(bundleDirJ, bundleDirChars);

  const auto paths = DiscoverCasePaths(bundleDir);

  std::ostringstream oss;
  oss << "bundle=" << bundleDir;

  oss << " | libc++_shared.so=" << (FileExists(paths.libcxx) ? "OK" : "MISSING");
  oss << " | libQnnSystem.so=" << (FileExists(paths.system) ? "OK" : "MISSING");
  oss << " | libQnnGpuNetRunExtensions.so=" << (FileExists(paths.gpuExt) ? "OK" : "MISSING");
  oss << " | libQnnGpu.so=" << (FileExists(paths.backend) ? "OK" : "MISSING");
  oss << " | model=" << (!paths.model.empty() ? paths.model.filename().string() : "<missing>");
  oss << " | input=" << (!paths.input.empty() ? paths.input.filename().string() : "<missing>");
  oss << " | expected=" << (!paths.expected.empty() ? paths.expected.filename().string() : "<missing>");
  oss << " | probeOutput="
      << (!paths.probeOutput.empty() ? paths.probeOutput.filename().string() : "<missing>");
  if (!FileExists(paths.libcxx) || !FileExists(paths.system) || !FileExists(paths.gpuExt) ||
      !FileExists(paths.backend) || paths.model.empty() || paths.input.empty() || paths.expected.empty()) {
    return env->NewStringUTF(oss.str().c_str());
  }

  dlerror();
  void* libcxxHandle = dlopen(paths.libcxx.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (libcxxHandle == nullptr) {
    oss << " | libcxx_dlopen_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | libcxx=loaded";

  void* systemHandle = dlopen(paths.system.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (systemHandle == nullptr) {
    oss << " | system_dlopen_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | system=loaded";

  void* extHandle = dlopen(paths.gpuExt.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (extHandle == nullptr) {
    oss << " | gpu_ext_dlopen_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | gpu_ext=loaded";

  void* backendHandleLib = dlopen(paths.backend.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (backendHandleLib == nullptr) {
    oss << " | backend_dlopen_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | backend=loaded";

  void* modelHandleLib = dlopen(paths.model.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (modelHandleLib == nullptr) {
    oss << " | model_dlopen_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | model=loaded";

  const auto getSystemProviders =
      ResolveSymbol<QnnSystemInterfaceGetProvidersFn_t>(systemHandle, "QnnSystemInterface_getProviders");
  if (getSystemProviders == nullptr) {
    oss << " | system_providers_symbol_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }

  const QnnSystemInterface_t** systemProviders = nullptr;
  uint32_t systemProviderCount = 0;
  const auto systemProvidersStatus = getSystemProviders(&systemProviders, &systemProviderCount);
  oss << " | systemProviders=" << systemProviderCount << "@" << QnnErrorString(systemProvidersStatus);
  const auto* systemProvider = SelectQnnSystemInterface(systemProviders, systemProviderCount);
  if (systemProvider == nullptr) {
    oss << " | system_provider_select=FAIL";
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | system_provider_select=OK";

  const auto getProviders =
      ResolveSymbol<QnnInterfaceGetProvidersFn_t>(backendHandleLib, "QnnInterface_getProviders");
  if (getProviders == nullptr) {
    oss << " | providers_symbol_fail=" << DlErrorString();
    return env->NewStringUTF(oss.str().c_str());
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t providerCount = 0;
  const auto providersStatus = getProviders(&providers, &providerCount);
  oss << " | providers=" << providerCount << "@" << QnnErrorString(providersStatus);
  const auto* provider = SelectQnnInterface(providers, providerCount);
  if (provider == nullptr) {
    oss << " | provider_select=FAIL";
    return env->NewStringUTF(oss.str().c_str());
  }
  oss << " | provider_select=OK";

  const auto composeGraphs = ResolveSymbol<ComposeGraphsFn_t>(modelHandleLib, "QnnModel_composeGraphs");
  const auto freeGraphsInfo =
      ResolveSymbol<FreeGraphsInfoFn_t>(modelHandleLib, "QnnModel_freeGraphsInfo");
  oss << " | model_symbols=" << ((composeGraphs != nullptr && freeGraphsInfo != nullptr) ? "OK" : "FAIL");
  if (composeGraphs == nullptr || freeGraphsInfo == nullptr) {
    return env->NewStringUTF(oss.str().c_str());
  }

  if (provider->QNN_INTERFACE_VER_NAME.backendCreate == nullptr ||
      provider->QNN_INTERFACE_VER_NAME.backendFree == nullptr) {
    oss << " | backend_api=missing";
    return env->NewStringUTF(oss.str().c_str());
  }

  Qnn_BackendHandle_t backendHandle = nullptr;
  const auto backendCreateStatus =
      provider->QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &backendHandle);
  oss << " | backendCreate=" << QnnErrorString(backendCreateStatus);
  if (backendCreateStatus != QNN_SUCCESS || backendHandle == nullptr) {
    return env->NewStringUTF(oss.str().c_str());
  }

  Qnn_DeviceHandle_t deviceHandle = nullptr;
  if (provider->QNN_INTERFACE_VER_NAME.deviceCreate != nullptr) {
    const auto deviceCreateStatus =
        provider->QNN_INTERFACE_VER_NAME.deviceCreate(nullptr, nullptr, &deviceHandle);
    oss << " | deviceCreate=" << QnnErrorString(deviceCreateStatus);
    if (deviceCreateStatus != QNN_SUCCESS &&
        deviceCreateStatus != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
      return env->NewStringUTF(oss.str().c_str());
    }
  } else {
    oss << " | deviceCreate=missing";
  }

  Qnn_ContextHandle_t contextHandle = nullptr;
  if (provider->QNN_INTERFACE_VER_NAME.contextCreate == nullptr) {
    oss << " | contextCreate=missing";
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  const auto contextCreateStatus =
      provider->QNN_INTERFACE_VER_NAME.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle);
  oss << " | contextCreate=" << QnnErrorString(contextCreateStatus);
  if (contextCreateStatus != QNN_SUCCESS || contextHandle == nullptr) {
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  qnn_wrapper_api::GraphInfo_t** graphsInfo = nullptr;
  uint32_t graphCount = 0;
  const auto composeStatus = composeGraphs(backendHandle,
                                           provider->QNN_INTERFACE_VER_NAME,
                                           contextHandle,
                                           nullptr,
                                           0,
                                           &graphsInfo,
                                           &graphCount,
                                           false,
                                           nullptr,
                                           QNN_LOG_LEVEL_ERROR);
  oss << " | composeGraphs=" << ModelErrorString(composeStatus) << " graphs=" << graphCount;
  if (composeStatus != qnn_wrapper_api::MODEL_NO_ERROR || graphsInfo == nullptr || graphCount == 0) {
    provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  if (provider->QNN_INTERFACE_VER_NAME.graphFinalize == nullptr) {
    oss << " | graphFinalize=missing";
  } else {
    bool finalizeOk = true;
    for (uint32_t i = 0; i < graphCount; ++i) {
      auto* graphInfo = graphsInfo[i];
      if (graphInfo == nullptr) {
        finalizeOk = false;
        oss << " | graph[" << i << "]=null";
        break;
      }
      const auto finalizeStatus =
          provider->QNN_INTERFACE_VER_NAME.graphFinalize(graphInfo->graph, nullptr, nullptr);
      oss << " | graphFinalize[" << i << "]=" << QnnErrorString(finalizeStatus)
          << "(" << (graphInfo->graphName == nullptr ? "<unnamed>" : graphInfo->graphName) << ")";
      if (finalizeStatus != QNN_SUCCESS) {
        finalizeOk = false;
        break;
      }
    }
    if (!finalizeOk) {
      freeGraphsInfo(&graphsInfo, graphCount);
      provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
      if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
        provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
      }
      provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
      return env->NewStringUTF(oss.str().c_str());
    }
  }

  auto* graphInfo = graphsInfo[0];
  if (graphInfo == nullptr || graphInfo->numInputTensors != 1 || graphInfo->numOutputTensors != 1) {
    oss << " | graphShape=unexpected";
    freeGraphsInfo(&graphsInfo, graphCount);
    provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  std::vector<uint8_t> inputBytes = ReadBinaryFile(paths.input);
  std::vector<uint8_t> expectedBytes = ReadBinaryFile(paths.expected);
  std::vector<uint8_t> probeOutputBytes =
      paths.probeOutput.empty() ? std::vector<uint8_t>{} : ReadBinaryFile(paths.probeOutput);
  oss << " | inputBytes=" << inputBytes.size() << " expectedBytes=" << expectedBytes.size();

  Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
  std::vector<uint8_t> inputBuffer;
  std::vector<uint8_t> outputBuffer;
  const bool inputOk = SetupTensorFromTemplate(
      graphInfo->inputTensors[0], inputTensor, inputBuffer, inputBytes.data(), inputBytes.size());
  const bool outputOk =
      SetupTensorFromTemplate(graphInfo->outputTensors[0], outputTensor, outputBuffer, nullptr, 0);
  oss << " | setupInput=" << (inputOk ? "OK" : "FAIL") << " setupOutput=" << (outputOk ? "OK" : "FAIL");
  if (!inputOk || !outputOk) {
    qnn_wrapper_api::freeQnnTensor(inputTensor);
    qnn_wrapper_api::freeQnnTensor(outputTensor);
    freeGraphsInfo(&graphsInfo, graphCount);
    provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  if (provider->QNN_INTERFACE_VER_NAME.graphExecute == nullptr) {
    oss << " | graphExecute=missing";
    qnn_wrapper_api::freeQnnTensor(inputTensor);
    qnn_wrapper_api::freeQnnTensor(outputTensor);
    freeGraphsInfo(&graphsInfo, graphCount);
    provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
    if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
      provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    }
    provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
    return env->NewStringUTF(oss.str().c_str());
  }

  constexpr int kWarmupRuns = 3;
  constexpr int kMeasuredRuns = 20;
  std::vector<double> runTimesMs;
  runTimesMs.reserve(kMeasuredRuns);

  Qnn_ErrorHandle_t executeStatus = QNN_SUCCESS;
  for (int i = 0; i < kWarmupRuns; ++i) {
    executeStatus = provider->QNN_INTERFACE_VER_NAME.graphExecute(
        graphInfo->graph, &inputTensor, 1, &outputTensor, 1, nullptr, nullptr);
    if (executeStatus != QNN_SUCCESS) break;
  }
  oss << " | warmup=" << kWarmupRuns << " warmupStatus=" << QnnErrorString(executeStatus);

  if (executeStatus == QNN_SUCCESS) {
    for (int i = 0; i < kMeasuredRuns; ++i) {
      const auto start = std::chrono::steady_clock::now();
      executeStatus = provider->QNN_INTERFACE_VER_NAME.graphExecute(
          graphInfo->graph, &inputTensor, 1, &outputTensor, 1, nullptr, nullptr);
      const auto end = std::chrono::steady_clock::now();
      if (executeStatus != QNN_SUCCESS) break;
      runTimesMs.push_back(
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count());
    }
  }

  oss << " | graphExecute=" << QnnErrorString(executeStatus);
  if (executeStatus == QNN_SUCCESS && !runTimesMs.empty()) {
    const double avgMs =
        std::accumulate(runTimesMs.begin(), runTimesMs.end(), 0.0) / static_cast<double>(runTimesMs.size());
    auto sortedTimes = runTimesMs;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    const double medianMs = sortedTimes[sortedTimes.size() / 2];
    oss << " avgMs=" << FormatMs(avgMs) << " medianMs=" << FormatMs(medianMs)
        << " runs=" << runTimesMs.size();
  }
  if (executeStatus == QNN_SUCCESS) {
    const auto actualSize = TensorByteSize(outputTensor);
    std::vector<uint8_t> actualBytes(actualSize);
    std::memcpy(actualBytes.data(), outputBuffer.data(), actualSize);
    oss << " | compare=" << CompareFloatBuffers(expectedBytes, actualBytes);
    if (!probeOutputBytes.empty()) {
      oss << " | compareProbeOutput=" << CompareFloatBuffers(probeOutputBytes, actualBytes);
    }
  }

  qnn_wrapper_api::freeQnnTensor(inputTensor);
  qnn_wrapper_api::freeQnnTensor(outputTensor);

  const auto freeGraphsStatus = freeGraphsInfo(&graphsInfo, graphCount);
  oss << " | freeGraphsInfo=" << ModelErrorString(freeGraphsStatus);

  const auto contextFreeStatus = provider->QNN_INTERFACE_VER_NAME.contextFree(contextHandle, nullptr);
  oss << " | contextFree=" << QnnErrorString(contextFreeStatus);

  if (deviceHandle != nullptr && provider->QNN_INTERFACE_VER_NAME.deviceFree != nullptr) {
    const auto deviceFreeStatus = provider->QNN_INTERFACE_VER_NAME.deviceFree(deviceHandle);
    oss << " | deviceFree=" << QnnErrorString(deviceFreeStatus);
  }

  if (provider->QNN_INTERFACE_VER_NAME.backendGetBuildId != nullptr) {
    const char* buildId = nullptr;
    if (provider->QNN_INTERFACE_VER_NAME.backendGetBuildId(&buildId) == QNN_SUCCESS && buildId != nullptr) {
      oss << " | buildId=" << buildId;
    }
  }

  const auto backendFreeStatus = provider->QNN_INTERFACE_VER_NAME.backendFree(backendHandle);
  oss << " | backendFree=" << QnnErrorString(backendFreeStatus);
  return env->NewStringUTF(oss.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_initCachedQnnNodeEncoderCase(
    JNIEnv* env,
    jclass,
    jstring bundleDirJ) {
  const char* bundleDirChars = env->GetStringUTFChars(bundleDirJ, nullptr);
  if (bundleDirChars == nullptr) {
    return env->NewStringUTF("failed_to_read_bundle_dir");
  }
  const std::string bundleDir(bundleDirChars);
  env->ReleaseStringUTFChars(bundleDirJ, bundleDirChars);
  const auto result = InitializeCachedNodeEncoderCase(bundleDir, g_cachedNodeEncoderCase);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnNodeEncoderCase(
    JNIEnv* env,
    jclass,
    jint warmupRuns,
    jint measuredRuns) {
  std::ostringstream oss;
  if (!g_cachedNodeEncoderCase.initialized) {
    oss << "not_initialized";
    return env->NewStringUTF(oss.str().c_str());
  }
  if (warmupRuns < 0 || measuredRuns <= 0) {
    oss << "invalid_args warmup=" << warmupRuns << " runs=" << measuredRuns;
    return env->NewStringUTF(oss.str().c_str());
  }

  Qnn_ErrorHandle_t executeStatus = QNN_SUCCESS;
  for (int i = 0; i < warmupRuns; ++i) {
    executeStatus = g_cachedNodeEncoderCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
        g_cachedNodeEncoderCase.graphHandle,
        &g_cachedNodeEncoderCase.inputTensor,
        1,
        &g_cachedNodeEncoderCase.outputTensor,
        1,
        nullptr,
        nullptr);
    if (executeStatus != QNN_SUCCESS) break;
  }
  oss << "graph=" << g_cachedNodeEncoderCase.graphName
      << " | warmup=" << warmupRuns
      << " warmupStatus=" << QnnErrorString(executeStatus);
  if (executeStatus != QNN_SUCCESS) {
    return env->NewStringUTF(oss.str().c_str());
  }

  std::vector<double> runTimesMs;
  runTimesMs.reserve(static_cast<size_t>(measuredRuns));
  for (int i = 0; i < measuredRuns; ++i) {
    const auto start = std::chrono::steady_clock::now();
    executeStatus = g_cachedNodeEncoderCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
        g_cachedNodeEncoderCase.graphHandle,
        &g_cachedNodeEncoderCase.inputTensor,
        1,
        &g_cachedNodeEncoderCase.outputTensor,
        1,
        nullptr,
        nullptr);
    const auto end = std::chrono::steady_clock::now();
    if (executeStatus != QNN_SUCCESS) break;
    runTimesMs.push_back(
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count());
  }
  oss << " | graphExecute=" << QnnErrorString(executeStatus);
  if (executeStatus != QNN_SUCCESS) {
    return env->NewStringUTF(oss.str().c_str());
  }

  const double avgMs =
      std::accumulate(runTimesMs.begin(), runTimesMs.end(), 0.0) / static_cast<double>(runTimesMs.size());
  auto sortedTimes = runTimesMs;
  std::sort(sortedTimes.begin(), sortedTimes.end());
  const double medianMs = sortedTimes[sortedTimes.size() / 2];
  const auto actualSize = TensorByteSize(g_cachedNodeEncoderCase.outputTensor);
  std::vector<uint8_t> actualBytes(actualSize);
  std::memcpy(actualBytes.data(), g_cachedNodeEncoderCase.outputBuffer.data(), actualSize);
  oss << " avgMs=" << FormatMs(avgMs)
      << " medianMs=" << FormatMs(medianMs)
      << " runs=" << runTimesMs.size()
      << " | compare=" << CompareFloatBuffers(g_cachedNodeEncoderCase.expectedBytes, actualBytes);
  if (!g_cachedNodeEncoderCase.buildId.empty()) {
    oss << " | buildId=" << g_cachedNodeEncoderCase.buildId;
  }
  return env->NewStringUTF(oss.str().c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnNodeEncoderCaseInput(
    JNIEnv* env,
    jclass,
    jfloatArray inputJ) {
  if (!g_cachedNodeEncoderCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "cached qnn node encoder case is not initialized");
    return nullptr;
  }
  if (inputJ == nullptr) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "input is null");
    return nullptr;
  }
  const jsize inputSize = env->GetArrayLength(inputJ);
  const size_t expectedFloats = g_cachedNodeEncoderCase.inputBuffer.size() / sizeof(float);
  if (static_cast<size_t>(inputSize) != expectedFloats) {
    std::ostringstream oss;
    oss << "input size mismatch expected=" << expectedFloats << " actual=" << inputSize;
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, oss.str().c_str());
    return nullptr;
  }

  env->GetFloatArrayRegion(
      inputJ, 0, inputSize, reinterpret_cast<jfloat*>(g_cachedNodeEncoderCase.inputBuffer.data()));
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  const auto executeStatus = g_cachedNodeEncoderCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      g_cachedNodeEncoderCase.graphHandle,
      &g_cachedNodeEncoderCase.inputTensor,
      1,
      &g_cachedNodeEncoderCase.outputTensor,
      1,
      nullptr,
      nullptr);
  if (executeStatus != QNN_SUCCESS) {
    std::ostringstream oss;
    oss << "graphExecute failed: " << executeStatus;
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exClass, oss.str().c_str());
    return nullptr;
  }

  const jsize outputFloats = static_cast<jsize>(g_cachedNodeEncoderCase.outputBuffer.size() / sizeof(float));
  jfloatArray outputJ = env->NewFloatArray(outputFloats);
  if (outputJ == nullptr) return nullptr;
  env->SetFloatArrayRegion(
      outputJ, 0, outputFloats, reinterpret_cast<const jfloat*>(g_cachedNodeEncoderCase.outputBuffer.data()));
  return outputJ;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_releaseCachedQnnNodeEncoderCase(
    JNIEnv* env,
    jclass) {
  const bool wasInitialized = g_cachedNodeEncoderCase.initialized;
  ReleaseCachedCase(g_cachedNodeEncoderCase);
  const std::string result = wasInitialized ? "released" : "already_released";
  return env->NewStringUTF(result.c_str());
}
