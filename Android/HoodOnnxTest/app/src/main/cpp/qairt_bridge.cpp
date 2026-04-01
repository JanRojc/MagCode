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
#include "qnn_case_runtime.h"

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
  std::filesystem::path htpPrepare;
  std::filesystem::path htpStub;
  std::filesystem::path htpSkel;
  std::filesystem::path backend;
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path expected;
  std::filesystem::path probeOutput;
  std::filesystem::path layernormGamma;
  std::filesystem::path layernormBeta;
  std::filesystem::path manifest;
  std::string backendKind;
};

struct CachedQnnCaseState {
  bool initialized = false;
  std::string bundleDir;
  std::string buildId;
  std::string graphName;
  void* libcxxHandle = nullptr;
  void* systemHandle = nullptr;
  void* gpuExtHandle = nullptr;
  void* htpPrepareHandle = nullptr;
  void* htpStubHandle = nullptr;
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
  std::vector<float> layernormGamma;
  std::vector<float> layernormBeta;
  float layernormEps = 1.0e-5f;
  int layernormRows = 0;
  int layernormCols = 0;
};

CachedQnnCaseState g_cachedNodeEncoderCase;
CachedQnnCaseState g_cachedBlock000EdgeMeshCase;
CachedQnnCaseState g_cachedBlock001EdgeMeshCase;
std::vector<float> g_cachedMeshEdgeState;
int g_cachedMeshEdgeStateLatent = 0;
int g_cachedMeshEdgeStateEdgeCount = 0;

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
  const auto htpBackend = bundle / "libQnnHtp.so";
  const bool useHtp = FileExists(htpBackend);
  return {
      bundle,
      bundle / "libc++_shared.so",
      bundle / "libQnnSystem.so",
      bundle / "libQnnGpuNetRunExtensions.so",
      bundle / "libQnnHtpPrepare.so",
      FindFirstMatchingFile(bundle,
                            {"libQnnHtpV81Stub.so",
                             "libQnnHtpV79Stub.so",
                             "libQnnHtpV75Stub.so",
                             "libQnnHtpV73Stub.so",
                             "libQnnHtpV69Stub.so",
                             "libQnnHtpV68Stub.so"}),
      FindFirstMatchingFile(bundle,
                            {"libQnnHtpV81Skel.so",
                             "libQnnHtpV79Skel.so",
                             "libQnnHtpV75Skel.so",
                             "libQnnHtpV73Skel.so",
                             "libQnnHtpV69Skel.so",
                             "libQnnHtpV68Skel.so"}),
      useHtp ? htpBackend : (bundle / "libQnnGpu.so"),
      FindProbeModelLib(bundle),
      FindFirstMatchingFile(bundle, {"node_encoder_input.raw", "input.raw"}),
      FindFirstMatchingFile(bundle, {"expected_node_encoder_output.raw", "node_encoder_probe_expected.raw", "expected_output.raw"}),
      FindFirstMatchingFile(bundle, {"output_probe/output/Result_0/output.raw", "hybrid_output.raw"}),
      FindFirstMatchingFile(bundle, {"layernorm_gamma.bin"}),
      FindFirstMatchingFile(bundle, {"layernorm_beta.bin"}),
      FindFirstMatchingFile(bundle, {"manifest.json", "case_manifest.json"}),
      useHtp ? "htp" : "gpu",
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

std::vector<float> ReadFloatFile(const std::filesystem::path& path) {
  const auto bytes = ReadBinaryFile(path);
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) return {};
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
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
  DlCloseQuiet(state.htpStubHandle);
  DlCloseQuiet(state.htpPrepareHandle);
  DlCloseQuiet(state.gpuExtHandle);
  DlCloseQuiet(state.systemHandle);
  DlCloseQuiet(state.libcxxHandle);
  state = CachedQnnCaseState{};
}

void ResetCachedMeshEdgeState() {
  g_cachedMeshEdgeState.clear();
  g_cachedMeshEdgeStateLatent = 0;
  g_cachedMeshEdgeStateEdgeCount = 0;
}

bool ValidateMeshEdgeState(const CachedQnnCaseState& state, int latent, int edgeCount) {
  return state.initialized &&
      latent > 0 &&
      edgeCount > 0 &&
      !g_cachedMeshEdgeState.empty() &&
      g_cachedMeshEdgeStateLatent == latent &&
      g_cachedMeshEdgeStateEdgeCount == edgeCount &&
      static_cast<size_t>(edgeCount) * static_cast<size_t>(latent) == g_cachedMeshEdgeState.size() &&
      state.inputBuffer.size() == static_cast<size_t>(edgeCount) * static_cast<size_t>(latent) * 3 * sizeof(float) &&
      state.outputBuffer.size() == static_cast<size_t>(edgeCount) * static_cast<size_t>(latent) * sizeof(float);
}

bool ExecuteMeshEdgeCaseToAggregated(
    const CachedQnnCaseState& state,
    const std::vector<float>& clothNodes,
    const std::vector<jint>& edgeIndex,
    int latent,
    std::vector<float>& aggregated) {
  const int edgeCount = static_cast<int>(edgeIndex.size() / 2);
  const int targetNodeCount = static_cast<int>(clothNodes.size() / latent);
  if (!ValidateMeshEdgeState(state, latent, edgeCount)) return false;

  auto* packed = reinterpret_cast<float*>(const_cast<std::vector<uint8_t>&>(state.inputBuffer).data());
  for (int edge = 0; edge < edgeCount; ++edge) {
    const int src = edgeIndex[edge];
    const int tgt = edgeIndex[edge + edgeCount];
    const int rowBase = edge * latent * 3;
    const int tgtBase = tgt * latent;
    const int srcBase = src * latent;
    const int edgeBase = edge * latent;
    std::memcpy(packed + rowBase, clothNodes.data() + tgtBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent, clothNodes.data() + srcBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent * 2, g_cachedMeshEdgeState.data() + edgeBase, static_cast<size_t>(latent) * sizeof(float));
  }

  const auto executeStatus = state.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      state.graphHandle,
      const_cast<Qnn_Tensor_t*>(&state.inputTensor),
      1,
      const_cast<Qnn_Tensor_t*>(&state.outputTensor),
      1,
      nullptr,
      nullptr);
  if (executeStatus != QNN_SUCCESS) return false;

  std::vector<float> post(static_cast<size_t>(edgeCount * latent));
  const float* qnnOutput = reinterpret_cast<const float*>(state.outputBuffer.data());
  if (!state.layernormGamma.empty() && !state.layernormBeta.empty()) {
    hood::qnn::ApplyLayerNormRows(
        qnnOutput,
        edgeCount,
        latent,
        state.layernormGamma.data(),
        state.layernormBeta.data(),
        state.layernormEps,
        post.data());
  } else {
    std::memcpy(post.data(), qnnOutput, static_cast<size_t>(edgeCount * latent) * sizeof(float));
  }

  for (size_t i = 0; i < g_cachedMeshEdgeState.size(); ++i) {
    g_cachedMeshEdgeState[i] += post[i];
  }

  aggregated.assign(static_cast<size_t>(targetNodeCount * latent), 0.0f);
  for (int edge = 0; edge < edgeCount; ++edge) {
    const int tgt = edgeIndex[edge + edgeCount];
    const int inBase = edge * latent;
    const int outBase = tgt * latent;
    for (int f = 0; f < latent; ++f) {
      aggregated[outBase + f] += post[inBase + f];
    }
  }
  return true;
}

std::string InitializeCachedCase(const std::string& bundleDir, CachedQnnCaseState& state) {
  ReleaseCachedCase(state);
  const auto paths = DiscoverCasePaths(bundleDir);
  std::ostringstream oss;
  oss << "bundle=" << bundleDir;
  const bool needsGpuExt = paths.backendKind == "gpu";
  const bool needsHtp = paths.backendKind == "htp";
  if (!FileExists(paths.libcxx) || !FileExists(paths.system) ||
      (needsGpuExt && !FileExists(paths.gpuExt)) ||
      (needsHtp && (!FileExists(paths.htpPrepare) || paths.htpStub.empty())) ||
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
  if (needsGpuExt) {
    state.gpuExtHandle = dlopen(paths.gpuExt.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (state.gpuExtHandle == nullptr) {
      oss << " | gpu_ext_dlopen_fail=" << DlErrorString();
      ReleaseCachedCase(state);
      return oss.str();
    }
  }
  if (needsHtp) {
    const std::string adspPath = bundleDir +
        ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
    state.htpPrepareHandle = dlopen(paths.htpPrepare.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (state.htpPrepareHandle == nullptr) {
      oss << " | htp_prepare_dlopen_fail=" << DlErrorString();
      ReleaseCachedCase(state);
      return oss.str();
    }
    state.htpStubHandle = dlopen(paths.htpStub.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (state.htpStubHandle == nullptr) {
      oss << " | htp_stub_dlopen_fail=" << DlErrorString();
      ReleaseCachedCase(state);
      return oss.str();
    }
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

  if (!paths.layernormGamma.empty() && !paths.layernormBeta.empty() && !paths.manifest.empty()) {
    state.layernormGamma = ReadFloatFile(paths.layernormGamma);
    state.layernormBeta = ReadFloatFile(paths.layernormBeta);
    std::ifstream manifest(paths.manifest);
    if (manifest && !state.layernormGamma.empty() &&
        state.layernormGamma.size() == state.layernormBeta.size()) {
      std::stringstream buffer;
      buffer << manifest.rdbuf();
      const std::string text = buffer.str();
      auto findNumber = [&](const char* key, double fallback) -> double {
        const std::string needle = std::string("\"") + key + "\"";
        const auto pos = text.find(needle);
        if (pos == std::string::npos) return fallback;
        const auto colon = text.find(':', pos + needle.size());
        if (colon == std::string::npos) return fallback;
        const auto start = text.find_first_of("-0123456789", colon + 1);
        if (start == std::string::npos) return fallback;
        char* endPtr = nullptr;
        const double value = std::strtod(text.c_str() + start, &endPtr);
        return endPtr == text.c_str() + start ? fallback : value;
      };
      state.layernormRows = static_cast<int>(findNumber("rows", 0.0));
      state.layernormCols = static_cast<int>(
          findNumber("layernorm_dim", static_cast<double>(state.layernormGamma.size())));
      state.layernormEps = static_cast<float>(findNumber("layernorm_eps", 1.0e-5));
    }
  }
  if (!state.layernormGamma.empty() &&
      state.layernormGamma.size() == state.layernormBeta.size() &&
      state.layernormCols > 0) {
    oss << " | layernorm=OK gamma=" << state.layernormGamma.size()
        << " cols=" << state.layernormCols
        << " eps=" << state.layernormEps;
  } else if (!paths.layernormGamma.empty() || !paths.layernormBeta.empty() || !paths.manifest.empty()) {
    oss << " | layernorm=BAD gamma=" << state.layernormGamma.size()
        << " beta=" << state.layernormBeta.size()
        << " cols=" << state.layernormCols
        << " eps=" << state.layernormEps;
  } else {
    oss << " | layernorm=MISSING";
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
  hood::qnn::ApplyLayerNormRows(
      inputData,
      rows,
      cols,
      gammaData,
      betaData,
      eps,
      outputData.data());

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
  oss << " | backendKind=" << paths.backendKind;

  oss << " | libc++_shared.so=" << (FileExists(paths.libcxx) ? "OK" : "MISSING");
  oss << " | libQnnSystem.so=" << (FileExists(paths.system) ? "OK" : "MISSING");
  if (paths.backendKind == "gpu") {
    oss << " | libQnnGpuNetRunExtensions.so=" << (FileExists(paths.gpuExt) ? "OK" : "MISSING");
    oss << " | libQnnGpu.so=" << (FileExists(paths.backend) ? "OK" : "MISSING");
  } else {
    oss << " | libQnnHtpPrepare.so=" << (FileExists(paths.htpPrepare) ? "OK" : "MISSING");
    oss << " | htpStub=" << (!paths.htpStub.empty() ? paths.htpStub.filename().string() : "<missing>");
    oss << " | htpSkel=" << (!paths.htpSkel.empty() ? paths.htpSkel.filename().string() : "<missing>");
    oss << " | libQnnHtp.so=" << (FileExists(paths.backend) ? "OK" : "MISSING");
  }
  oss << " | model=" << (!paths.model.empty() ? paths.model.filename().string() : "<missing>");
  oss << " | input=" << (!paths.input.empty() ? paths.input.filename().string() : "<missing>");
  oss << " | expected=" << (!paths.expected.empty() ? paths.expected.filename().string() : "<missing>");
  oss << " | probeOutput="
      << (!paths.probeOutput.empty() ? paths.probeOutput.filename().string() : "<missing>");
  const bool needsGpuExt = paths.backendKind == "gpu";
  const bool needsHtp = paths.backendKind == "htp";
  if (!FileExists(paths.libcxx) || !FileExists(paths.system) ||
      (needsGpuExt && !FileExists(paths.gpuExt)) ||
      (needsHtp && (!FileExists(paths.htpPrepare) || paths.htpStub.empty())) ||
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

  void* extHandle = nullptr;
  void* htpPrepareHandle = nullptr;
  void* htpStubHandle = nullptr;
  if (needsGpuExt) {
    extHandle = dlopen(paths.gpuExt.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (extHandle == nullptr) {
      oss << " | gpu_ext_dlopen_fail=" << DlErrorString();
      return env->NewStringUTF(oss.str().c_str());
    }
    oss << " | gpu_ext=loaded";
  } else {
    const std::string adspPath =
        bundleDir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
    htpPrepareHandle = dlopen(paths.htpPrepare.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (htpPrepareHandle == nullptr) {
      oss << " | htp_prepare_dlopen_fail=" << DlErrorString();
      return env->NewStringUTF(oss.str().c_str());
    }
    htpStubHandle = dlopen(paths.htpStub.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (htpStubHandle == nullptr) {
      oss << " | htp_stub_dlopen_fail=" << DlErrorString();
      return env->NewStringUTF(oss.str().c_str());
    }
    oss << " | htp_prepare=loaded | htp_stub=loaded";
  }

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
  const auto result = InitializeCachedCase(bundleDir, g_cachedNodeEncoderCase);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_initCachedQnnBlock000EdgeMeshMlpBodyCase(
    JNIEnv* env,
    jclass,
    jstring bundleDirJ) {
  const char* bundleDirChars = env->GetStringUTFChars(bundleDirJ, nullptr);
  if (bundleDirChars == nullptr) {
    return env->NewStringUTF("failed_to_read_bundle_dir");
  }
  const std::string bundleDir(bundleDirChars);
  env->ReleaseStringUTFChars(bundleDirJ, bundleDirChars);
  const auto result = InitializeCachedCase(bundleDir, g_cachedBlock000EdgeMeshCase);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_initCachedQnnBlock001EdgeMeshMlpBodyCase(
    JNIEnv* env,
    jclass,
    jstring bundleDirJ) {
  const char* bundleDirChars = env->GetStringUTFChars(bundleDirJ, nullptr);
  if (bundleDirChars == nullptr) {
    return env->NewStringUTF("failed_to_read_bundle_dir");
  }
  const std::string bundleDir(bundleDirChars);
  env->ReleaseStringUTFChars(bundleDirJ, bundleDirChars);
  const auto result = InitializeCachedCase(bundleDir, g_cachedBlock001EdgeMeshCase);
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

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnBlock000EdgeMeshMlpBodyCaseInput(
    JNIEnv* env,
    jclass,
    jfloatArray inputJ) {
  if (!g_cachedBlock000EdgeMeshCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "cached qnn block_0_0 edge mesh case is not initialized");
    return nullptr;
  }
  if (inputJ == nullptr) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "input is null");
    return nullptr;
  }
  const jsize inputSize = env->GetArrayLength(inputJ);
  const size_t expectedFloats = g_cachedBlock000EdgeMeshCase.inputBuffer.size() / sizeof(float);
  if (static_cast<size_t>(inputSize) != expectedFloats) {
    std::ostringstream oss;
    oss << "input size mismatch expected=" << expectedFloats << " actual=" << inputSize;
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, oss.str().c_str());
    return nullptr;
  }

  env->GetFloatArrayRegion(
      inputJ, 0, inputSize, reinterpret_cast<jfloat*>(g_cachedBlock000EdgeMeshCase.inputBuffer.data()));
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  const auto executeStatus = g_cachedBlock000EdgeMeshCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      g_cachedBlock000EdgeMeshCase.graphHandle,
      &g_cachedBlock000EdgeMeshCase.inputTensor,
      1,
      &g_cachedBlock000EdgeMeshCase.outputTensor,
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

  const jsize outputFloats = static_cast<jsize>(g_cachedBlock000EdgeMeshCase.outputBuffer.size() / sizeof(float));
  jfloatArray outputJ = env->NewFloatArray(outputFloats);
  if (outputJ == nullptr) return nullptr;
  env->SetFloatArrayRegion(
      outputJ, 0, outputFloats, reinterpret_cast<const jfloat*>(g_cachedBlock000EdgeMeshCase.outputBuffer.data()));
  return outputJ;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnBlock000EdgeMeshMlpBodyCasePacked(
    JNIEnv* env,
    jclass,
    jfloatArray tgtNodesJ,
    jfloatArray srcNodesJ,
    jfloatArray edgeFeatJ,
    jintArray edgeIndexJ,
    jint latent) {
  if (!g_cachedBlock000EdgeMeshCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "cached qnn block_0_0 edge mesh case is not initialized");
    return nullptr;
  }
  if (tgtNodesJ == nullptr || srcNodesJ == nullptr || edgeFeatJ == nullptr || edgeIndexJ == nullptr) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "null input");
    return nullptr;
  }
  if (latent <= 0) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "latent must be > 0");
    return nullptr;
  }

  const jsize tgtSize = env->GetArrayLength(tgtNodesJ);
  const jsize srcSize = env->GetArrayLength(srcNodesJ);
  const jsize edgeFeatSize = env->GetArrayLength(edgeFeatJ);
  const jsize edgeIndexSize = env->GetArrayLength(edgeIndexJ);
  const int edgeCount = edgeIndexSize / 2;
  const size_t expectedInputFloats = static_cast<size_t>(g_cachedBlock000EdgeMeshCase.inputBuffer.size() / sizeof(float));
  const size_t expectedOutputFloats = static_cast<size_t>(g_cachedBlock000EdgeMeshCase.outputBuffer.size() / sizeof(float));
  if ((edgeIndexSize % 2) != 0 ||
      static_cast<size_t>(edgeCount) * static_cast<size_t>(latent) * 3 != expectedInputFloats ||
      static_cast<size_t>(edgeCount) * static_cast<size_t>(latent) != expectedOutputFloats ||
      (tgtSize % latent) != 0 || (srcSize % latent) != 0 || edgeFeatSize != edgeCount * latent) {
    std::ostringstream oss;
    oss << "shape mismatch edgeCount=" << edgeCount
        << " latent=" << latent
        << " expectedInputFloats=" << expectedInputFloats
        << " expectedOutputFloats=" << expectedOutputFloats
        << " tgtSize=" << tgtSize
        << " srcSize=" << srcSize
        << " edgeFeatSize=" << edgeFeatSize
        << " edgeIndexSize=" << edgeIndexSize;
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, oss.str().c_str());
    return nullptr;
  }

  std::vector<float> tgtNodes(static_cast<size_t>(tgtSize));
  std::vector<float> srcNodes(static_cast<size_t>(srcSize));
  std::vector<float> edgeFeat(static_cast<size_t>(edgeFeatSize));
  std::vector<jint> edgeIndex(static_cast<size_t>(edgeIndexSize));
  env->GetFloatArrayRegion(tgtNodesJ, 0, tgtSize, tgtNodes.data());
  env->GetFloatArrayRegion(srcNodesJ, 0, srcSize, srcNodes.data());
  env->GetFloatArrayRegion(edgeFeatJ, 0, edgeFeatSize, edgeFeat.data());
  env->GetIntArrayRegion(edgeIndexJ, 0, edgeIndexSize, edgeIndex.data());
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  auto* packed = reinterpret_cast<float*>(g_cachedBlock000EdgeMeshCase.inputBuffer.data());
  for (int edge = 0; edge < edgeCount; ++edge) {
    const int src = edgeIndex[edge];
    const int tgt = edgeIndex[edge + edgeCount];
    const int rowBase = edge * latent * 3;
    const int tgtBase = tgt * latent;
    const int srcBase = src * latent;
    const int edgeBase = edge * latent;
    std::memcpy(packed + rowBase, tgtNodes.data() + tgtBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent, srcNodes.data() + srcBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent * 2, edgeFeat.data() + edgeBase, static_cast<size_t>(latent) * sizeof(float));
  }

  const auto executeStatus = g_cachedBlock000EdgeMeshCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      g_cachedBlock000EdgeMeshCase.graphHandle,
      &g_cachedBlock000EdgeMeshCase.inputTensor,
      1,
      &g_cachedBlock000EdgeMeshCase.outputTensor,
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

  const jsize outputFloats = static_cast<jsize>(expectedOutputFloats);
  jfloatArray outputJ = env->NewFloatArray(outputFloats);
  if (outputJ == nullptr) return nullptr;

  if (!g_cachedBlock000EdgeMeshCase.layernormGamma.empty() &&
      !g_cachedBlock000EdgeMeshCase.layernormBeta.empty()) {
    std::vector<float> post(static_cast<size_t>(outputFloats));
    hood::qnn::ApplyLayerNormRows(
        reinterpret_cast<const float*>(g_cachedBlock000EdgeMeshCase.outputBuffer.data()),
        edgeCount,
        latent,
        g_cachedBlock000EdgeMeshCase.layernormGamma.data(),
        g_cachedBlock000EdgeMeshCase.layernormBeta.data(),
        g_cachedBlock000EdgeMeshCase.layernormEps,
        post.data());
    env->SetFloatArrayRegion(outputJ, 0, outputFloats, post.data());
  } else {
    env->SetFloatArrayRegion(
        outputJ, 0, outputFloats, reinterpret_cast<const jfloat*>(g_cachedBlock000EdgeMeshCase.outputBuffer.data()));
  }
  return outputJ;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnBlock000EdgeMeshMlpBodyCasePackedAgg(
    JNIEnv* env,
    jclass,
    jfloatArray tgtNodesJ,
    jfloatArray srcNodesJ,
    jfloatArray edgeFeatJ,
    jintArray edgeIndexJ,
    jint latent) {
  if (!g_cachedBlock000EdgeMeshCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "block_0_0_edge_mesh case not initialized");
    return nullptr;
  }

  const jsize tgtFloats = env->GetArrayLength(tgtNodesJ);
  const jsize srcFloats = env->GetArrayLength(srcNodesJ);
  const jsize edgeFloats = env->GetArrayLength(edgeFeatJ);
  const jsize edgeIndexCount = env->GetArrayLength(edgeIndexJ);
  if (latent <= 0 || tgtFloats % latent != 0 || srcFloats % latent != 0 || edgeFloats % latent != 0 ||
      edgeIndexCount % 2 != 0) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "invalid packed edge mesh dimensions");
    return nullptr;
  }

  const int edgeCount = edgeIndexCount / 2;
  const int targetNodeCount = tgtFloats / latent;
  const int expectedInputFloats = edgeCount * latent * 3;
  const int expectedOutputFloats = edgeCount * latent;
  if (edgeFloats != edgeCount * latent ||
      expectedInputFloats * static_cast<int>(sizeof(float)) != static_cast<int>(g_cachedBlock000EdgeMeshCase.inputBuffer.size()) ||
      expectedOutputFloats * static_cast<int>(sizeof(float)) != static_cast<int>(g_cachedBlock000EdgeMeshCase.outputBuffer.size())) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "cached tensor shape mismatch");
    return nullptr;
  }

  std::vector<float> tgtNodes(static_cast<size_t>(tgtFloats));
  std::vector<float> srcNodes(static_cast<size_t>(srcFloats));
  std::vector<float> edgeFeat(static_cast<size_t>(edgeFloats));
  std::vector<jint> edgeIndex(static_cast<size_t>(edgeIndexCount));
  env->GetFloatArrayRegion(tgtNodesJ, 0, tgtFloats, tgtNodes.data());
  env->GetFloatArrayRegion(srcNodesJ, 0, srcFloats, srcNodes.data());
  env->GetFloatArrayRegion(edgeFeatJ, 0, edgeFloats, edgeFeat.data());
  env->GetIntArrayRegion(edgeIndexJ, 0, edgeIndexCount, edgeIndex.data());

  auto* packed = reinterpret_cast<float*>(g_cachedBlock000EdgeMeshCase.inputBuffer.data());
  for (int edge = 0; edge < edgeCount; ++edge) {
    const int src = edgeIndex[edge];
    const int tgt = edgeIndex[edge + edgeCount];
    const int rowBase = edge * latent * 3;
    const int tgtBase = tgt * latent;
    const int srcBase = src * latent;
    const int edgeBase = edge * latent;
    std::memcpy(packed + rowBase, tgtNodes.data() + tgtBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent, srcNodes.data() + srcBase, static_cast<size_t>(latent) * sizeof(float));
    std::memcpy(packed + rowBase + latent * 2, edgeFeat.data() + edgeBase, static_cast<size_t>(latent) * sizeof(float));
  }

  const auto executeStatus = g_cachedBlock000EdgeMeshCase.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      g_cachedBlock000EdgeMeshCase.graphHandle,
      &g_cachedBlock000EdgeMeshCase.inputTensor,
      1,
      &g_cachedBlock000EdgeMeshCase.outputTensor,
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

  std::vector<float> post(static_cast<size_t>(expectedOutputFloats));
  const float* qnnOutput = reinterpret_cast<const float*>(g_cachedBlock000EdgeMeshCase.outputBuffer.data());
  if (!g_cachedBlock000EdgeMeshCase.layernormGamma.empty() &&
      !g_cachedBlock000EdgeMeshCase.layernormBeta.empty()) {
    hood::qnn::ApplyLayerNormRows(
        qnnOutput,
        edgeCount,
        latent,
        g_cachedBlock000EdgeMeshCase.layernormGamma.data(),
        g_cachedBlock000EdgeMeshCase.layernormBeta.data(),
        g_cachedBlock000EdgeMeshCase.layernormEps,
        post.data());
  } else {
    std::memcpy(post.data(), qnnOutput, static_cast<size_t>(expectedOutputFloats) * sizeof(float));
  }

  std::vector<float> aggregated(static_cast<size_t>(targetNodeCount * latent), 0.0f);
  for (int edge = 0; edge < edgeCount; ++edge) {
    const int tgt = edgeIndex[edge + edgeCount];
    const int inBase = edge * latent;
    const int outBase = tgt * latent;
    for (int f = 0; f < latent; ++f) {
      aggregated[outBase + f] += post[inBase + f];
    }
  }

  const jsize outputFloats = static_cast<jsize>(aggregated.size());
  jfloatArray outputJ = env->NewFloatArray(outputFloats);
  if (outputJ == nullptr) return nullptr;
  env->SetFloatArrayRegion(outputJ, 0, outputFloats, aggregated.data());
  return outputJ;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_primeCachedQnnMeshEdgeState(
    JNIEnv* env,
    jclass,
    jfloatArray edgeFeatJ,
    jint latent) {
  if (edgeFeatJ == nullptr || latent <= 0) {
    return env->NewStringUTF("invalid_args");
  }
  const jsize edgeFeatSize = env->GetArrayLength(edgeFeatJ);
  if ((edgeFeatSize % latent) != 0) {
    return env->NewStringUTF("shape_mismatch");
  }
  const int edgeCount = edgeFeatSize / latent;
  g_cachedMeshEdgeState.assign(static_cast<size_t>(edgeFeatSize), 0.0f);
  env->GetFloatArrayRegion(edgeFeatJ, 0, edgeFeatSize, g_cachedMeshEdgeState.data());
  if (env->ExceptionCheck()) {
    ResetCachedMeshEdgeState();
    return nullptr;
  }
  g_cachedMeshEdgeStateLatent = latent;
  g_cachedMeshEdgeStateEdgeCount = edgeCount;
  std::ostringstream oss;
  oss << "primed edges=" << edgeCount << " latent=" << latent;
  return env->NewStringUTF(oss.str().c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnBlock000EdgeMeshMlpBodyCaseStateAgg(
    JNIEnv* env,
    jclass,
    jfloatArray clothNodesJ,
    jintArray edgeIndexJ,
    jint latent) {
  if (!g_cachedBlock000EdgeMeshCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "cached qnn block_0_0 edge mesh case is not initialized");
    return nullptr;
  }
  const jsize clothSize = env->GetArrayLength(clothNodesJ);
  const jsize edgeIndexSize = env->GetArrayLength(edgeIndexJ);
  if (clothNodesJ == nullptr || edgeIndexJ == nullptr || latent <= 0 || (clothSize % latent) != 0 ||
      (edgeIndexSize % 2) != 0) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "invalid block_0_0 mesh state args");
    return nullptr;
  }
  std::vector<float> clothNodes(static_cast<size_t>(clothSize));
  std::vector<jint> edgeIndex(static_cast<size_t>(edgeIndexSize));
  env->GetFloatArrayRegion(clothNodesJ, 0, clothSize, clothNodes.data());
  env->GetIntArrayRegion(edgeIndexJ, 0, edgeIndexSize, edgeIndex.data());
  if (env->ExceptionCheck()) return nullptr;

  std::vector<float> aggregated;
  if (!ExecuteMeshEdgeCaseToAggregated(g_cachedBlock000EdgeMeshCase, clothNodes, edgeIndex, latent, aggregated)) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exClass, "block_0_0 mesh QNN execution failed");
    return nullptr;
  }
  jfloatArray outputJ = env->NewFloatArray(static_cast<jsize>(aggregated.size()));
  if (outputJ == nullptr) return nullptr;
  env->SetFloatArrayRegion(outputJ, 0, static_cast<jsize>(aggregated.size()), aggregated.data());
  return outputJ;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_runCachedQnnBlock001EdgeMeshMlpBodyCaseStateAgg(
    JNIEnv* env,
    jclass,
    jfloatArray clothNodesJ,
    jintArray edgeIndexJ,
    jint latent) {
  if (!g_cachedBlock001EdgeMeshCase.initialized) {
    jclass exClass = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exClass, "cached qnn block_0_1 edge mesh case is not initialized");
    return nullptr;
  }
  const jsize clothSize = env->GetArrayLength(clothNodesJ);
  const jsize edgeIndexSize = env->GetArrayLength(edgeIndexJ);
  if (clothNodesJ == nullptr || edgeIndexJ == nullptr || latent <= 0 || (clothSize % latent) != 0 ||
      (edgeIndexSize % 2) != 0) {
    jclass exClass = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exClass, "invalid block_0_1 mesh state args");
    return nullptr;
  }
  std::vector<float> clothNodes(static_cast<size_t>(clothSize));
  std::vector<jint> edgeIndex(static_cast<size_t>(edgeIndexSize));
  env->GetFloatArrayRegion(clothNodesJ, 0, clothSize, clothNodes.data());
  env->GetIntArrayRegion(edgeIndexJ, 0, edgeIndexSize, edgeIndex.data());
  if (env->ExceptionCheck()) return nullptr;

  std::vector<float> aggregated;
  if (!ExecuteMeshEdgeCaseToAggregated(g_cachedBlock001EdgeMeshCase, clothNodes, edgeIndex, latent, aggregated)) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exClass, "block_0_1 mesh QNN execution failed");
    return nullptr;
  }
  jfloatArray outputJ = env->NewFloatArray(static_cast<jsize>(aggregated.size()));
  if (outputJ == nullptr) return nullptr;
  env->SetFloatArrayRegion(outputJ, 0, static_cast<jsize>(aggregated.size()), aggregated.data());
  return outputJ;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_exportCachedQnnMeshEdgeState(
    JNIEnv* env,
    jclass) {
  jfloatArray outputJ = env->NewFloatArray(static_cast<jsize>(g_cachedMeshEdgeState.size()));
  if (outputJ == nullptr) return nullptr;
  if (!g_cachedMeshEdgeState.empty()) {
    env->SetFloatArrayRegion(outputJ, 0, static_cast<jsize>(g_cachedMeshEdgeState.size()), g_cachedMeshEdgeState.data());
  }
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

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_releaseCachedQnnBlock000EdgeMeshMlpBodyCase(
    JNIEnv* env,
    jclass) {
  const bool wasInitialized = g_cachedBlock000EdgeMeshCase.initialized;
  ReleaseCachedCase(g_cachedBlock000EdgeMeshCase);
  ResetCachedMeshEdgeState();
  const std::string result = wasInitialized ? "released" : "already_released";
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_magcode_hoodonnxtest_NativeQairtBridge_releaseCachedQnnBlock001EdgeMeshMlpBodyCase(
    JNIEnv* env,
    jclass) {
  const bool wasInitialized = g_cachedBlock001EdgeMeshCase.initialized;
  ReleaseCachedCase(g_cachedBlock001EdgeMeshCase);
  ResetCachedMeshEdgeState();
  const std::string result = wasInitialized ? "released" : "already_released";
  return env->NewStringUTF(result.c_str());
}
