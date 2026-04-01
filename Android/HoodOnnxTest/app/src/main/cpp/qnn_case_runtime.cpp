#include "qnn_case_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>

namespace hood::qnn {

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
        name != "libQnnHtpPrepare.so" &&
        name.find("libQnnHtpV") != 0 &&
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
      FindFirstMatchingFile(bundle, {"expected_output.raw", "expected_node_encoder_output.raw", "node_encoder_probe_expected.raw", "expected_output.raw"}),
      FindFirstMatchingFile(bundle, {"output_probe/output/Result_0/output.raw", "hybrid_output.raw"}),
      FindFirstMatchingFile(bundle, {"layernorm_gamma.bin"}),
      FindFirstMatchingFile(bundle, {"layernorm_beta.bin"}),
      FindFirstMatchingFile(bundle, {"manifest.json"}),
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
                                float atol,
                                float rtol) {
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
  DlCloseQuiet(state.rpcHandle);
  DlCloseQuiet(state.hardwareHandle);
  DlCloseQuiet(state.gpuExtHandle);
  DlCloseQuiet(state.systemHandle);
  DlCloseQuiet(state.libcxxHandle);
  state = CachedQnnCaseState{};
}

static bool LoadLayerNormParams(const QnnCasePaths& paths,
                                CachedQnnCaseState& state,
                                std::string& error) {
  if (paths.layernormGamma.empty() || paths.layernormBeta.empty() || paths.manifest.empty()) {
    error = "layernorm_files_missing";
    return false;
  }
  state.layernormGamma = ReadFloatFile(paths.layernormGamma);
  state.layernormBeta = ReadFloatFile(paths.layernormBeta);
  if (state.layernormGamma.empty() || state.layernormBeta.empty() ||
      state.layernormGamma.size() != state.layernormBeta.size()) {
    error = "layernorm_data_invalid";
    return false;
  }

  std::ifstream manifest(paths.manifest);
  if (!manifest) {
    error = "manifest_open_failed";
    return false;
  }
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
  state.layernormCols = static_cast<int>(findNumber("layernorm_dim", static_cast<double>(state.layernormGamma.size())));
  state.layernormEps = static_cast<float>(findNumber("layernorm_eps", 1.0e-5));
  if (state.layernormCols <= 0 || state.layernormRows <= 0 ||
      static_cast<size_t>(state.layernormCols) != state.layernormGamma.size()) {
    error = "layernorm_manifest_invalid";
    return false;
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
      !FileExists(paths.backend) || paths.model.empty() || paths.input.empty()) {
    oss << " | missing_required_files";
    return oss.str();
  }

  std::string layerNormError;
  if (!LoadLayerNormParams(paths, state, layerNormError)) {
    oss << " | " << layerNormError;
    return oss.str();
  }

  state.bundleDir = bundleDir;
  state.inputBytes = ReadBinaryFile(paths.input);
  state.expectedBytes = paths.expected.empty() ? std::vector<uint8_t>{} : ReadBinaryFile(paths.expected);

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
    const std::string adspPath = bundleDir + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
    std::string ldLibraryPath = getenv("LD_LIBRARY_PATH") == nullptr ? "" : getenv("LD_LIBRARY_PATH");
    auto appendLdPath = [&](const char* path) {
      if (ldLibraryPath.find(path) != std::string::npos) return;
      if (!ldLibraryPath.empty()) ldLibraryPath += ":";
      ldLibraryPath += path;
    };
    appendLdPath("/vendor/lib64");
    appendLdPath("/system/lib64");
    appendLdPath("/vendor/lib");
    appendLdPath("/system/lib");
    setenv("LD_LIBRARY_PATH", ldLibraryPath.c_str(), 1);
    for (const auto& hardwareCandidate : {"/system/lib64/libhardware.so", "/vendor/lib64/libhardware.so",
                                          "/system/lib/libhardware.so", "/vendor/lib/libhardware.so"}) {
      state.hardwareHandle = dlopen(hardwareCandidate, RTLD_NOW | RTLD_GLOBAL);
      if (state.hardwareHandle != nullptr) break;
    }
    if (state.hardwareHandle == nullptr) {
      oss << " | htp_hardware_dlopen_fail=" << DlErrorString();
      ReleaseCachedCase(state);
      return oss.str();
    }
    for (const auto& rpcCandidate : {"/vendor/lib64/libcdsprpc.so", "/system/lib64/libcdsprpc.so",
                                     "/vendor/lib/libcdsprpc.so", "/system/lib/libcdsprpc.so"}) {
      state.rpcHandle = dlopen(rpcCandidate, RTLD_NOW | RTLD_GLOBAL);
      if (state.rpcHandle != nullptr) break;
    }
    if (state.rpcHandle == nullptr) {
      oss << " | htp_rpc_dlopen_fail=" << DlErrorString();
      ReleaseCachedCase(state);
      return oss.str();
    }
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

void ApplyLayerNormRows(const float* input,
                        int rows,
                        int cols,
                        const float* gamma,
                        const float* beta,
                        float eps,
                        float* output) {
  for (int r = 0; r < rows; ++r) {
    const int rowOffset = r * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) {
      mean += input[rowOffset + c];
    }
    mean /= static_cast<double>(cols);

    double variance = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double centered = static_cast<double>(input[rowOffset + c]) - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(cols);

    const double invStd = 1.0 / std::sqrt(variance + static_cast<double>(eps));
    for (int c = 0; c < cols; ++c) {
      const double normalized =
          (static_cast<double>(input[rowOffset + c]) - mean) * invStd;
      output[rowOffset + c] = static_cast<float>(
          normalized * static_cast<double>(gamma[c]) + static_cast<double>(beta[c]));
    }
  }
}

}  // namespace hood::qnn
