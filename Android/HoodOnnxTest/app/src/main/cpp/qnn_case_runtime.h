#pragma once

#include <jni.h>

#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnModel.hpp"

namespace hood::qnn {

using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);
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
  void* hardwareHandle = nullptr;
  void* rpcHandle = nullptr;
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

template <typename T>
T ResolveSymbol(void* handle, const char* symbol) {
  return reinterpret_cast<T>(dlsym(handle, symbol));
}

std::string DlErrorString();
std::string QnnErrorString(Qnn_ErrorHandle_t code);
std::string ModelErrorString(qnn_wrapper_api::ModelError_t code);
std::filesystem::path JoinPath(const std::string& base, const std::string& child);
bool FileExists(const std::filesystem::path& path);
std::filesystem::path FindFirstMatchingFile(const std::filesystem::path& dir,
                                            const std::vector<std::string>& candidates);
std::filesystem::path FindProbeModelLib(const std::filesystem::path& dir);
QnnCasePaths DiscoverCasePaths(const std::string& bundleDir);
std::vector<uint8_t> ReadBinaryFile(const std::filesystem::path& path);
std::vector<float> ReadFloatFile(const std::filesystem::path& path);
size_t DataTypeByteSize(Qnn_DataType_t dataType);
size_t TensorByteSize(const Qnn_Tensor_t& tensor);
bool SetupTensorFromTemplate(const Qnn_Tensor_t& src,
                             Qnn_Tensor_t& dst,
                             std::vector<uint8_t>& ownedBuffer,
                             const uint8_t* initialData,
                             size_t initialSize);
std::string CompareFloatBuffers(const std::vector<uint8_t>& expectedBytes,
                                const std::vector<uint8_t>& actualBytes,
                                float atol = 1e-5f,
                                float rtol = 1e-4f);
std::string FormatMs(double ms);
void DlCloseQuiet(void*& handle);
const QnnInterface_t* SelectQnnInterface(const QnnInterface_t** providers, uint32_t count);
void ReleaseCachedCase(CachedQnnCaseState& state);
std::string InitializeCachedCase(const std::string& bundleDir, CachedQnnCaseState& state);
void ApplyLayerNormRows(const float* input,
                        int rows,
                        int cols,
                        const float* gamma,
                        const float* beta,
                        float eps,
                        float* output);

}  // namespace hood::qnn
