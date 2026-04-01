#include "qnn_case_runtime.h"

#include <android/log.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

using hood::qnn::CachedQnnCaseState;

constexpr const char* kLogTag = "HoodHtpDaemon";
constexpr uint32_t kMagic = 0x48445450;  // HDTP
constexpr uint32_t kVersion = 1;
constexpr uint32_t kOpRunNodeEncoderStage = 1;
constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusBadRequest = 1;
constexpr uint32_t kStatusRuntimeError = 2;

struct RequestHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t op;
  uint32_t floatCount;
};

struct ResponseHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t status;
  uint32_t outputFloatCount;
  uint64_t qnnNs;
  uint64_t layerNormNs;
  uint64_t totalNs;
};

CachedQnnCaseState g_state;
int g_serverFd = -1;
std::string g_socketPath;

void LogInfo(const std::string& msg) {
  __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s", msg.c_str());
}

void LogError(const std::string& msg) {
  __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s", msg.c_str());
}

bool ReadExact(int fd, void* data, size_t size) {
  auto* cursor = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    const ssize_t n = TEMP_FAILURE_RETRY(read(fd, cursor, remaining));
    if (n <= 0) return false;
    cursor += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

bool WriteExact(int fd, const void* data, size_t size) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    const ssize_t n = TEMP_FAILURE_RETRY(write(fd, cursor, remaining));
    if (n <= 0) return false;
    cursor += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

void Cleanup() {
  hood::qnn::ReleaseCachedCase(g_state);
  if (g_serverFd >= 0) {
    close(g_serverFd);
    g_serverFd = -1;
  }
  if (!g_socketPath.empty()) {
    unlink(g_socketPath.c_str());
  }
}

void HandleSignal(int) {
  Cleanup();
  _exit(0);
}

bool SendErrorResponse(int fd, uint32_t status) {
  const ResponseHeader header{
      .magic = kMagic,
      .version = kVersion,
      .status = status,
      .outputFloatCount = 0,
      .qnnNs = 0,
      .layerNormNs = 0,
      .totalNs = 0,
  };
  return WriteExact(fd, &header, sizeof(header));
}

bool HandleRunRequest(int clientFd, uint32_t floatCount) {
  if (!g_state.initialized) {
    LogError("run request before init");
    return SendErrorResponse(clientFd, kStatusRuntimeError);
  }

  const uint32_t expectedFloatCount = static_cast<uint32_t>(g_state.inputBuffer.size() / sizeof(float));
  if (floatCount != expectedFloatCount) {
    LogError("input float count mismatch");
    return SendErrorResponse(clientFd, kStatusBadRequest);
  }

  if (!ReadExact(clientFd, g_state.inputBuffer.data(), g_state.inputBuffer.size())) {
    return false;
  }

  const auto totalStart = std::chrono::steady_clock::now();
  const auto qnnStart = totalStart;
  const auto executeStatus = g_state.provider->QNN_INTERFACE_VER_NAME.graphExecute(
      g_state.graphHandle,
      &g_state.inputTensor,
      1,
      &g_state.outputTensor,
      1,
      nullptr,
      nullptr);
  const auto qnnEnd = std::chrono::steady_clock::now();
  if (executeStatus != QNN_SUCCESS) {
    LogError("graphExecute failed");
    return SendErrorResponse(clientFd, kStatusRuntimeError);
  }

  const size_t outputFloatCount = g_state.outputBuffer.size() / sizeof(float);
  if (static_cast<int>(outputFloatCount) != g_state.layernormRows * g_state.layernormCols) {
    LogError("layernorm shape mismatch");
    return SendErrorResponse(clientFd, kStatusRuntimeError);
  }

  std::vector<float> normalized(outputFloatCount);
  const auto layerNormStart = std::chrono::steady_clock::now();
  hood::qnn::ApplyLayerNormRows(
      reinterpret_cast<const float*>(g_state.outputBuffer.data()),
      g_state.layernormRows,
      g_state.layernormCols,
      g_state.layernormGamma.data(),
      g_state.layernormBeta.data(),
      g_state.layernormEps,
      normalized.data());
  const auto layerNormEnd = std::chrono::steady_clock::now();

  const auto qnnNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(qnnEnd - qnnStart).count());
  const auto layerNormNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(layerNormEnd - layerNormStart).count());
  const auto totalNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(layerNormEnd - totalStart).count());

  const ResponseHeader header{
      .magic = kMagic,
      .version = kVersion,
      .status = kStatusOk,
      .outputFloatCount = static_cast<uint32_t>(outputFloatCount),
      .qnnNs = qnnNs,
      .layerNormNs = layerNormNs,
      .totalNs = totalNs,
  };
  if (!WriteExact(clientFd, &header, sizeof(header))) {
    return false;
  }
  return WriteExact(clientFd, normalized.data(), normalized.size() * sizeof(float));
}

bool ServeClient(int clientFd) {
  while (true) {
    RequestHeader header{};
    if (!ReadExact(clientFd, &header, sizeof(header))) {
      return false;
    }
    if (header.magic != kMagic || header.version != kVersion) {
      LogError("bad request header");
      return SendErrorResponse(clientFd, kStatusBadRequest);
    }
    switch (header.op) {
      case kOpRunNodeEncoderStage:
        if (!HandleRunRequest(clientFd, header.floatCount)) return false;
        break;
      default:
        if (!SendErrorResponse(clientFd, kStatusBadRequest)) return false;
        break;
    }
  }
}

bool InitServerSocket(const std::string& socketPath) {
  g_socketPath = socketPath;
  unlink(socketPath.c_str());
  g_serverFd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (g_serverFd < 0) {
    LogError("socket() failed");
    return false;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (socketPath.size() >= sizeof(addr.sun_path)) {
    LogError("socket path too long");
    return false;
  }
  std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);
  if (bind(g_serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    LogError("bind() failed");
    return false;
  }
  if (listen(g_serverFd, 1) != 0) {
    LogError("listen() failed");
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    LogError("usage: daemon <bundle_dir> <socket_path>");
    return 2;
  }

  const std::string bundleDir = argv[1];
  const std::string socketPath = argv[2];
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);

  const auto initResult = hood::qnn::InitializeCachedCase(bundleDir, g_state);
  LogInfo("init: " + initResult);
  if (!g_state.initialized) {
    Cleanup();
    return 3;
  }

  if (!InitServerSocket(socketPath)) {
    Cleanup();
    return 4;
  }
  LogInfo("listening: " + socketPath);

  while (true) {
    const int clientFd = TEMP_FAILURE_RETRY(accept(g_serverFd, nullptr, nullptr));
    if (clientFd < 0) {
      LogError("accept() failed");
      break;
    }
    LogInfo("client connected");
    ServeClient(clientFd);
    close(clientFd);
    LogInfo("client disconnected");
  }

  Cleanup();
  return 0;
}
