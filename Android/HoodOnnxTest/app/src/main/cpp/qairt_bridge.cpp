#include <jni.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

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
