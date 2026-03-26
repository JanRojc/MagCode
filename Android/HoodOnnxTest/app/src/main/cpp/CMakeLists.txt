cmake_minimum_required(VERSION 3.22.1)

project(hood_gpu)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES ON)

add_library(
    hood_gpu
    SHARED
    qairt_bridge.cpp
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/share/QNN/converter/jni/QnnWrapperUtils.cpp
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/share/QNN/converter/jni/linux/QnnModelPal.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(dl-lib dl)

target_compile_features(hood_gpu PRIVATE cxx_std_17)
target_include_directories(
    hood_gpu
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ort_headers
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/include
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/include/QNN
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/include/QNN/System
    /Users/jan.rojc/Documents/MagCode/Libs/qairt/2.44.0.260225/share/QNN/converter/jni
)

target_link_options(hood_gpu PRIVATE "-Wl,-z,max-page-size=16384")

target_link_libraries(
    hood_gpu
    ${log-lib}
    ${android-lib}
    ${dl-lib}
)
