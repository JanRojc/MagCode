cmake_minimum_required(VERSION 3.22.1)

project(hood_gpu)

add_library(
    hood_gpu
    SHARED
    gles_compute_backend.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(egl-lib EGL)
find_library(gles-lib GLESv3)

target_compile_features(hood_gpu PRIVATE cxx_std_17)

target_link_libraries(
    hood_gpu
    ${log-lib}
    ${android-lib}
    ${egl-lib}
    ${gles-lib}
)
