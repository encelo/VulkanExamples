cmake_minimum_required(VERSION 3.5...3.30)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # Enables override of options from parent CMakeLists.txt
endif()

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW) # Enables use of MSVC_RUNTIME_LIBRARY
endif()

if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW) # Enables clean /W4 override for MSVC
endif()

project(meshoptimizer VERSION 1.1 LANGUAGES CXX)

set(SOURCES
    src/meshoptimizer.h
    src/allocator.cpp
    src/clusterizer.cpp
    src/indexanalyzer.cpp
    src/indexcodec.cpp
    src/indexgenerator.cpp
    src/meshletcodec.cpp
    src/meshletutils.cpp
    src/opacitymap.cpp
    src/overdrawoptimizer.cpp
    src/partition.cpp
    src/quantization.cpp
    src/rasterizer.cpp
    src/simplifier.cpp
    src/spatialorder.cpp
    src/stripifier.cpp
    src/vcacheoptimizer.cpp
    src/vertexcodec.cpp
    src/vertexfilter.cpp
    src/vfetchoptimizer.cpp
)

add_library(meshoptimizer STATIC ${SOURCES})

target_include_directories(meshoptimizer INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
