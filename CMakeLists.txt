cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

set(VERSION_MAJOR 1)
add_definitions(-DVERSION_MAJOR=${VERSION_MAJOR})    # 一级版本号
set(VERSION_MINOR 0)
add_definitions(-DVERSION_MINOR=${VERSION_MINOR})    # 二级版本号
set(VERSION_BUILD 3)
add_definitions(-DVERSION_BUILD=${VERSION_BUILD})    # 三级版本号
set(REVISION_PREFIX 130)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)
set(UTILITY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/utility)
include(utility/cmake/utility.cmake)

set(CMAKE_CXX_STANDARD 17)
# Set C++ version and optimization level
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug build enabled.")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
endif()

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-8.6.1.6/)
endif()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SRC_LIST
        src/engine.cpp
)

foreach (sourcefile ${SRC_LIST})
    redefine_file_base(${sourcefile})
endforeach ()


add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BUILD}.${REVISION} SOVERSION 1)

#target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog) #spdlog
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC -lstdc++fs)

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)