cmake_minimum_required(VERSION 3.15)
project(SwarmCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA)

file(GLOB SRC_FILES src/*.cpp)
list(APPEND SRC_FILES src/cyber_grid.cpp)
# Exclude emergence_analyzer.cpp to avoid missing symbol errors
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/emergence_analyzer.cpp)
pybind11_add_module(swarm_core ${SRC_FILES} bindings/pybind_module.cpp)
target_include_directories(swarm_core PRIVATE include)
if(CUDA_FOUND AND TARGET CUDA::cudart)
    target_link_libraries(swarm_core PRIVATE CUDA::cudart)
    message(STATUS "CUDA library found - linking CUDA runtime")
else()
    message(STATUS "CUDA not fully available - building without CUDA acceleration")
endif()

install(TARGETS swarm_core DESTINATION .)
