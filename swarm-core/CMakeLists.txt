cmake_minimum_required(VERSION 3.15)
project(SwarmCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SRC_FILES src/*.cpp)
list(APPEND SRC_FILES src/cyber_grid.cpp)
pybind11_add_module(swarm_core ${SRC_FILES})
target_include_directories(swarm_core PRIVATE include)
target_link_libraries(swarm_core PRIVATE CUDA::cudart)

install(TARGETS swarm_core DESTINATION .)
