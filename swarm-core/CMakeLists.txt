cmake_minimum_required(VERSION 3.15)
project(SwarmCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find Python3 first for proper pybind11 integration
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Auto-detect pybind11 in virtual environment if activated
set(PYBIND11_FINDPYTHON ON)
if(DEFINED ENV{VIRTUAL_ENV})
    # Try to find pybind11 in virtual environment
    find_package(pybind11 QUIET HINTS "$ENV{VIRTUAL_ENV}/lib/python3.12/site-packages/pybind11/share/cmake")
    if(NOT pybind11_FOUND)
        # Fallback to standard search paths
        find_package(pybind11 CONFIG REQUIRED)
    endif()
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()
find_package(CUDA)

file(GLOB SRC_FILES src/*.cpp)
list(APPEND SRC_FILES src/cyber_grid.cpp)
# Exclude emergence_analyzer.cpp to avoid missing symbol errors
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/emergence_analyzer.cpp)
pybind11_add_module(swarm_core ${SRC_FILES} bindings/pybind_module.cpp)
target_include_directories(swarm_core PRIVATE include)
target_link_libraries(swarm_core PRIVATE pybind11::module Python3::Module)
if(CUDA_FOUND AND TARGET CUDA::cudart)
    target_link_libraries(swarm_core PRIVATE CUDA::cudart)
    message(STATUS "CUDA library found - linking CUDA runtime")
else()
    message(STATUS "CUDA not fully available - building without CUDA acceleration")
endif()

install(TARGETS swarm_core DESTINATION .)
