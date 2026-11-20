cmake_minimum_required(VERSION 3.18)
project(llama_mpi_disagg CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

# Path to a llama.cpp source checkout. It is built in-tree as *static* libraries
# so llama.cpp runs inside each MPI process (no server). Pass GPU options straight
# through, e.g. -DGGML_CUDA=ON, -DGGML_HIP=ON, -DGGML_SYCL=ON, -DGGML_VULKAN=ON.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp source directory")
if (NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "llama.cpp not found at LLAMA_CPP_DIR=${LLAMA_CPP_DIR}")
endif()

set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)   # common_speculative, common_sampler, chat templates
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

# GPU-direct KV transfer (--kv-transport device) needs llama.cpp with the split-state API from
# patches/llama.cpp-state-seq-split.patch; without it the build still works with --kv-transport host.
file(READ "${LLAMA_CPP_DIR}/include/llama.h" _llama_h)
string(FIND "${_llama_h}" "LLAMA_STATE_SEQ_SPLIT_API" _split_pos)
if (_split_pos EQUAL -1)
    message(STATUS "llama-mpi-disagg: split-state API not found in llama.cpp: --kv-transport device unavailable "
                   "(apply patches/llama.cpp-state-seq-split.patch to enable it)")
else()
    message(STATUS "llama-mpi-disagg: split-state API found: --kv-transport device available")
endif()

add_executable(llama_mpi_disagg src/llama_mpi_disagg.cpp)
target_link_libraries(llama_mpi_disagg PRIVATE llama-common llama ggml MPI::MPI_CXX Threads::Threads)
install(TARGETS llama_mpi_disagg RUNTIME DESTINATION bin)
