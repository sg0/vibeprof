cmake_minimum_required(VERSION 3.18)
project(vibeprof C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(MPI REQUIRED COMPONENTS C CXX)

# collector: markers + MPI wrappers (LD_PRELOAD or link)
add_library(vibeprof SHARED src/collector.cpp src/gpu_cupti.cpp)
target_include_directories(vibeprof PUBLIC include PRIVATE src)
target_link_libraries(vibeprof PRIVATE MPI::MPI_CXX ${CMAKE_DL_LIBS})

# no-op markers for applications that keep the markers in their code permanently
add_library(vibeprof_stub SHARED src/stub.c)
target_include_directories(vibeprof_stub PUBLIC include)

# perturbation library (ground truth for evaluation)
add_library(vibeperturb SHARED src/perturb.cpp)
target_include_directories(vibeperturb PRIVATE src)
target_link_libraries(vibeperturb PRIVATE MPI::MPI_CXX ${CMAKE_DL_LIBS})

# toy test program with known behavior
add_executable(vibeprof_toy tests/toy.c)
target_link_libraries(vibeprof_toy PRIVATE vibeprof_stub MPI::MPI_C m)

# GPU testing without a GPU: simulated CUDA device + CUPTI, and the GPU toy built against it
add_library(fakecuda SHARED tests/fakecuda/fakecuda.cpp)
target_include_directories(fakecuda PUBLIC tests/fakecuda)
add_executable(gpu_toy_sim tests/gpu_toy.cu)
set_source_files_properties(tests/gpu_toy.cu TARGET_DIRECTORY gpu_toy_sim PROPERTIES LANGUAGE CXX)
target_compile_options(gpu_toy_sim PRIVATE -x c++)
target_compile_definitions(gpu_toy_sim PRIVATE VIBEPROF_FAKECUDA)
target_link_libraries(gpu_toy_sim PRIVATE fakecuda vibeprof_stub MPI::MPI_CXX)

add_executable(kokkos_hooks_test tests/kokkos_hooks_test.c)
target_link_libraries(kokkos_hooks_test PRIVATE fakecuda MPI::MPI_C ${CMAKE_DL_LIBS})

# the same toy on a real NVIDIA GPU, when nvcc is available
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
        endif()
    endif()
    # nvcc compiles the toy, so it needs mpi.h explicitly: with compiler wrappers that add MPI themselves
    # (Cray cc/CC, mpicxx as CMAKE_CXX_COMPILER) FindMPI reports no include directory.
    find_path(VIBEPROF_MPI_INCLUDE_DIR mpi.h
              HINTS ${MPI_CXX_INCLUDE_DIRS} ${MPI_C_INCLUDE_DIRS} ENV MPICH_DIR ENV CRAY_MPICH_DIR ENV MPI_ROOT ENV MPI_HOME
              PATH_SUFFIXES include)
    find_package(CUDAToolkit)
    if (VIBEPROF_MPI_INCLUDE_DIR AND TARGET CUDA::cudart)
        add_executable(gpu_toy tests/gpu_toy_real.cu)
        target_include_directories(gpu_toy PRIVATE ${VIBEPROF_MPI_INCLUDE_DIR})
        target_compile_definitions(gpu_toy PRIVATE MPICH_SKIP_MPICXX OMPI_SKIP_MPICXX)   # C API only
        # link with the C++ compiler (MPI wrappers add their libraries there) plus the CUDA runtime
        set_target_properties(gpu_toy PROPERTIES LINKER_LANGUAGE CXX CUDA_RUNTIME_LIBRARY Shared)
        target_link_libraries(gpu_toy PRIVATE vibeprof_stub MPI::MPI_CXX CUDA::cudart)
        message(STATUS "vibeprof: building gpu_toy (mpi.h from ${VIBEPROF_MPI_INCLUDE_DIR})")
    else()
        message(STATUS "vibeprof: not building gpu_toy: mpi.h or the CUDA runtime not found "
                       "(set VIBEPROF_MPI_INCLUDE_DIR to the directory with mpi.h)")
    endif()
endif()

install(TARGETS vibeprof vibeprof_stub vibeperturb LIBRARY DESTINATION lib)
install(FILES include/vibeprof.h include/vibeprof.f90 DESTINATION include)
