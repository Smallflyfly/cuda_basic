cmake_minimum_required(VERSION 3.19)
project(cuda)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

cuda_add_executable(cuda
        matrix_multipy.cu)
