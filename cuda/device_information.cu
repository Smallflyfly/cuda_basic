//
// Created by smallflyfly on 2021/5/18.
//

#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    int deviceCount = 0;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);
    if (errorId != cudaSuccess) {
        printf("cudaDeviceCount returned %d\n -> %s\n", (int)errorId, cudaGetErrorString(errorId));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device\n", deviceCount);
    }

    int device = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    printf("Device %d:\"%s\"\n", device, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("Driver version / Runtime Version               %d.%d  /  %d.%d\n", driverVersion/1000, (driverVersion%100)/10,
           runtimeVersion/1000, (runtimeVersion%100)/10);

    printf("CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

    printf("Total amount of global memory:                 %.2f GBytes\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3));

    printf("GPU clock rate:                                %.0f MHz  (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    printf("Max number of thread per multiprocessor:       %d\n", deviceProp.maxThreadsPerMultiProcessor);

    printf("max number of thread per block:                %d\n", deviceProp.maxThreadsPerBlock);

    printf("max size of each dims of a block:              %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

    printf("max size of each dim of a grid:                %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    exit(EXIT_SUCCESS);
}