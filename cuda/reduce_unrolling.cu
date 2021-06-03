//
// Created by smallflyfly on 2021/5/27.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_code.h"


__global__ void warmUp(int *iData, int *oData, int n) {
    unsigned int tid = threadIdx.x;
    if (tid >= n) return;

    int *data = iData + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            data[tid] += data[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    if (tid == 0) {
        oData[blockIdx.x] = data[0];
    }
}


int main(int argc, char **argv) {
    initDevice(0);
    int size = 1 << 24;

    int blockSize = 1024;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }
    dim3 block(blockSize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);

    size_t bytes = size * sizeof(int);

    int *iDataH = (int*) malloc(bytes);
    int *oDataH = (int*) malloc(grid.x * sizeof(int));
    int *tmp = (int*) malloc(bytes);

    initDataInt(iDataH, size);

    memcpy(tmp, iDataH, bytes);

    double iStart, iElaps;
    int gupSum = 0;

    int *iDataD = NULL;
    int *oDataD = NULL;

    CHECK(cudaMalloc((void**)&iDataD, bytes));
    CHECK(cudaMalloc((void**)&oDataD, grid.x * sizeof(int)));

    int cpuSum = 0;
    iStart = cpuSecond();
    for (int i = 0; i < size; i++) {
        cpuSum += tmp[i];
    }
    printf("cpu sum: %d\n", cpuSum);
    iElaps = cpuSecond() - iStart;
    printf("cpu cost time %lf ms\n", iElaps);

    CHECK(cudaMemcpyAsync(iDataD, iDataH, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();

    warmUp<<<grid.x / 2, block>>>(iDataD, oDataD, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("gpu warmup elapsed %lf ms\n", iElaps);





    // cpu
    int cpu;

    cudaDeviceReset();

    return 0;



}