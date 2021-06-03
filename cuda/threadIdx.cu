//
// Created by smallflyfly on 2021/5/17.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_code.h"

__global__ void printThreadIndex(float *a, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + iy * ny;

    printf("thread id(%d, %d) block(%d, %d) coordinate(%d, %d) global index %d val %f\n", threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y, ix, iy, idx, a[idx]);
}


int main(int argc, char **argv) {
    initDevice(0);
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = sizeof(float) * nxy;

    // malloc
    float *a_h = (float*) malloc(nBytes);
    initData(a_h, nxy);
//    printMatrix(a_h, nx, ny);

    float *a_d;
    CHECK(cudaMalloc((void**)&a_d, nBytes));

    CHECK(cudaMemcpyAsync(a_d, a_h, nBytes, cudaMemcpyHostToDevice));

    dim3 block(4, 2);
    dim3 grid((nx - 1) / block.x + 1, (ny-1) / block.y);

    printThreadIndex<<<grid, block>>>(a_d, nx, ny);

    CHECK(cudaDeviceSynchronize());
    cudaFree(a_d);
    free(a_h);

    cudaDeviceReset();

    return 0;
}