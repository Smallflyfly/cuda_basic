//
// Created by smallflyfly on 2021/5/27.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_code.h"


__global__ void sumMat(float *a, float *b, float *c, int nx, int ny) {
    int threadIdX = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIdy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = threadIdX + threadIdy * ny;
    c[idx] = a[idx] + b[idx];
}


int main(int argc, char **argv) {

    initDevice(0);

    int nx = 1<<13;
    int ny = 1<<13;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc
    float *ah = (float*) malloc(nBytes);
    float *bh = (float*) malloc(nBytes);
    float *ch = (float*) malloc(nBytes);

    initData(ah, nxy);
    initData(bh, nxy);

    float *ad = NULL;
    float *bd = NULL;
    float *cd = NULL;

    CHECK(cudaMalloc((void **)&ad, nBytes));
    CHECK(cudaMalloc((void **)&bd, nBytes));
    CHECK(cudaMalloc((void **)&cd, nBytes));

    CHECK(cudaMemcpyAsync(ad, ah, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(bd, bh, nBytes, cudaMemcpyHostToDevice));

    int dimx = argc > 1 ? atoi(argv[1]) : 64;
    int dimy = argc > 2 ? atoi(argv[2]) : 64;

    double iStart, iElaps;

    // 2d block 2d grid
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    iStart = cpuSecond();

    sumMat<<<grid, block>>>(ad, bd, cd, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration <<<(%d %d), (%d, %d)>>> | %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaMemcpyAsync(ch, cd, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    free(ah);
    free(bh);
    free(ch);

    cudaDeviceReset();

    return 0;
}