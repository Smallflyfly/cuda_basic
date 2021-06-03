//
// Created by smallflyfly on 2021/6/3.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_code.h"


__global__ void matMultiply(float *a, float *b, float *c, int width, int height) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= width || ty >= height) return;

    float mulValue = 0.0;
    for (int i = 0; i < width; i++) {
        mulValue += a[ty * width + i] * b[tx * height + i];
    }
    c[ty * width + tx] = mulValue;
}


int main() {
    int width = 1 << 2;
    int height = 1 << 2;

    float *ah, *bh, *ch;
    unsigned nBytes = width * height * sizeof(float);

    ah = (float*)malloc(nBytes);
    bh = (float*)malloc(nBytes);
    ch = (float*)malloc(nBytes);

    for (int i = 0; i < width * height; i++) {
        ah[i] = 1.0;
        bh[i] = 2.0;
    }

    float *ad, *bd, *cd;

    // malloc device
    CHECK(cudaMalloc((void**)&ad, nBytes));
    CHECK(cudaMalloc((void**)&bd, nBytes));
    CHECK(cudaMalloc((void**)&cd, nBytes));

    // copy host data to device
    CHECK(cudaMemcpyAsync(ad, ah, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(bd, bh, nBytes, cudaMemcpyHostToDevice));

    // block grid
    dim3 blockSize(2, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // run kernel
    matMultiply<<<gridSize, blockSize>>>(ad, bd, cd, width, height);
    cudaDeviceSynchronize();

    // copy result from device to host
    CHECK(cudaMemcpyAsync(ch, cd, nBytes, cudaMemcpyDeviceToHost));

    float maxError = 0.0;
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", ch[i]);
        if ((i+1) % width == 0) printf("\n");
    }


    // 释放内存

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    free(ah);
    free(bh);
    free(ch);

    cudaDeviceReset();

    return 0;
}