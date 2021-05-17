//
// Created by smallflyfly on 2021/5/17.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_code.h"


void initData(float *a, int length) {
    for (int i=0; i<length; i++) {
        a[i] = i * 1.0;
    }
}

__global__ void sumCuda(const float *a, const float *b, float *sum_d) {
    int i = threadIdx.x;
    sum_d[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int device = 0;
    cudaSetDevice(device);

    int numElement = 10;

    int nBytes = sizeof(float) * numElement;

    float *a_h = (float*)malloc(nBytes);
    float *b_h = (float*)malloc(nBytes);
    float *sum_h = (float*) malloc(nBytes);

    memset(a_h, 0 , nBytes);
    memset(b_h, 0, nBytes);

    float *a_d, *b_d, *sum_d;
    CHECK(cudaMalloc((float**)&a_d, nBytes));
    CHECK(cudaMalloc((float**)&b_d, nBytes));
    CHECK(cudaMalloc((float**)&sum_d, nBytes));

    initData(a_h, numElement);
    initData(b_h, numElement);

    CHECK(cudaMemcpyAsync(a_d, a_h, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(b_d, b_h, nBytes, cudaMemcpyHostToDevice));

    dim3 block(numElement);
    dim3 grid(numElement / block.x);

    sumCuda<<<grid, block>>>(a_d, b_d, sum_d);

    CHECK(cudaMemcpyAsync(sum_h, sum_d, nBytes, cudaMemcpyDeviceToHost));

    for (int i=0; i<numElement; i++) {
        printf("%f\n", sum_h[i]);
    }

    cudaDeviceReset();

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(sum_d);

    free(a_h);
    free(b_h);
    free(sum_h);

    return 0;
}