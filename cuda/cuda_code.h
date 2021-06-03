//
// Created by smallflyfly on 2021/5/17.
//

#include <sys/time.h>

#ifndef HELLO_WORLD_CUDA_CODE_H
#define HELLO_WORLD_CUDA_CODE_H

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double )tp.tv_sec + (double )tp.tv_sec * 1e-6;
}

void initDevice(int devNum) {
    int device = devNum;
    cudaDeviceProp cudaDeviceProp;
    CHECK(cudaGetDeviceProperties(&cudaDeviceProp, device));
    printf("Using device %d: %s\n", device, cudaDeviceProp.name);
    CHECK(cudaSetDevice(device));
}

void initData(float *a, int length) {
    for (int i = 0; i < length; i++) {
        a[i] = i * 1.0f;
    }
}

void initDataInt(int *a, int length) {
    for (int i = 0; i < length; i++) {
        a[i] = i % 10;
    }
}

void printMatrix(float *mat, int nx, int ny) {
    for (int i=0; i<ny; i++) {
        for (int j=0; j<nx; j++) {
            printf("%f ", mat[i * ny + nx]);
        }
        printf("\n");
    }
}

#endif //HELLO_WORLD_CUDA_CODE_H
