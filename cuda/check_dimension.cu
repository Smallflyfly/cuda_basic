//
// Created by smallflyfly on 2021/5/17.
//

#include <stdio.h>
#include <cuda_runtime.h>


__global__ void checkIndex() {
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d %d %d) gridDim(%d %d %d)\n", threadIdx.x, threadIdx.y,
           threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int agrc, char **argv) {
    int nElement = 6;
    dim3 block(4);
    dim3 grid((nElement+block.x - 1) / block.x);

    printf("grid.x %d, grid.y %d, grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d, block.y %d, block.z %d\n", block.x, block.y, block.z);

    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

}