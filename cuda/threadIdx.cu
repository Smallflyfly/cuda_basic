//
// Created by smallflyfly on 2021/5/17.
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_code.h"

int main(int argc, char **argv) {
    initDevice(0);
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = sizeof(float) * nxy;

    // malloc
    float *a_h = (float*) malloc(nBytes);
    initData(a_h, nxy);
    printMatrix(a_h, nx, ny);

    return 0;
}