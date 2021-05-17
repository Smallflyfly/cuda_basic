//
// Created by smallflyfly on 2021/5/17.
//

#include <stdio.h>


__global__ void hello_world() {
    printf("GPU HELLO WORLD!\n");
}


int main(int argc, char** argv) {
    hello_world<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}