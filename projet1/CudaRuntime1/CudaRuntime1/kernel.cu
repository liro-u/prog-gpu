#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define error trigger
#define TRIGGER_ERROR true  // Set to true to trigger errors

// Error handling macro
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char* file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
        exit(1);
    }
}

// CUDA Kernel for Hello World (1 + 1 addition)
__global__ void add_hello(float* a, float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// CUDA Kernel for vector addition with N blocks
__global__ void add1(float* a, float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// CUDA Kernel for vector addition with N threads in 1 block
__global__ void add2(float* a, float* b, float* c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

// CUDA Kernel for vector addition using threads and blocks
__global__ void add3(float* a, float* b, float* c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

// CUDA Kernel for vector addition with bounds checking
__global__ void add4(float* a, float* b, float* c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

// Function to compare two arrays (Root Mean Square Deviation)
float comparef(float* a, float* b, int n) {
    double diff = 0.0;
    for (int i = 0; i < n; i++)
        diff += (a[i] - b[i]) * (a[i] - b[i]);
    return (float)(sqrt(diff / (double)n));
}

// CPU version of vector addition
void cpu_add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Hello World function: Adds 1 + 1
void hello_world(void) {
    float* a, * b, * c, * d_a, * d_b, * d_c;
    int size = 1 * sizeof(float);

    // Initialize variables
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    *a = 1.0f;
    *b = 1.0f;

    // Allocate GPU memory with error checking
    checkCudaErrors(cudaMalloc((void**)&d_a, size));
    checkCudaErrors(cudaMalloc((void**)&d_b, size));
    checkCudaErrors(cudaMalloc((void**)&d_c, size));

    // Copy to device
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Run the kernel
    add_hello << < 1, 1 >> > (d_a, d_b, d_c);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result (1.0 + 1.0)
    printf("Hello World Result (1 + 1): %f\n", *c);

    // Free memory
    free(a); free(b); free(c);
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

// First vector addition main: N blocks, 1 thread per block
void main1(void) {
    const int N = 512;
    float* a, * b, * c, * d_a, * d_b, * d_c, * cpu_c;
    int size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    cpu_c = (float*)malloc(size);  // for CPU result

    // Generate random vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // CPU vector addition (golden model)
    cpu_add(a, b, cpu_c, N);

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_a, size));
    checkCudaErrors(cudaMalloc((void**)&d_b, size));
    checkCudaErrors(cudaMalloc((void**)&d_c, size));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Trigger error if defined
    if (TRIGGER_ERROR) {
        checkCudaErrors(cudaMalloc((void**)&d_a, size * 1000000)); // Intentionally allocate a large amount of memory
    }

    // Run the kernel
    add1 << < N, 1 >> > (d_a, d_b, d_c);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result
    printf("Result from main1: First element of result = %f\n", c[0]);

    // Compare with CPU result
    float rms_error = comparef(c, cpu_c, N);
    printf("RMS Error with CPU result: %f\n", rms_error);

    // Free memory
    free(a); free(b); free(c); free(cpu_c);
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

// Second vector addition main: 1 block, N threads
void main2(void) {
    const int N = 200000;  // Set an overly large number of threads for a block
    float* a, * b, * c, * d_a, * d_b, * d_c, * cpu_c;
    int size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    cpu_c = (float*)malloc(size);  // for CPU result

    // Generate random vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory with error checking
    checkCudaErrors(cudaMalloc((void**)&d_a, size));
    checkCudaErrors(cudaMalloc((void**)&d_b, size));
    checkCudaErrors(cudaMalloc((void**)&d_c, size));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Trigger error if defined
    if (TRIGGER_ERROR) {
        add2 << <1, N + 1 >> > (d_a, d_b, d_c); // Intentionally launch too many threads
    }
    else {
        // Run the kernel with an excessive number of threads
        add2 << <1, N >> > (d_a, d_b, d_c);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result
    printf("Result from main2: First element of result = %f\n", c[0]);

    // Free memory
    free(a); free(b); free(c); free(cpu_c);
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

// Third vector addition main: N * M threads
void main3(void) {
    const int N = 1024 * 1024;
    const int M = 512;
    float* a, * b, * c, * d_a, * d_b, * d_c;
    int size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Generate random vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_a, size));
    checkCudaErrors(cudaMalloc((void**)&d_b, size));
    checkCudaErrors(cudaMalloc((void**)&d_c, size));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Trigger error if defined
    if (TRIGGER_ERROR) {
        add3 << <M, N >> > (d_a, d_b, d_c); // Launch with too many threads
    }
    else {
        // Run the kernel
        add3 << <M, N >> > (d_a, d_b, d_c);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result
    printf("Result from main3: First element of result = %f\n", c[0]);

    // Free memory
    free(a); free(b); free(c);
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

// Fourth vector addition main: bounds checking
void main4(void) {
    const int N = 200000;
    float* a, * b, * c, * d_a, * d_b, * d_c;
    int size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Generate random vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_a, size));
    checkCudaErrors(cudaMalloc((void**)&d_b, size));
    checkCudaErrors(cudaMalloc((void**)&d_c, size));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Trigger error if defined
    if (TRIGGER_ERROR) {
        add4 << <(N + 255) / 256, 256 >> > (d_a, d_b, d_c, N + 1); // Intentionally pass wrong size
    }
    else {
        add4 << <(N + 255) / 256, 256 >> > (d_a, d_b, d_c, N);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Print result
    printf("Result from main4: First element of result = %f\n", c[0]);

    // Free memory
    free(a); free(b); free(c);
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

// 2D-Indexed kernel
__global__ void print_thread_info() {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    printf("Block: (%d, %d), Thread: (%d, %d)\n", x, y, tx, ty);
}

// 2D kernel function to launch the print_thread_info kernel
void main_2D_kernel() {
    dim3 blocks(2, 2);   // 2x2 grid of blocks
    dim3 threads(4, 4);  // 4x4 threads per block

    print_thread_info << <blocks, threads >> > ();
    checkCudaErrors(cudaDeviceSynchronize());
}

// Main program to choose which function to run
int main(void) {
    int choice;

    printf("Choose which program to run:\n");
    printf("1: Hello World (1 + 1)\n");
    printf("2: Vector addition (N blocks, 1 thread per block)\n");
    printf("3: Vector addition (1 block, N threads)\n");
    printf("4: Vector addition (N * M threads)\n");
    printf("5: Vector addition with bounds checking\n");
    printf("6: 2D-Indexed Kernel\n");  // New option
    scanf("%d", &choice);

    switch (choice) {
    case 1:
        hello_world();
        break;
    case 2:
        main1();
        break;
    case 3:
        main2();
        break;
    case 4:
        main3();
        break;
    case 5:
        main4();
        break;
    case 6:
        main_2D_kernel();  // Call to 2D kernel
        break;
    default:
        printf("Invalid choice!\n");
    }

    return 0;
}