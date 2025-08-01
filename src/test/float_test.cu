#include "../include/Ideal.h"
#include "../include/query_context.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <chrono>

__global__ void computeFloatDistance(
    const float* floatA, int n,
    const float* floatB, int m,
    float* outDistances)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxA < n && idxB < m) {
        int a = floatA[idxA];
        float b = floatB[idxB];

		float dist = a * a + b * b;

        int index = idxA * m + idxB;
        outDistances[index] = dist;
    }
}

__global__ void computeIntDistance(
    const int* intA, int n,
    const int* intB, int m,
    int* outDistances)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxA < n && idxB < m) {
        int a = intA[idxA];
        int b = intB[idxB];

		int dist = a * a + b * b;

        int index = idxA * m + idxB;
        outDistances[index] = dist;
    }
}

__global__ void warmup_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 做点无意义的计算
    for (int i = 0; i < 100000; ++i)
        idx *= i;
}

int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	int n = 0;
	for(int i = 0; i < global_ctx.source_ideals.size(); i ++){
		n += global_ctx.source_ideals[i]->get_num_vertices();
	}

	printf("n = %d\n", n);

	int total = n * n;

    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (n + 15) / 16);

	int* h_intA = new int[n];
	int* h_intB = new int[n];
    float* h_floatA = new float[n];
    float* h_floatB = new float[n];

    int id = 0;
	int id2 = 0;
	for(int i = 0; i < global_ctx.source_ideals.size(); i ++){
		auto polygon = global_ctx.source_ideals[i];
		for(int j = 0; j < polygon->get_num_vertices(); j ++){
            h_floatA[id] = (float)polygon->get_boundary()->p[j].x;
            h_floatB[id ++] = (float)polygon->get_boundary()->p[j].y;
            h_intA[id2] = (int)polygon->get_boundary()->p[j].x;
			h_intB[id2 ++] = (int)polygon->get_boundary()->p[j].y;
		}
	}
    
    float* d_floatA;
    float* d_floatB;
    float* d_distances;
	int* d_intA;
    int* d_intB;
    int* d_int_distances;

    cudaMalloc((void **)&d_floatA, n * sizeof(float));
    cudaMalloc((void **)&d_floatB, n * sizeof(float));
    cudaMalloc(&d_distances, total * sizeof(float));
    cudaMalloc((void **)&d_intA, n * sizeof(int));
    cudaMalloc((void **)&d_intB, n * sizeof(int));
    cudaMalloc(&d_int_distances, total * sizeof(int));

    cudaMemcpy(d_floatA, h_floatA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_floatB, h_floatB, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intA, h_intA, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_intB, h_intB, n * sizeof(int), cudaMemcpyHostToDevice);

    warmup_kernel<<<32, 256>>>();
    cudaDeviceSynchronize(); // 确保预热完成

    for(int i = 0; i < 10 ;i ++){
        auto start = std::chrono::high_resolution_clock::now();

        computeFloatDistance<<<gridSize, blockSize>>>(d_floatA, n, d_floatB, n, d_distances);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "float GPU kernel execution time: " << elapsed.count() << " ms\n";

        auto start2 = std::chrono::high_resolution_clock::now();

        computeIntDistance<<<gridSize, blockSize>>>(d_intA, n, d_intB, n, d_int_distances);
        cudaDeviceSynchronize();

	
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;

        std::cout << "int GPU kernel execution time: " << elapsed2.count() << " ms\n";
    }


	return 0;
}