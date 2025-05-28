#include "../include/Ideal.h"
#include "../include/query_context.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

__global__ void computeFloatDistance(
    const float* floatA, int n,
    const float* floatB, int m,
    float* outDistances)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxA < n && idxB < m) {
        float a = floatA[idxA];
        float b = floatB[idxB];

		float dist = a * a + b * b;

        int index = idxA * m + idxB;
        outDistances[index] = dist;
    }
}

int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(), global_ctx);
	int n = global_ctx.source_ideals[0]->get_num_vertices(), m = global_ctx.target_ideals[0]->get_num_vertices();

	float* h_floatA = new float[n];
	float* h_floatB = new float[m];

	for(int i = 0; i < n; i ++){
		h_floatA[i] = global_ctx.source_ideals[0]->get_boundary()->p[i].x;
	}

	for(int i = 0; i < m; i ++){
		h_floatB[i] = global_ctx.target_ideals[0]->get_boundary()->p[i].x;
	}

	float* d_floatA;
    float* d_floatB;
    float* d_distances;

    int total = n * m;
    cudaMalloc((void **)&d_floatA, n * sizeof(float));
    cudaMalloc((void **)&d_floatB, m * sizeof(float));
    cudaMalloc(&d_distances, total * sizeof(float));

    cudaMemcpy(d_floatA, h_floatA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_floatB, h_floatB, m * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (m + 15) / 16);

	cudaEventRecord(start);

	computeFloatDistance<<<gridSize, blockSize>>>(d_floatA, n, d_floatB, m, d_distances);
    cudaDeviceSynchronize();

	// unsigned long long sum = 0;
	// int size = global_ctx.source_ideals.size();
	// for(auto p : global_ctx.source_ideals){
	// 	sum += p->get_num_vertices();
	// 	// if(p->get_num_vertices() > 1000){
	// 		// p->MyPolygon::print();
	// 	// }
	// }
	// printf("%lu\n", sum);
	// preprocess(&global_ctx);
	// cout << "rasterization finished!" << endl;

	// // read all the points
	// global_ctx.load_points();

	return 0;
}