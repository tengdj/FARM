/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */



#include "../index/RTree.h"
#include <queue>
#include <fstream>
#include "../include/Ideal.h"
#include <chrono>

int main(int argc, char** argv) {

	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::contain;
	
	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	if(!global_ctx.batch_size) global_ctx.batch_size = global_ctx.source_ideals.size();
	// read all the points
	global_ctx.load_points();

	// 如果显存空间不够了，就先传到cpu里
	// 或者加一个load_factor
	indexFilter(&global_ctx);

	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

#ifdef USE_GPU
	auto preprocess_gpu_start = std::chrono::high_resolution_clock::now();
	preprocess_for_gpu(&global_ctx);
	auto preprocess_gpu_end = std::chrono::high_resolution_clock::now();
	auto preprocess_gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_gpu_end - preprocess_gpu_start);
	std::cout << "preprocess for gpu time: " << preprocess_gpu_duration.count() << " ms" << std::endl;

	auto gpu_start = std::chrono::high_resolution_clock::now();
	cuda_contain(&global_ctx);
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;
#endif
	cout << endl;
	printf("FOUND: %d\n", global_ctx.h_result);
	// global_ctx.print_stats();

	return 0;
}



