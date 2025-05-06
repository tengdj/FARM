#include "../include/Ideal.h"
#include <fstream>
#include "../index/RTree.h"
#include <queue>
#include <chrono>

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::contain_polygon;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
	global_ctx.target_num = global_ctx.target_ideals.size();

	if(!global_ctx.batch_size) global_ctx.batch_size = global_ctx.target_num;
	indexBuild(&global_ctx);

	auto rtree_query_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < global_ctx.target_num; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.target_num);
		indexQuery(&global_ctx);
	}
	auto rtree_query_end = std::chrono::high_resolution_clock::now();
	auto rtree_query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_query_end - rtree_query_start);
	std::cout << "rtree query: " << rtree_query_duration.count() << " ms" << std::endl;
	indexDestroy(&global_ctx);

	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

	auto preprocess_gpu_start = std::chrono::high_resolution_clock::now();
	preprocess_for_gpu(&global_ctx);
	auto preprocess_gpu_end = std::chrono::high_resolution_clock::now();
	auto preprocess_gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_gpu_end - preprocess_gpu_start);
	std::cout << "preprocess for gpu time: " << preprocess_gpu_duration.count() << " ms" << std::endl;

	auto gpu_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);

		ResetDevice(&global_ctx);

		auto batch_start = std::chrono::high_resolution_clock::now();
		cuda_contain(&global_ctx, true);
		cuda_contain_polygon(&global_ctx);
		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
	}
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

	cout << endl;
	printf("FOUND: %d\n", global_ctx.found);

	return 0;
}



