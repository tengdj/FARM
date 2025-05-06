/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#include "../include/Ideal.h"

using namespace std;

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	global_ctx.load_points();

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
		cuda_contain(&global_ctx, false);
		cuda_within(&global_ctx);
		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
	}

	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

	// size_t size = global_ctx.source_ideals.size() * global_ctx.target_num * 0.001;
	// global_ctx.h_candidate_pairs = new pair<uint32_t, uint32_t>[size];

	// auto rtree_insert_start = std::chrono::high_resolution_clock::now();
	// for(auto p : global_ctx.source_ideals){
	// 	rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	// }
	// auto rtree_insert_end = std::chrono::high_resolution_clock::now();
	// auto rtree_insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_insert_end - rtree_insert_start);
	// std::cout << "rtree insert: " << rtree_insert_duration.count() << " ms" << std::endl;

	// // auto rtree_query_start = std::chrono::high_resolution_clock::now();
	// pthread_t threads[global_ctx.num_threads];
	// query_context ctx[global_ctx.num_threads];
	// for (int i = 0; i < global_ctx.num_threads; i++){
	// 	ctx[i] = global_ctx;
	// 	ctx[i].thread_id = i;
	// 	ctx[i].global_ctx = &global_ctx;
	// 	ctx[i].h_candidate_pairs = new pair<uint32_t, uint32_t>[size / 128];
	// }

	// for (int i = 0; i < global_ctx.num_threads; i++){
	// 	pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
	// }

	// for (int i = 0; i < global_ctx.num_threads; i++){
	// 	void *status;
	// 	pthread_join(threads[i], &status);
	// }

	// for (int i = 0; i < global_ctx.num_threads; i++){
	// 	delete[] ctx[i].h_candidate_pairs;
	// }

	// global_ctx.index = 0;

	// printf("%d\n", global_ctx.num_pairs);
	// for(int i = 0; i < global_ctx.num_pairs; i ++){
	// 	printf("%d %d\n", global_ctx.h_candidate_pairs[i].first, global_ctx.h_candidate_pairs[i].second);
	// }

	// CopyCandidate(&global_ctx);

	// auto rtree_query_end = std::chrono::high_resolution_clock::now();
	// auto rtree_query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_query_end - rtree_query_start);
	// std::cout << "rtree query: " << rtree_query_duration.count() << " ms" << std::endl;
	// return 0;

	cout << endl;
	printf("FOUND: %d\n", global_ctx.found);
	return 0;
}


