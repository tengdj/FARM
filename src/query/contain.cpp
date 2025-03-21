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

// some shared parameters

RTree<Ideal *, double, 2, double> ideal_rtree;

bool IdealSearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;
	if(!ctx->use_gpu){
		ctx->found += ideal->contain(*(Point *)ctx->target, ctx);
	}
#ifdef USE_GPU
	else{
		if(ideal->getMBB()->contain(*(Point *)ctx->target)){
			ctx->point_polygon_pairs.emplace_back(make_pair(ctx->target_id, ideal->id));
		}
	}
#endif
	return true;
}

void *query(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started",ctx->thread_id);
	ctx->query_count = 0;
	// ctx->point_polygon_pairs = new pair<Point*, Ideal*>[gctx->target_num];

	while(ctx->next_batch(100)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			ctx->target = (void *)&gctx->points[i];
			ctx->target_id = i;
			ideal_rtree.Search((double *)(gctx->points+i), (double *)(gctx->points+i), IdealSearchCallback, (void *)ctx);
			ctx->report_progress();
		}
	}

	ctx->merge_global();

	return NULL;
}



int main(int argc, char** argv) {

	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::contain;
	

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	if(!global_ctx.batch_size) global_ctx.batch_size = global_ctx.source_ideals.size();
	// read all the points
	global_ctx.load_points();

	// 如果显存空间不够了，就先传到cpu里
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
#endif

	// auto total_runtime_start = std::chrono::high_resolution_clock::now();
	// pthread_t threads[global_ctx.num_threads];
	// query_context ctx[global_ctx.num_threads];
	// for(int i=0;i<global_ctx.num_threads;i++){
	// 	ctx[i] = global_ctx;
	// 	ctx[i].thread_id = i;
	// 	ctx[i].global_ctx = &global_ctx;
	// }
	// for(int i=0;i<global_ctx.num_threads;i++){
	// 	pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
	// }

	// for(int i = 0; i < global_ctx.num_threads; i++ ){
	// 	void *status;
	// 	pthread_join(threads[i], &status);
	// }

	// auto total_runtime_end = std::chrono::high_resolution_clock::now();
	// auto total_runtime_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_runtime_end - total_runtime_start);
	// std::cout << "rtree query: " << total_runtime_duration.count() << " ms" << std::endl;

#ifdef USE_GPU
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



