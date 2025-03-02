/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#include "../include/Ideal.h"
#include <fstream>
#include "../index/RTree.h"
#include <queue>
#include <chrono>

RTree<Ideal *, double, 2, double> ideal_rtree;

bool IdealSearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;
	Ideal *target = (Ideal *)ctx->target;
	if(!ctx->use_gpu){
		ctx->found += ideal->contain(target, ctx);
	}
#ifdef USE_GPU
	else{
		if(ideal->getMBB()->contain(*target->getMBB())){
			ctx->polygon_pairs.emplace_back(make_pair(ctx->target_id, ideal->id));
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

	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			if(gctx->use_ideal){
				Ideal *ideal = gctx->target_ideals[i];
				ctx->target = (void *)ideal;
				ctx->target_id = ideal->id;
				box *bx = ideal->getMBB();
				ideal_rtree.Search(bx->low, bx->high, IdealSearchCallback, (void *)ctx);
			}
			ctx->report_progress();
		}
	}

	ctx->merge_global();

	return NULL;
}



int main(int argc, char** argv) {

	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::contain_polygon;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
	global_ctx.target_num = global_ctx.target_ideals.size();

	timeval start = get_cur_time();
	for(auto p : global_ctx.source_ideals){
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	logt("building R-Tree with %d nodes", start,global_ctx.source_ideals.size());

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

	global_ctx.num_threads = 1;
	auto total_runtime_start = std::chrono::high_resolution_clock::now();
	pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for(int i=0;i<global_ctx.num_threads;i++){
		ctx[i] = query_context(global_ctx);
		ctx[i].thread_id = i;
		ctx[i].global_ctx = &global_ctx;
	}
	for(int i=0;i<global_ctx.num_threads;i++){
		pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
	}
	for(int i = 0; i < global_ctx.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	auto total_runtime_end = std::chrono::high_resolution_clock::now();
	auto total_runtime_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_runtime_end - total_runtime_start);
	std::cout << "rtree query: " << total_runtime_duration.count() << " ms" << std::endl;

#ifdef USE_GPU
	auto gpu_start = std::chrono::high_resolution_clock::now();
	global_ctx.found = cuda_contain_polygon(&global_ctx);
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;
#endif
	cout << endl;
	printf("FOUND: %d\n", global_ctx.found);

	return 0;
}



