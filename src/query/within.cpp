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

using namespace std;

RTree<Ideal *, double, 2, double> ideal_rtree;

bool IdealSearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;
	Point *p = (Point *)ctx->target;

	if(ideal->getMBB()->distance(*p, ctx->geography) > ctx->within_distance){
    	return true;
	}

	
	if(!ctx->use_gpu){
		ctx->distance = ideal->distance(*p,ctx);
		ctx->found += ctx->distance <= ctx->within_distance;
	}
#ifdef USE_GPU
	else{
		if(ideal->contain(*p, ctx)){
			ctx->found ++;
		}
		else{
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
	double buffer_low[2];
	double buffer_high[2];
	char point_buffer[200];
	while(ctx->next_batch(100)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			double shiftx = degree_per_kilometer_longitude(gctx->points[i].y)*gctx->within_distance;
			double shifty = degree_per_kilometer_latitude*gctx->within_distance;
			buffer_low[0] = gctx->points[i].x-shiftx;
			buffer_low[1] = gctx->points[i].y-shifty;
			buffer_high[0] = gctx->points[i].x+shiftx;
			buffer_high[1] = gctx->points[i].y+shifty;

			ctx->target = (void *)&gctx->points[i];
			ctx->target_id = i;
			ideal_rtree.Search(buffer_low, buffer_high, IdealSearchCallback, (void *)ctx);
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within;
	if(getFileExtension(global_ctx.source_path) == ".wkt"){
		global_ctx.source_ideals = load_polygon_wkt(global_ctx.source_path.c_str());
	}else{
		global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	}
	if(getFileExtension(global_ctx.target_path) == ".wkt"){
		global_ctx.points = load_point_wkt(global_ctx.target_path.c_str(), global_ctx.target_num, &global_ctx);
	}else{
		global_ctx.load_points();
	}
	
	timeval start = get_cur_time();
	for(auto p : global_ctx.source_ideals){
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	logt("building R-Tree with %d nodes", start, global_ctx.source_ideals.size());
	
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

	auto total_runtime_start = std::chrono::high_resolution_clock::now();
    pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for(int i=0;i<global_ctx.num_threads;i++){
		ctx[i] = global_ctx;
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
	printf("rtree output: %d\n", global_ctx.found);

#ifdef USE_GPU
	auto gpu_start = std::chrono::high_resolution_clock::now();
	global_ctx.found += cuda_within(&global_ctx);
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;
#endif
	cout << endl;
	printf("FOUND: %d\n", global_ctx.found);

	// cout << global_ctx.source_ideals[22]->true_mbr->distance(global_ctx.points[3941], ctx->geography) << endl;

	// printf("22\n");
	// int target_id = global_ctx.point_polygon_pairs[0].first;
	// int source_id = global_ctx.point_polygon_pairs[0].second;
	// global_ctx.points[target_id].print();
	// global_ctx.source_ideals[source_id]->getMBB()->print();
	// global_ctx.source_ideals[source_id]->MyPolygon::print();
	// global_ctx.source_ideals[source_id]->MyRaster::print();



	return 0;
}


