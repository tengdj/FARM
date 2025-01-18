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
RTree<MyPolygon *, double, 2, double> poly_rtree;
int ct = 0;

bool MySearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;
	Ideal *target = (Ideal *)ctx->target;
	if(!ctx->use_gpu){
		ctx->found += ideal->contain(target, ctx);
	}
#ifdef USE_GPU
	else{
		if(ideal->getMBB()->contain(*target->getMBB())){
			ctx->polygon_pairs.push_back(make_pair(ideal, target));
		}
	}
#endif
	return true;
}

bool PolygonSearchCallback(MyPolygon *poly, void* arg){
	query_context *ctx = (query_context *)arg;
	MyPolygon *target = (MyPolygon *)ctx->target;
	ctx->found += poly->contain(target, ctx);
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
				box *bx = ideal->getMBB();
				ideal_rtree.Search(bx->low, bx->high, MySearchCallback, (void *)ctx);
			}else{
				MyPolygon *poly = gctx->target_polygons[i];
				ctx->target = (void *)poly;
				box *px = poly->getMBB();
				poly_rtree.Search(px->low, px->high, PolygonSearchCallback, (void *)ctx);
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
	global_ctx.query_type = QueryType::contain;

	if(global_ctx.use_ideal){
		global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
		timeval start = get_cur_time();
		for(auto p : global_ctx.source_ideals){
			ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start,global_ctx.source_ideals.size());

		global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
		global_ctx.target_num = global_ctx.target_ideals.size();

		global_ctx.reset_stats();
	}else{
		global_ctx.source_polygons = load_polygons_from_path(global_ctx.source_path.c_str(),global_ctx);
		timeval start = get_cur_time();
		for(auto p : global_ctx.source_polygons){
			poly_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start,global_ctx.source_polygons.size());

		global_ctx.target_polygons = load_polygons_from_path(global_ctx.target_path.c_str(),global_ctx);
		global_ctx.target_num = global_ctx.target_polygons.size();

		global_ctx.reset_stats();
	}

	preprocess(&global_ctx);

	pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];

	auto total_runtime_start = std::chrono::high_resolution_clock::now();
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
	preprocess_for_gpu(&global_ctx);
	timeval start = get_cur_time();
	global_ctx.found = cuda_contain_polygon(&global_ctx);
#endif
	cout << endl;
	global_ctx.print_stats();
	logt("query",start);
	return 0;
}



