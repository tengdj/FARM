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
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

RTree<Ideal *, double, 2, double> ideal_rtree;
RTree<MyPolygon *, double, 2, double> poly_rtree;

bool MySearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;

	Ideal *target= (Ideal *)ctx->target;

	if(ideal == target){
		return true;
	}
	// the minimum possible distance is larger than the threshold
	if(ideal->getMBB()->distance(*target->getMBB(), ctx->geography)>ctx->within_distance){
        return true;
	}
	// the maximum possible distance is smaller than the threshold
	if(ideal->getMBB()->max_distance(*target->getMBB(), ctx->geography)<=ctx->within_distance){
		ctx->found++;
        return true;
	}

	if (ideal->getMBB()->contain(*target->getMBB()) && ideal->contain(target->get_boundary()->p[0], ctx))
	{
		ctx->found++;
        return true;
	}
	if (target->getMBB()->contain(*ideal->getMBB()) && target->contain(ideal->get_boundary()->p[0], ctx))
	{
		ctx->found++;
        return true;
	}
        
	if (!ctx->use_gpu)
	{
		ctx->distance = ideal->distance(target,ctx);
		ctx->found += ctx->distance <= ctx->within_distance;
	}
#ifdef USE_GPU
	else{
		ctx->polygon_pairs.push_back(make_pair(ideal, target));
	}
#endif

	return true;
}

bool PolygonSearchCallback(MyPolygon *poly, void* arg){
	query_context *ctx = (query_context *)arg;

	MyPolygon *target= (MyPolygon *)ctx->target;
	// query with rasterization

	if(poly == target){
		return true;
	}
	// the minimum possible distance is larger than the threshold
	if(poly->getMBB()->distance(*target->getMBB(), ctx->geography)>ctx->within_distance){
        return true;
	}
	// the maximum possible distance is smaller than the threshold
	if(poly->getMBB()->max_distance(*target->getMBB(), ctx->geography)<=ctx->within_distance){
		ctx->found++;
        return true;
	}

	ctx->distance = poly->distance(target,ctx);
	ctx->found += ctx->distance <= ctx->within_distance;

	return true;
}

void *query(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started",ctx->thread_id);
	ctx->query_count = 0;
	double buffer_low[2];
	double buffer_high[2];

	while(ctx->next_batch(100)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			if(gctx->use_ideal){
				ctx->target = (void *)(gctx->source_ideals[i]);
				box qb = gctx->source_ideals[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
				ideal_rtree.Search(qb.low, qb.high, MySearchCallback, (void *)ctx);
			}else{
				ctx->target = (void *)(gctx->source_polygons[i]);
				box qb = gctx->source_polygons[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
				poly_rtree.Search(qb.low, qb.high, PolygonSearchCallback, (void *)ctx);
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
	global_ctx.query_type = QueryType::within;
    global_ctx.geography = true;

	if(global_ctx.use_ideal){
    	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
		global_ctx.source_ideals.resize(50000);
		timeval start = get_cur_time();
		for(Ideal *p : global_ctx.source_ideals){
			ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start, global_ctx.source_ideals.size());
		// the target is also the source
		global_ctx.target_num = global_ctx.source_ideals.size();
	}else{
    	global_ctx.source_polygons = load_polygons_from_path(global_ctx.source_path.c_str(), global_ctx);
		timeval start = get_cur_time();
		for(MyPolygon *p:global_ctx.source_polygons){
			poly_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start, global_ctx.source_polygons.size());
		// the target is also the source
		global_ctx.target_num = global_ctx.source_polygons.size();
	}

	preprocess(&global_ctx);

	// timeval start = get_cur_time();

    pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for(int i=0;i<global_ctx.num_threads;i++){
		ctx[i] = global_ctx; //query_context(global_ctx);
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
#ifdef USE_GPU
	preprocess_for_gpu(&global_ctx);
	timeval start = get_cur_time();
	global_ctx.found += cuda_within_polygon(&global_ctx);
#endif	
	cout << endl;
	global_ctx.print_stats();
	logt("total query",start);
	printf("Found: %d\n", global_ctx.found);
	return 0;
}



