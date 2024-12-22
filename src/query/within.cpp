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
RTree<MyPolygon *, double, 2, double> poly_rtree;

bool IdealSearchCallback(Ideal *ideal, void* arg){
	query_context *ctx = (query_context *)arg;
	Point *p = (Point *)ctx->target;
	if(ideal->getMBB()->distance(*p, ctx->geography)>ctx->within_distance){
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
		}else{
			// ctx->point_polygon_pairs.push_back(make_pair(p, ideal));
		}
	}
#endif
	return true;
}

bool PolygonSearchCallback(MyPolygon *poly, void* arg){
	query_context *ctx = (query_context *)arg;
	Point *p = (Point *)ctx->target;
	if(poly->getMBB()->distance(*p, ctx->geography)>ctx->within_distance){
        return true;
	}
	ctx->distance = poly->distance(*p,ctx);
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
			if(gctx->use_ideal){
				ideal_rtree.Search(buffer_low, buffer_high, IdealSearchCallback, (void *)ctx);
			}else{
				poly_rtree.Search(buffer_low, buffer_high, PolygonSearchCallback, (void *)ctx);
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

	if(global_ctx.use_ideal){
    	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
		timeval start = get_cur_time();
		for(auto p : global_ctx.source_ideals){
			ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start, global_ctx.source_ideals.size());
	}else{
		global_ctx.source_polygons = load_polygons_from_path(global_ctx.source_path.c_str(),global_ctx);
		timeval start = get_cur_time();
		for(auto p : global_ctx.source_polygons){
			poly_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
		}
		logt("building R-Tree with %d nodes", start, global_ctx.source_polygons.size());
	}

	preprocess(&global_ctx);

	// read all the points
	global_ctx.load_points();

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
#ifdef USE_GPU
	preprocess_for_gpu(&global_ctx);
	timeval start = get_cur_time();
	global_ctx.found += cuda_within(&global_ctx);
#endif
	cout << endl;
	printf("found: %d\n", global_ctx.found);
	logt("total query",start);

	return 0;
}


