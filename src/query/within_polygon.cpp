#include "../include/Ideal.h"
#include "../index/RTree.h"
#include <fstream>
#include <queue>
#include "UniversalGrid.h"

using namespace std;

RTree<Ideal *, double, 2, double> ideal_rtree;

bool MySearchCallback(Ideal *ideal, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Ideal *target = (Ideal *)ctx->target;
	if (ideal->id == target->id)
		return true;
	// two datasets
	// ctx->object_pairs.push_back(make_pair(ideal->id, target->id + gctx->source_ideals.size()));
	// self within
	ctx->object_pairs.push_back(make_pair(ideal->id, target->id));
	return true;
}

void *rtree_query(void *args)
{
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started", ctx->thread_id);
	ctx->query_count = 0;
	while (ctx->next_batch(10))
	{
		for (int i = ctx->index; i < ctx->index_end; i++)
		{
			// two datasets
			// Ideal *target = gctx->target_ideals[i];
			// ctx->target = (void *)target;
			// box qb = gctx->target_ideals[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
			
			// self within
			Ideal *target = gctx->source_ideals[i];
			ctx->target = (void *)target;
			box qb = gctx->source_ideals[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
			ideal_rtree.Search(qb.low, qb.high, MySearchCallback, (void *)ctx);
		}
	}
	ctx->merge_global();
	return NULL;
}

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within_polygon;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	for (Ideal *p : global_ctx.source_ideals)
	{
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	if (!global_ctx.target_path.empty())
	{
		global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
		global_ctx.target_num = global_ctx.target_ideals.size();
	}else{
		global_ctx.target_num = global_ctx.source_ideals.size();
	}

	timeval start = get_cur_time();
	pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		ctx[i] = query_context(global_ctx);
		ctx[i].thread_id = i;
	}
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		pthread_create(&threads[i], NULL, rtree_query, (void *)&ctx[i]);
	}
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("rtree query finished", start);

	global_ctx.index = 0;
	global_ctx.target_num = global_ctx.object_pairs.size();

	start = get_cur_time();	
	preprocess(&global_ctx);
	logt("preprocess finished", start);

	start = get_cur_time();
	preprocess_for_gpu(&global_ctx);
	logt("preprocess for gpu finished", start);
	
	if (!global_ctx.batch_size) global_ctx.batch_size = global_ctx.num_pairs;
	start = get_cur_time();
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);
		ResetDevice(&global_ctx);
		cuda_within_polygon(&global_ctx);
	}

	global_ctx.print_stats();
	logt("query finished", start);

	return 0;
}



