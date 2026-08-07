#include "../include/farm.h"

RTree<Farm *, double, 2, double> rtree;

bool MySearchCallback(Farm *obj, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Farm *target = (Farm *)ctx->target;
	ctx->object_pairs.push_back(make_pair(obj->id, target->id + gctx->source_objects.size()));

	return true;
}

void *rtree_query(void *args)
{
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("rtree thread %d is started", ctx->thread_id);
	
	while (ctx->next_batch(10))
	{
		for (int i = ctx->index; i < ctx->index_end; i++)
		{
			Farm *target = gctx->target_objects[i];
			ctx->target = (void *)target;
			box *bx = target->getMBB();
			rtree.Search(bx->low, bx->high, MySearchCallback, (void *)ctx);
		}
	}

	ctx->merge_object_pairs();
	return NULL;
}

int main(int argc, char** argv) {
	query_context global_ctx;
	get_parameters(argc, argv, global_ctx);
	global_ctx.query_type = QueryType::intersect;

    global_ctx.source_objects = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	for (Farm *p : global_ctx.source_objects)
	{
		rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	global_ctx.target_objects = load_binary_file(global_ctx.target_path.c_str(), global_ctx);
	global_ctx.target_num = global_ctx.target_objects.size();

	timeval start = get_cur_time();
	global_ctx.run_worker_threads(rtree_query);
	
	logt("rtree filtering finished", start);

	log("total %zu pairs found", global_ctx.num_pairs);
	if(global_ctx.num_pairs == 0){
		global_ctx.print_stats();
		logt("query finished", start);
		return 0;
	}

	start = get_cur_time();
	preprocess(&global_ctx);
	logt("preprocess finished", start);

	start = get_cur_time();
	preprocess_for_gpu(&global_ctx);
	logt("preprocess for gpu finished", start);

	start = get_cur_time();
	if(global_ctx.batch_size == 0) global_ctx.batch_size = global_ctx.num_pairs;
	for(size_t i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);
		cuda_intersect(&global_ctx);
	}

	global_ctx.print_stats();
	logt("query finished", start);

	return 0;
}
