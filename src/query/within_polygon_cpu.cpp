#include "../include/farm.h"
#include "../index/RTree.h"

RTree<Farm *, double, 2, double> rtree;
static constexpr int WORKER_BATCH_SIZE = 10;

bool MySearchCallback(Farm *obj, void *arg)
{
	query_context *ctx = static_cast<query_context *>(arg);
	query_context *gctx = ctx->global_ctx;

	Farm *target = static_cast<Farm *>(ctx->target);
	if (gctx->target_path.empty() && obj->id == target->id)
		return true;
	ctx->object_pairs.emplace_back(obj->id, target->id);
	return true;
}

void *rtree_query(void *args)
{
	query_context *ctx = static_cast<query_context *>(args);
	query_context *gctx = ctx->global_ctx;
	log("rtree thread %d is started", ctx->thread_id);
	vector<Farm *> &targets =
		gctx->target_path.empty() ? gctx->source_objects : gctx->target_objects;

	while (ctx->next_batch(WORKER_BATCH_SIZE))
	{
		for (size_t i = ctx->index; i < ctx->index_end; i++)
		{
			Farm *target = targets[i];
			box qb = target->getMBB()->expand(gctx->within_distance, ctx->geography);
			ctx->target = target;
			rtree.Search(qb.low, qb.high, MySearchCallback, ctx);
		}
	}
	ctx->merge_object_pairs();
	return nullptr;
}

void *query(void *args)
{
	query_context *ctx = static_cast<query_context *>(args);
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started", ctx->thread_id);
	vector<Farm *> &targets =
		gctx->target_path.empty() ? gctx->source_objects : gctx->target_objects;

	while (ctx->next_batch(WORKER_BATCH_SIZE))
	{
		for (size_t i = ctx->index; i < ctx->index_end; i++)
		{
			auto pair = gctx->object_pairs[i];
			auto sourceIdx = pair.first;
			auto targetIdx = pair.second;
			Farm *source = gctx->source_objects[sourceIdx];
			Farm *target = targets[targetIdx];
			ctx->found += source->within(target, ctx, gctx->use_approximation);
			ctx->report_progress();
		}
	}

	ctx->merge_global();
	return nullptr;
}

int main(int argc, char **argv)
{
	query_context global_ctx;
	get_parameters(argc, argv, global_ctx);
	global_ctx.query_type = QueryType::within;

	global_ctx.source_objects = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	for (Farm *p : global_ctx.source_objects)
	{
		rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}

	if (global_ctx.target_path.empty())
	{
		global_ctx.target_num = global_ctx.source_objects.size();
	}
	else
	{
		global_ctx.target_objects = load_binary_file(global_ctx.target_path.c_str(), global_ctx);
		global_ctx.target_num = global_ctx.target_objects.size();
	}

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
	logt("preprocessing finished", start);

	start = get_cur_time();

	global_ctx.index = 0;
	global_ctx.target_num = global_ctx.num_pairs;

	global_ctx.run_worker_threads(query);

	global_ctx.print_stats();
	logt("query finished", start);

	return 0;
}
