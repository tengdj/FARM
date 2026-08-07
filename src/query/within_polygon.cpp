#include "../include/farm.h"
#include "../index/RTree.h"

RTree<Farm *, double, 2, double> rtree;

namespace {
constexpr size_t DEFAULT_GPU_BATCH_SIZE = 1000000;
}

bool MySearchCallback(Farm *source, void *arg)
{
	query_context *ctx = static_cast<query_context *>(arg);
	query_context *gctx = ctx->global_ctx;

	Farm *target = static_cast<Farm *>(ctx->target);
	bool self_join = gctx->target_path.empty();
	if (self_join && source->id == target->id)
		return true;

	uint32_t target_id = target->id;
	if (!self_join)
		target_id += static_cast<uint32_t>(gctx->source_objects.size());
	ctx->object_pairs.emplace_back(source->id, target_id);
	return true;
}

void *rtree_query(void *args)
{
	query_context *ctx = static_cast<query_context *>(args);
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started", ctx->thread_id);
	bool self_join = gctx->target_path.empty();
	vector<Farm *> &targets = self_join ? gctx->source_objects : gctx->target_objects;
	while (ctx->next_batch(10))
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

int main(int argc, char** argv) {
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
	}else{
		global_ctx.target_objects = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
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
	preprocess_for_gpu(&global_ctx);
	logt("preprocessing for gpu finished", start);
	
	start = get_cur_time();
	if(global_ctx.batch_size == 0){
		global_ctx.batch_size = std::min(global_ctx.num_pairs, DEFAULT_GPU_BATCH_SIZE);
	}

	for(size_t i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = std::min(i + global_ctx.batch_size, global_ctx.num_pairs);
		cuda_within_polygon(&global_ctx);
	}

	global_ctx.print_stats();
	logt("query finished", start);

	return 0;
}
