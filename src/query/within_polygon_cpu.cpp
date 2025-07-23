#include "../include/Ideal.h"
#include "../index/RTree.h"
#include <fstream>
#include <queue>

RTree<Ideal *, double, 2, double> ideal_rtree;

bool MySearchCallback(Ideal *ideal, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Ideal *target = (Ideal *)ctx->target;
	if (ideal->id == target->id)
		return true;
	// if(ideal->getMBB()->intersect(*target->getMBB()))
	// 	return true;
	// if(ideal->getMBB()->contain(*target->getMBB()))
	// 	return true;
	// if(target->getMBB()->contain(*ideal->getMBB()))
	// 	return true;
	ctx->object_pairs.push_back(make_pair(ideal->id, target->id));
	return true;
}

void *rtree_query(void *args)
{
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("rtree thread %d is started", ctx->thread_id);
	ctx->query_count = 0;
	while (ctx->next_batch(10))
	{
		for (int i = ctx->index; i < ctx->index_end; i++)
		{
			Ideal *target = gctx->source_ideals[i];
			ctx->target = (void *)target;
			box qb = gctx->source_ideals[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
			ideal_rtree.Search(qb.low, qb.high, MySearchCallback, (void *)ctx);
		}
	}
	ctx->merge_global();
	return NULL;
}

void *query(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started",ctx->thread_id);
	ctx->query_count = 0;
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
            auto pair = gctx->object_pairs[i];
            auto sourceIdx = pair.first;
            auto targetIdx = pair.second;
            Ideal *source = gctx->source_ideals[sourceIdx];
            Ideal *target = gctx->source_ideals[targetIdx];
            ctx->found += source->within(target, ctx);
			ctx->report_progress();
		}
	}
	// ctx->merge_global();
	gctx->lock();
	gctx->found += ctx->found;
	gctx->unlock();
	return NULL;
}



int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within_polygon;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
	for (Ideal *p : global_ctx.source_ideals)
	{
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	global_ctx.target_num = global_ctx.source_ideals.size();

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

	global_ctx.index = 0;
	global_ctx.target_num = global_ctx.object_pairs.size();

	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

	// global_ctx.num_threads = 1;
	timeval start = get_cur_time();
	pthread_t threads2[global_ctx.num_threads];
	query_context ctx2[global_ctx.num_threads];
	for(int i=0;i<global_ctx.num_threads;i++){
		ctx2[i] = query_context();
		ctx2[i].global_ctx = &global_ctx;
		ctx2[i].thread_id = i;
	}
	for(int i=0;i<global_ctx.num_threads;i++){
		pthread_create(&threads2[i], NULL, query, (void *)&ctx2[i]);
	}
	for(int i = 0; i < global_ctx.num_threads; i++ ){
		void *status;
		pthread_join(threads2[i], &status);
	}

	printf("FOUND: %d\n", global_ctx.found);

	cout << endl;
	global_ctx.print_stats();
	logt("query",start);
	return 0;
}