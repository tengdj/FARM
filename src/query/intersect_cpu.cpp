#include "../include/Ideal.h"
#include <fstream>
#include <queue>

RTree<Ideal *, double, 2, double> ideal_rtree;

bool MySearchCallback(Ideal *ideal, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Ideal *target = (Ideal *)ctx->target;
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
			Ideal *target = gctx->target_ideals[i];
			ctx->target = (void *)target;
			box *bx = target->getMBB();
			ideal_rtree.Search(bx->low, bx->high, MySearchCallback, (void *)ctx);
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
            Ideal *target = gctx->target_ideals[targetIdx];
            ctx->found += source->intersect(target, ctx);
			ctx->report_progress();
		}
	}

	gctx->lock();
	gctx->found += ctx->found;
	gctx->raster_filter_time += ctx->raster_filter_time;
	gctx->refine_time += ctx->refine_time;
	gctx->unlock();
	return NULL;
}

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::intersect;
	global_ctx.num_threads = 1;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
    for (Ideal *p : global_ctx.source_ideals)
	{
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
    global_ctx.target_num = global_ctx.target_ideals.size();

	auto rtree_start = std::chrono::high_resolution_clock::now();
	pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for (int i = 0; i < global_ctx.num_threads; i++)
	{
		ctx[i] = query_context();
		ctx[i].global_ctx = &global_ctx;
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

	auto rtree_end = std::chrono::high_resolution_clock::now();
	auto rtree_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_end - rtree_start);
	std::cout << "rtree filter time: " << rtree_duration.count() << " ms" << std::endl;

	global_ctx.index = 0;
	global_ctx.target_num = global_ctx.object_pairs.size();    
	global_ctx.num_threads = 128;
	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;
 
	printf("%d\n", global_ctx.object_pairs.size());
	global_ctx.num_threads = 1;
	auto gpu_start = std::chrono::high_resolution_clock::now();
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

	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "gpu time: " << gpu_duration.count() << " ms" << std::endl;

	printf("FOUND: %d\n", global_ctx.found);
	printf("rater filter time: %lf ms\n", global_ctx.raster_filter_time);
	printf("refine time: %lf ms\n", global_ctx.refine_time);
	// printf("intersec duration = %lf\n", global_ctx.test_duration);

	cout << endl;
	return 0;
}