#include "../include/Ideal.h"
#include <fstream>
#include <queue>
#include <chrono>

RTree<Ideal *, double, 2, double> ideal_rtree;

bool MySearchCallback(Ideal *ideal, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Ideal *target = (Ideal *)ctx->target;
    if(ideal->getMBB()->intersect(*target->getMBB()))
	    ctx->object_pairs.push_back(make_pair(ideal->id, target->id + gctx->source_ideals.size()));

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

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::intersection;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
    for (Ideal *p : global_ctx.source_ideals)
	{
		ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
    global_ctx.target_num = global_ctx.target_ideals.size();

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

	auto preprocess_gpu_start = std::chrono::high_resolution_clock::now();
	preprocess_for_gpu(&global_ctx);
	auto preprocess_gpu_end = std::chrono::high_resolution_clock::now();
	auto preprocess_gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_gpu_end - preprocess_gpu_start);
	std::cout << "preprocess for gpu time: " << preprocess_gpu_duration.count() << " ms" << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();

	printf("num_pairs = %d\n", global_ctx.num_pairs);

    if(global_ctx.batch_size == 0) global_ctx.batch_size = global_ctx.num_pairs;
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
        auto batch_start = std::chrono::high_resolution_clock::now();
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);

		ResetDevice(&global_ctx);

	    auto gpu_start = std::chrono::high_resolution_clock::now();

		cuda_intersection(&global_ctx);

        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
    }

	auto total_end = std::chrono::high_resolution_clock::now();
	auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
	std::cout << "total query time: " << total_duration.count() << " ms" << std::endl;

    return 0;
}