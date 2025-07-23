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
	global_ctx.target_num = global_ctx.source_ideals.size();

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

	auto gpu_start = std::chrono::high_resolution_clock::now();
	// // printf("%lf %lf\n", UniversalGrid::getInstance().get_step_x(), UniversalGrid::getInstance().get_step_y());
	// // for(auto p : global_ctx.source_ideals){
	// // 	if(p->id == 12787){
	// // 		printf("id = %d\n", p->id);
	// // 		printf("\ndimx = %d, dimy = %d\n", p->get_dimx(), p->get_dimy());
	// // 		p->MyPolygon::print();
	// // 		p->MyRaster::print();
	// // 		for(int i = 0; i < p->get_num_pixels(); i ++){
	// // 			if(i % p->get_dimx() == 0) printf("\n");
	// // 			printf("%d ", p->get_offset(i));
	// // 		}
	// // 	}
	// // }

	// // return 0;

	// // int a = global_ctx.h_candidate_pairs[0].first;
	// // int b = global_ctx.h_candidate_pairs[0].second - global_ctx.source_ideals.size();
	// // printf("polygon id = %d\n", global_ctx.source_ideals[a]->id);
	// // global_ctx.source_ideals[a]->MyPolygon::print();
	// // global_ctx.source_ideals[a]->MyRaster::print();
	
	// // for (int i = 0; i <= global_ctx.source_ideals[a]->get_num_layers(); i++)
	// // {
	// // 	printf("level %d:\n", i);
	// // 	printf("dimx=%d, dimy=%d, step_x = %lf, step_y = %lf\n", global_ctx.source_ideals[a]->get_layers()[i].get_dimx(), global_ctx.source_ideals[a]->get_layers()[i].get_dimy(), global_ctx.source_ideals[a]->get_layers()[i].get_step_x(), global_ctx.source_ideals[a]->get_layers()[i].get_step_y());
	// // 	global_ctx.source_ideals[a]->get_layers()[i].mbr->print();
	// // 	global_ctx.source_ideals[a]->get_layers()[i].print();
	// // }
	
	// // printf("polygon id = %d\n", global_ctx.target_ideals[b]->id);
	// // global_ctx.target_ideals[b]->MyPolygon::print();
	// // global_ctx.target_ideals[b]->MyRaster::print();

	// // for (int i = 0; i <= global_ctx.target_ideals[b]->get_num_layers(); i++)
	// // {
	// // 	printf("level %d:\n", i);
	// // 	printf("dimx=%d, dimy=%d, step_x = %lf, step_y = %lf\n", global_ctx.target_ideals[b]->get_layers()[i].get_dimx(), global_ctx.target_ideals[b]->get_layers()[i].get_dimy(), global_ctx.target_ideals[b]->get_layers()[i].get_step_x(), global_ctx.target_ideals[b]->get_layers()[i].get_step_y());
	// // 	global_ctx.target_ideals[b]->get_layers()[i].mbr->print();
	// // 	global_ctx.target_ideals[b]->get_layers()[i].print();
	// // }
	// global_ctx.batch_size = 1;
	printf("num pairs: %d\n", global_ctx.num_pairs);
	
	if (!global_ctx.batch_size) global_ctx.batch_size = global_ctx.num_pairs;
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);

		ResetDevice(&global_ctx);

		auto batch_start = std::chrono::high_resolution_clock::now();
		cuda_within_polygon(&global_ctx);
		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
		// return 0;
	}
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

	cout << endl;
	printf("Found: %d\n", global_ctx.found);
	return 0;
}



