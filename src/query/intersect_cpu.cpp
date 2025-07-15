#include "../include/Ideal.h"
#include <fstream>
#include <queue>

void *query(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started",ctx->thread_id);
	ctx->query_count = 0;
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
            auto pair = gctx->h_candidate_pairs[i];
            auto sourceIdx = pair.first;
            auto targetIdx = pair.second - gctx->source_ideals.size();
            // printf("%d\t%d\n", sourceIdx, targetIdx);
            Ideal *source = gctx->source_ideals[sourceIdx];
            Ideal *target = gctx->target_ideals[targetIdx];
            // if(ctx->thread_id == 0){
            // 	source->MyPolygon::print();
            // 	target->MyPolygon::print();
            // }
            ctx->found += source->intersect(target, ctx);
			ctx->report_progress();
		}
	}
	// if(ctx->thread_id == 0){
	// 	printf("--------------------------------\n");
	// 	for(auto p : ctx->intersection_polygons){
	// 		p->MyPolygon::print();
	// 	}
	// }
	ctx->merge_global();
	return NULL;
}



int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::intersect;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
    global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
    global_ctx.target_num = global_ctx.target_ideals.size();

    if(!global_ctx.batch_size) global_ctx.batch_size = global_ctx.target_num;

    indexBuild(&global_ctx);

    auto rtree_query_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < global_ctx.target_num; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.target_num);
		indexQuery(&global_ctx);
	}
	auto rtree_query_end = std::chrono::high_resolution_clock::now();
	auto rtree_query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_query_end - rtree_query_start);
	std::cout << "rtree query: " << rtree_query_duration.count() << " ms" << std::endl;
	indexDestroy(&global_ctx);

	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

	timeval start = get_cur_time();
	pthread_t threads[global_ctx.num_threads];
	query_context ctx[global_ctx.num_threads];
	for(int i=0;i<global_ctx.num_threads;i++){
		ctx[i] = query_context(global_ctx);
		ctx[i].thread_id = i;
	}
	for(int i=0;i<global_ctx.num_threads;i++){
		pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
	}
	for(int i = 0; i < global_ctx.num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	printf("FOUND: %d\n", global_ctx.found);
	// printf("intersec duration = %lf\n", global_ctx.test_duration);

	cout << endl;
	global_ctx.print_stats();
	logt("query",start);
	return 0;
}