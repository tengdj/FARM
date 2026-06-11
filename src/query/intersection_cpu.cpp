#include "../include/Ideal.h"
#include <algorithm>
#include <fstream>
#include <queue>

RTree<Ideal *, double, 2, double> rtree;

bool MySearchCallback(Ideal *ideal, void *arg)
{
	query_context *ctx = (query_context *)arg;
	query_context *gctx = ctx->global_ctx;

	Ideal *target = (Ideal *)ctx->target;
    if(ideal->getMBB()->intersect(*target->getMBB()))
	    ctx->object_pairs.push_back(make_pair(ideal->id, target->id));

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
			Ideal *target = gctx->target_ideals[i];
			ctx->target = (void *)target;
			box *bx = target->getMBB();
			rtree.Search(bx->low, bx->high, MySearchCallback, (void *)ctx);
		}
	}
	ctx->merge_object_pairs();
	return NULL;
}

void *query(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	log("thread %d is started",ctx->thread_id);
	
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			auto pair = gctx->object_pairs[i];
			auto sourceIdx = pair.first;
			auto targetIdx = pair.second;
			Ideal *source = gctx->source_ideals[sourceIdx];
			Ideal *target = gctx->target_ideals[targetIdx];
			assert(source->get_step_x() == source->get_step_y() && target->get_step_x() == target->get_step_y());
			if(source->get_step_x() < target->get_step_x()){
				swap(source, target);	
			}
			ctx->target_id = i;
			source->intersection(target, ctx);	
			ctx->report_progress();
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
		rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
	}
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
    global_ctx.target_num = global_ctx.target_ideals.size();
	global_ctx.object_pairs.resize(global_ctx.source_ideals.size() * 200);
	std::atomic<size_t> global_counter(0);
	global_ctx.global_write_index = &global_counter;

	timeval start = get_cur_time();
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

	logt("rtree filtering finished", start);

	global_ctx.index = 0;
	global_ctx.target_num = global_ctx.num_pairs; 
	// if(global_ctx.batch_size == 0) global_ctx.batch_size = global_ctx.num_pairs;

	start = get_cur_time();
	preprocess(&global_ctx);
	if(global_ctx.areas == nullptr){
		global_ctx.areas = new double[global_ctx.num_pairs];
	}
	std::fill(global_ctx.areas, global_ctx.areas + global_ctx.num_pairs, 0.0);
	logt("preprocess finished", start);

	// for(int i = 0; i < 10; i ++){
	// 	printf("object pair %d: source %d, target %d\n", i, global_ctx.object_pairs[i].first, global_ctx.object_pairs[i].second);
	// 	auto source = global_ctx.source_ideals[global_ctx.object_pairs[i].first];
	// 	auto target = global_ctx.target_ideals[global_ctx.object_pairs[i].second];
	// 	source->MyPolygon::print();
	// 	source->MyRaster::print();
	// 	target->MyPolygon::print();
	// 	target->MyRaster::print();
	// }

	start = get_cur_time();
	global_ctx.query_count = 0;
	global_ctx.previous = start;
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

	for(size_t i = 0; i < global_ctx.num_pairs; i++){
		printf("area = %lf\n", 0.5 * global_ctx.areas[i]);
	}

	global_ctx.print_stats();
	logt("query finished", start);
	return 0;
}
