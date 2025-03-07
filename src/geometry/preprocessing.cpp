#include "../include/Ideal.h"

void *rasterization_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;

	vector<Ideal *> &ideals = *(vector<Ideal *> *)gctx->target;

	// log("thread %d is started",ctx->thread_id);

	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			struct timeval start = get_cur_time();
			ideals[i]->use_hierachy = gctx->use_hierachy;
			if(gctx->use_hierachy) ideals[i]->grid_align(gctx);
			ideals[i]->rasterization(ctx->vpr);
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_rasterization(query_context *gctx){

	log("start rasterizing the referred polygons");
	vector<Ideal *> &ideals = *(vector<Ideal *> *)gctx->target;
	assert(ideals.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = ideals.size();

	struct timeval start = get_cur_time();
	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, rasterization_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void preprocess(query_context *gctx){
	vector<Ideal *> target_ideals;
	target_ideals.insert(target_ideals.end(), gctx->source_ideals.begin(), gctx->source_ideals.end());
	target_ideals.insert(target_ideals.end(), gctx->target_ideals.begin(), gctx->target_ideals.end());
	gctx->target = (void *)&target_ideals;

	if(gctx->use_hierachy){
		for(auto ideal : target_ideals){
			ideal->init_raster(ideal->get_boundary()->num_vertices / gctx->vpr);
			gctx->min_step_x = min(gctx->min_step_x, ideal->get_step_x());
			gctx->min_step_y = min(gctx->min_step_y, ideal->get_step_y());
			
			gctx->space.low[0] = min(gctx->space.low[0], ideal->getMBB()->low[0]);
			gctx->space.low[1] = min(gctx->space.low[1], ideal->getMBB()->low[1]);
			gctx->space.high[0] = max(gctx->space.high[0], ideal->getMBB()->high[0]);
			gctx->space.high[1] = max(gctx->space.high[1], ideal->getMBB()->high[1]);
		}

		assert(gctx->space.low[0] < gctx->space.high[0] && gctx->space.low[1] < gctx->space.high[1]);

		gctx->min_step_x = roundToSignificantDigits(gctx->min_step_x, 1);
		gctx->min_step_y = roundToSignificantDigits(gctx->min_step_y, 1);

		int dimx = (gctx->space.high[0] - gctx->space.low[0]) / gctx->min_step_x;
		int dimy = (gctx->space.high[1] - gctx->space.low[1]) / gctx->min_step_y;

		gctx->num_layers = static_cast<int>(ceil(max(log(dimx + 1) / log(2.0), log(dimy + 1) / log(2.0))));
		// printf("最低的一层：%d\n", num_floor);

		gctx->space.low[0] = gctx->min_step_x * floor(gctx->space.low[0] / gctx->min_step_x);
		gctx->space.low[1] = gctx->min_step_y * floor(gctx->space.low[1] / gctx->min_step_y);
		gctx->space.high[0] = gctx->min_step_x * ceil(gctx->space.high[0] / gctx->min_step_x);
		gctx->space.high[1] = gctx->min_step_y * ceil(gctx->space.high[1] / gctx->min_step_y);
	
		// printf("\nBOX: %lf %lf %lf %lf\n", gctx->space.low[0], gctx->space.low[1], gctx->space.high[0], gctx->space.high[1]);
		// printf("MIN STEP_X = %lf, MIN STEP_Y = %lf\n", gctx->min_step_x, gctx->min_step_y);
		// exit(0);
	}
	process_rasterization(gctx);

#ifdef USE_GPU
	cuda_create_buffer(gctx);		
#endif
	target_ideals.clear();

	gctx->target = NULL;
}