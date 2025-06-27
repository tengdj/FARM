#include "../include/Ideal.h"
#include "UniversalGrid.h"

void *rasterization_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;

	vector<Ideal *> &ideals = *(vector<Ideal *> *)gctx->target;

	// log("thread %d is started",ctx->thread_id);

	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			struct timeval start = get_cur_time();
			ideals[i]->init_raster(ideals[i]->get_boundary()->num_vertices / gctx->vpr);
			ideals[i]->use_hierachy = gctx->use_hierachy;
			if(gctx->use_hierachy) {
				ideals[i]->grid_align();
				ideals[i]->layering();
			}else{
				ideals[i]->set_status_size();
			}
			ideals[i]->rasterization(ctx->vpr);
			ctx->report_progress();
		}
	}
	// ctx->merge_global();
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
		UniversalGrid::getInstance().configure(gctx->max_layers);
	}

	process_rasterization(gctx);

#ifdef USE_GPU
	cuda_create_buffer(gctx);		
#endif
	target_ideals.clear();

	gctx->target = NULL;
}