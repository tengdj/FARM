#include "../include/farm.h"
#include "UniversalGrid.h"

void *rasterization_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;

	vector<Farm *> &objs = *(vector<Farm *> *)gctx->target;

	log("thread %d is started",ctx->thread_id);

	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
				objs[i]->set_bitwidth(gctx->bitwidth);
			objs[i]->init_raster(objs[i]->get_boundary()->num_vertices / gctx->vpr);
			objs[i]->use_hierarchy = gctx->use_hierarchy;
			if(gctx->use_hierarchy) {
				objs[i]->grid_align();
				objs[i]->layering(gctx->NLow);
			}else{
				objs[i]->set_status_size();
			}
			objs[i]->rasterization(gctx->vpr);
			ctx->report_progress();
		}
	}
	// ctx->merge_global();
	return NULL;
}

void process_rasterization(query_context *gctx){
	log("start rasterizing the referred polygons");
	vector<Farm *> &objs = *(vector<Farm *> *)gctx->target;
	assert(objs.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = objs.size();

	// struct timeval start = get_cur_time();
	gctx->run_worker_threads(rasterization_unit);

	gctx->index = 0;
	gctx->target_num = former;
}

void preprocess(query_context *gctx){
	gctx->referred_objects.clear();
	vector<uint8_t> source_seen(gctx->source_objects.size(), 0);
	vector<uint8_t> target_seen(gctx->target_objects.size(), 0);

	auto add_source = [&](uint32_t idx) {
		if(idx < gctx->source_objects.size() && !source_seen[idx]){
			source_seen[idx] = 1;
			gctx->referred_objects.push_back(gctx->source_objects[idx]);
		}
	};
	auto add_target = [&](uint32_t idx) {
		if(idx < gctx->target_objects.size() && !target_seen[idx]){
			target_seen[idx] = 1;
			gctx->referred_objects.push_back(gctx->target_objects[idx]);
		}
	};

	if(gctx->num_pairs > 0){
		size_t source_count = gctx->source_objects.size();
		for(size_t i = 0; i < gctx->num_pairs; i++){
			auto p = gctx->object_pairs[i];
			add_source(p.first);
			if(gctx->target_path.empty()){
				add_source(p.second);
			}else{
#ifdef USE_GPU
				if(p.second >= source_count && p.second - source_count < gctx->target_objects.size()){
					add_target(p.second - source_count);
				}else{
					add_target(p.second);
				}
#else
				add_target(p.second);
#endif
			}
		}
	}

	if(gctx->referred_objects.empty()){
		gctx->referred_objects.insert(gctx->referred_objects.end(), gctx->source_objects.begin(), gctx->source_objects.end());
		gctx->referred_objects.insert(gctx->referred_objects.end(), gctx->target_objects.begin(), gctx->target_objects.end());
	}
	gctx->target = (void *)&gctx->referred_objects;

	if(gctx->use_hierarchy){
		UniversalGrid::getInstance().configure(gctx->max_layers);
	}
	
	process_rasterization(gctx);

#ifdef USE_GPU
	cuda_create_buffer(gctx);		
#endif

	gctx->target = NULL;

}
