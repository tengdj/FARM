#include "../include/Ideal.h"

void *rasterization_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;

	vector<Ideal *> &ideals = *(vector<Ideal *> *)gctx->target;

	// log("thread %d is started",ctx->thread_id);

	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			struct timeval start = get_cur_time();
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

	//collect partitioning status
	size_t num_partitions = 0;
	size_t num_crosses = 0;
	size_t num_border_partitions = 0;
	size_t num_edges = 0;
	for(auto ideal : ideals){
		num_partitions += ideal->get_num_pixels();
		num_crosses += ideal->get_num_crosses();
		num_border_partitions += ideal->get_num_pixels(BORDER);
		num_edges += ideal->get_num_border_edge();
	}
	logt("IDEALized %d polygons with (%ld)%ld average pixels %.2f average crosses per pixel %.2f edges per pixel", start,
			ideals.size(),
			num_border_partitions/ideals.size(),
			num_partitions/ideals.size(),
			1.0*num_crosses/num_border_partitions,
			1.0*num_edges/num_border_partitions);

	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void *convex_hull_unit(void *args){

	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	//log("thread %d is started",ctx->thread_id);
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			polygons[i]->get_convex_hull();
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_convex_hull(query_context *gctx){

	// must be non-empty
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	assert(polygons.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = polygons.size();

	struct timeval start = get_cur_time();
	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, convex_hull_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	//collect convex hull status
	size_t num_vertexes = 0;
	size_t data_size = 0;
	size_t ch_size = 0;
	for(MyPolygon *poly:polygons){
		VertexSequence *ch = poly->get_convex_hull();
		if(ch){
			num_vertexes += ch->num_vertices;
			ch_size += ch->num_vertices*16;
		}
		data_size += poly->get_data_size();
	}

	logt("get convex hull for %d polygons with %ld average vertices %f overhead", start,
			polygons.size(),
			num_vertexes/polygons.size(),
			ch_size*100.0/data_size);
	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void *mer_unit(void *args){

	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			polygons[i]->getMER(ctx);
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_mer(query_context *gctx){

	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	assert(polygons.size()>0);

	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = polygons.size();

	struct timeval start = get_cur_time();
	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, mer_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	//collect convex hull status
	double mbr_are = 0;
	double mer_are = 0;
	size_t data_size = 0;
	size_t mer_size = 0;
	for(MyPolygon *poly:polygons){
		data_size += poly->get_data_size();
		mer_size += 4*8;
		mbr_are += poly->getMBB()->area();
		if(poly->getMER(gctx)){
			mer_are += poly->getMER(gctx)->area();
		}
	}

	logt("get mer for %d polygons with %f mer/mbr with %f overhead", start,
			polygons.size(),
			mer_are/mbr_are,
			mer_size*100.0/data_size);
	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void *internal_rtree_unit(void *args){

	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			polygons[i]->build_rtree();
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_internal_rtree(query_context *gctx){
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	assert(polygons.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = polygons.size();

	struct timeval start = get_cur_time();
	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, internal_rtree_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	//collect convex hull status
	size_t data_size = 0;
	size_t rtree_size = 0;
	for(MyPolygon *poly:polygons){
		data_size += poly->get_data_size();
		rtree_size += poly->get_rtree_size();
	}

	logt("RTree of %d polygons with %f overhead", start,
			polygons.size(),
			rtree_size*100.0/data_size);

	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void *qtree_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;

	//log("thread %d is started",ctx->thread_id);
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			struct timeval start = get_cur_time();
			polygons[i]->partition_qtree(ctx->vpr);
			ctx->report_progress();
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_qtree(query_context *gctx){
	log("start rasterizing the referred polygons");
	vector<MyPolygon *> &polygons = *(vector<MyPolygon *> *)gctx->target;
	assert(polygons.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = polygons.size();

	struct timeval start = get_cur_time();
	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, qtree_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	//collect partitioning status
	size_t num_partitions = 0;
	size_t num_crosses = 0;
	size_t num_border_partitions = 0;
	size_t num_edges = 0;
	for(MyPolygon *poly:polygons){
		num_partitions += poly->get_qtree()->leaf_count();
		num_border_partitions += poly->get_qtree()->border_leaf_count();
	}
	logt("partitioned %d polygons with (%ld)%ld average pixels %.2f average crosses per pixel %.2f edges per pixel", start,
			polygons.size(),
			num_border_partitions/polygons.size(),
			num_partitions/polygons.size(),
			1.0*num_crosses/num_border_partitions,
			1.0*num_edges/num_border_partitions);

	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}


void preprocess(query_context *gctx){

	if(gctx->use_ideal){
		vector<Ideal *> target_ideals;
		target_ideals.insert(target_ideals.end(), gctx->source_ideals.begin(), gctx->source_ideals.end());
		target_ideals.insert(target_ideals.end(), gctx->target_ideals.begin(), gctx->target_ideals.end());
		gctx->target = (void *)&target_ideals;

		process_rasterization(gctx);

#ifdef USE_GPU
		cuda_create_buffer(gctx);		
#endif
		target_ideals.clear();
	}
	
	if(gctx->use_vector){
		vector<MyPolygon *> target_polygons;
		target_polygons.insert(target_polygons.end(), gctx->source_polygons.begin(), gctx->source_polygons.end());
		target_polygons.insert(target_polygons.end(), gctx->target_polygons.begin(), gctx->target_polygons.end());
		gctx->target = (void *)&target_polygons;

		process_convex_hull(gctx);
		process_mer(gctx);
		target_polygons.clear();
		target_polygons.insert(target_polygons.end(), gctx->source_polygons.begin(), gctx->source_polygons.end());
		// gctx->target = (void *)&target_polygons;
		process_internal_rtree(gctx);
	
		target_polygons.clear();
	}

	if(gctx->use_qtree){
		vector<MyPolygon *> target_polygons;
		target_polygons.insert(target_polygons.end(), gctx->source_polygons.begin(), gctx->source_polygons.end());
		target_polygons.insert(target_polygons.end(), gctx->target_polygons.begin(), gctx->target_polygons.end());
		gctx->target = (void *)&target_polygons;

		process_qtree(gctx);

		target_polygons.clear();
	}

	gctx->target = NULL;
}