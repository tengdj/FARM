/*
 * query_context.cpp
 *
 *  Created on: Sep 3, 2020
 *      Author: teng
 */
#include "query_context.h"
#include "../include/Ideal.h"

query_context::query_context(){
	num_threads = get_num_threads();
	pthread_mutex_init(&lk, NULL);
}
query_context::query_context(query_context &t){
	*this = t;
	this->source_polygons.clear();
	this->target_polygons.clear();
	this->global_ctx = &t;
	pthread_mutex_init(&lk, NULL);
}
query_context::~query_context(){

	vertex_number.clear();
	latency.clear();
	// this is the host context
	if(this->global_ctx == NULL){
		for(MyPolygon *p:source_polygons){
			delete p;
		}
		for(MyPolygon *p:target_polygons){
			delete p;
		}
		if(points){
			delete []points;
		}
	}

}

void query_context::lock(){
	pthread_mutex_lock(&lk);
}

void query_context::unlock(){
	pthread_mutex_unlock(&lk);
}

//query_context& query_context::operator=(query_context const &t){
//    geography = t.geography;
//
//	thread_id = t.thread_id;
//	num_threads = t.num_threads;
//	vpr = t.vpr;
//	vpr_end = t.vpr_end;
//	use_grid = t.use_grid;
//	use_qtree = t.use_qtree;
//	use_mer = t.use_mer;
//	use_triangulate = t.use_triangulate;
//	mer_sample_round = t.mer_sample_round;
//	use_convex_hull = t.use_convex_hull;
//	perform_refine = t.perform_refine;
//	gpu = t.gpu;
//	sample_rate = t.sample_rate;
//	small_threshold = t.small_threshold;
//	big_threshold = t.big_threshold;
//	sort_polygons = t.sort_polygons;
//	report_gap = t.report_gap;
//	distance_buffer_size = t.distance_buffer_size;
//	source_path = t.source_path;
//	target_path = t.target_path;
//	target_num = t.target_num;
//
//    query_type = t.query_type;
//    collect_latency = t.collect_latency;
//    pthread_mutex_init(&lock, NULL);
//	return *this;
//
//}

void query_context::report_latency(int num_v, double lt){
	if(vertex_number.find(num_v)==vertex_number.end()){
		vertex_number[num_v] = 1;
		latency[num_v] = lt;
	}else{
		vertex_number[num_v] = vertex_number[num_v]+1;
		latency[num_v] = latency[num_v]+lt;
	}
}

void query_context::load_points(){
	struct timeval start = get_cur_time();
	target_num = load_points_from_path(target_path.c_str(), &points);
	logt("loaded %ld points", start, target_num);
}

void query_context::report_progress(int eval_batch){
	if(++query_count==eval_batch){
		global_ctx->query_count += query_count;
		global_ctx->lock();
		double time_passed = get_time_elapsed(global_ctx->previous);
		if(time_passed>global_ctx->report_gap){
			log_refresh("%s %d (%.2f\%)",global_ctx->report_prefix, global_ctx->query_count,(double)global_ctx->query_count*100/(global_ctx->target_num));
			global_ctx->previous = get_cur_time();
		}
		global_ctx->unlock();
		query_count = 0;
	}
}

void query_context::merge_global(){
	global_ctx->lock();
	global_ctx->found += found;
	global_ctx->query_count += query_count;
	global_ctx->refine_count += refine_count;


	global_ctx->contain_check += contain_check;
	global_ctx->object_checked += object_checked;
	global_ctx->pixel_evaluated += pixel_evaluated;
	global_ctx->border_evaluated += border_evaluated;
	global_ctx->border_checked += border_checked;
	global_ctx->edge_checked += edge_checked;
	global_ctx->intersection_checked += intersection_checked;

	for(auto &it :vertex_number){
		const double lt = latency.at(it.first);
		if(global_ctx->vertex_number.find(it.first)!=global_ctx->vertex_number.end()){
			global_ctx->vertex_number[it.first] = global_ctx->vertex_number[it.first]+it.second;
			global_ctx->latency[it.first] = global_ctx->latency[it.first]+lt;
		}else{
			global_ctx->vertex_number[it.first] = it.second;
			global_ctx->latency[it.first] = lt;
		}
	}
	
	for(int i = 0; i < num_pairs; i ++){
		global_ctx->h_candidate_pairs[global_ctx->num_pairs ++] = h_candidate_pairs[i];
	}
	
	// point_polygon_pairs_size = point_polygon_pairs_idx;
	// global_ctx->point_polygon_pairs_size += point_polygon_pairs_size;
	
	// for(int i = 0; i < point_polygon_pairs_size; i ++){
	// 	global_ctx->point_polygon_pairs[global_ctx->point_polygon_pairs_idx ++] = point_polygon_pairs[i];
	// }

	global_ctx->unlock();
}

bool query_context::next_batch(int batch_num){
	global_ctx->lock();
	if(global_ctx->index==global_ctx->target_num){
		global_ctx->unlock();
		return false;
	}
	index = global_ctx->index;
	if(index+batch_num>global_ctx->target_num){
		index_end = global_ctx->target_num;
	}else {
		index_end = index+batch_num;
	}
	global_ctx->index = index_end;
	global_ctx->unlock();
	return true;
}

//epp = [10 20 30 40 50 60 70 80 90 100]
//
//border = [0.207285 0.288498 0.347719 0.39932 0.434262 0.46625 0.493314 0.516715 0.534164 0.551719]
//edges = [66.482446 94.79277 119.693534 146.500647 166.748356 196.448964 220.291613 240.176293 266.363209 289.240302]

//pixel = [0.04944511799 0.08518398015 0.1135131606 0.1363504804 0.156446252 0.1751551248 0.1916903609 0.2064784002 0.2204680656 0.2328603489]
//bordereval = [0.2738840444 0.3023781467 0.3175422434 0.3278811077 0.3397997682 0.3522231054 0.3644681595 0.3760295171 0.3860913525 0.3956647175]
//bordercheck = [0.5593827592 0.6193568462 0.6527505046 0.6755005932 0.6934979642 0.7063738728 0.7181750425 0.7268449196 0.7354106478 0.7425720885]
//edge = [33.1568436 52.6532413 70.7067904 88.4315092 106.2501207 124.6311698 143.7354825 161.9386059 181.0331808 198.2849336]

void query_context::print_stats(){

	// log("count-query:\t%ld",query_count);
	// log("count-contain:\t%ld",this->contain_check.counter);
	// log("count-checked:\t%ld",object_checked.counter);
	log("count-found:\t%ld",found);

	if(object_checked.counter>0){
		if(refine_count)
		log("checked-refine:\t%.7f",(double)refine_count/object_checked.counter);
		if(pixel_evaluated.counter)
		log("checked-pixel:\t%.7f",(double)pixel_evaluated.counter/object_checked.counter);
		if(edge_checked.counter)
		log("checked-edges:\t%.7f",(double)edge_checked.counter/object_checked.counter);
	}

	if(border_checked.counter>0){
		log("border-eval:\t%.7f",(double)border_evaluated.counter/object_checked.counter);
		log("border-checked:\t%.7f",(double)border_checked.counter/object_checked.counter);
		log("border-edges:\t%.7f",(double)edge_checked.counter/border_checked.counter);
		log("border-node:\t%.7f",(double)intersection_checked.counter/border_checked.counter);
	}

	if(contain_check.counter>0){
		log("latency-ctn:\t%.7f",contain_check.execution_time/contain_check.counter);
	}

	if(pixel_evaluated.counter>0){
		log("latency-pixel:\t%.7f",pixel_evaluated.execution_time/pixel_evaluated.counter);
	}
	if(border_evaluated.counter>0){
		log("latency-border:\t%.7f",border_evaluated.execution_time/border_evaluated.counter);
	}
	if(edge_checked.execution_time>0){
		log("latency-edge:\t%.7f",edge_checked.execution_time/edge_checked.counter);
	}
	if(intersection_checked.execution_time>0){
		log("latency-node:\t%.7f",intersection_checked.execution_time/intersection_checked.counter);
	}
	if(object_checked.execution_time>0){
		log("latency-other:\t%.7f",(object_checked.execution_time-
									pixel_evaluated.execution_time-border_evaluated.execution_time-
									edge_checked.execution_time-intersection_checked.execution_time-
									contain_check.execution_time
									)/object_checked.counter);
	}
	if(object_checked.execution_time>0){
		log("latency-all:\t%.7f",object_checked.execution_time/object_checked.counter);
	}

	if(collect_latency){
		for(auto it:vertex_number){
			cout<<it.first<<"\t"<<it.second<<"\t"<<latency[it.first]/it.second<<endl;
		}
	}

}


query_context get_parameters(int argc, char **argv){
	query_context global_ctx;

	po::options_description desc("query usage");
	desc.add_options()
		("help", "produce help message")
		("rasterize,r", "partition with rasterization")
		("qtree,q", "partition with qtree")
		("raster_only", "query with raster only")
		("vector", "use techniques like MER convex hull and internal RTree")
		("gpu,g", "query with gpu")
		("hierachy,h", "partition with hierarchical grid")

		("source,s", po::value<string>(&global_ctx.source_path), "path to the source")
		("target,t", po::value<string>(&global_ctx.target_path), "path to the target")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("vpr,v", po::value<int>(&global_ctx.vpr), "number of vertices per raster")
		("batch_size,b", po::value<size_t>(&global_ctx.batch_size), "batch size")
		("load_factor,l", po::value<double>(&global_ctx.load_factor), "load factor")
		// ("big_threshold,b", po::value<int>(&global_ctx.big_threshold), "up threshold for complex polygon")
		// ("small_threshold", po::value<int>(&global_ctx.small_threshold), "low threshold for complex polygon")
		// ("sample_rate", po::value<float>(&global_ctx.sample_rate), "sample rate")
		// ("latency,l","collect the latency information")
		;
	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
	}catch(...){
		cout << "Undefined Option!" << endl;
		exit(0);
	}
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);

	global_ctx.use_ideal = vm.count("rasterize");
	global_ctx.use_vector = vm.count("vector");
	global_ctx.use_raster = vm.count("raster_only");
	global_ctx.use_qtree = vm.count("qtree");
	global_ctx.use_gpu = vm.count("gpu");
	global_ctx.use_hierachy = vm.count("hierachy");


	assert(global_ctx.use_ideal+global_ctx.use_vector+global_ctx.use_qtree<=1
			&&"can only choose one from, IDEAL, VECTOR, QTree");

	return global_ctx;
}