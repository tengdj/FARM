#include "query_context.h"
#include "../include/farm.h"

query_context::query_context(){
	num_threads = get_num_threads();
	pthread_mutex_init(&lk, NULL);
}
query_context::~query_context(){
	// this is the host context
	if(this->global_ctx == NULL){
		delete[] areas;
		areas = nullptr;
	}
	pthread_mutex_destroy(&lk);

}

void query_context::lock(){
	pthread_mutex_lock(&lk);
}

void query_context::unlock(){
	pthread_mutex_unlock(&lk);
}

void query_context::run_worker_threads(WorkerFunction worker, void *worker_target, void *worker_target2){
	if(num_threads <= 0){
		log("ERROR: thread count must be positive");
		exit(EXIT_FAILURE);
	}

	vector<pthread_t> threads(num_threads);
	vector<query_context> contexts(num_threads);
	for(int i = 0; i < num_threads; i++){
		// Copy configuration only; shared data remains reachable through global_ctx.
		contexts[i].geography = geography;
		contexts[i].num_threads = num_threads;
		contexts[i].vpr = vpr;
		contexts[i].use_hierarchy = use_hierarchy;
		contexts[i].use_approximation = use_approximation;
		contexts[i].mer_sample_round = mer_sample_round;
		contexts[i].perform_refine = perform_refine;
		contexts[i].sample_rate = sample_rate;
		contexts[i].batch_size = batch_size;
		contexts[i].bitwidth = bitwidth;
		contexts[i].merge_threshold = merge_threshold;
		contexts[i].approx_confidence = approx_confidence;
		contexts[i].NLow = NLow;
		contexts[i].unroll_size = unroll_size;
		contexts[i].query_type = query_type;
		contexts[i].within_distance = within_distance;
		contexts[i].source_path = source_path;
		contexts[i].target_path = target_path;
		contexts[i].max_num_polygons = max_num_polygons;
		contexts[i].report_gap = report_gap;
		contexts[i].report_prefix = report_prefix;
		contexts[i].global_ctx = this;
		contexts[i].thread_id = i;
		contexts[i].target = worker_target;
		contexts[i].target2 = worker_target2;
	}

	for(int i = 0; i < num_threads; i++){
		int error = pthread_create(&threads[i], nullptr, worker, &contexts[i]);
		if(error != 0){
			log("ERROR: failed to create thread %d: %d", i, error);
			exit(EXIT_FAILURE);
		}
	}

	for(int i = 0; i < num_threads; i++){
		int error = pthread_join(threads[i], nullptr);
		if(error != 0){
			log("ERROR: failed to join thread %d: %d", i, error);
			exit(EXIT_FAILURE);
		}
	}
}

void query_context::report_progress(int eval_batch){
	if(++query_count==eval_batch){
		global_ctx->lock();
		global_ctx->query_count += query_count;
		double time_passed = get_time_elapsed(global_ctx->previous);
		if(time_passed>global_ctx->report_gap){
			log_refresh("%s %zu (%.2f%%)",global_ctx->report_prefix, global_ctx->query_count,(double)global_ctx->query_count*100/(global_ctx->target_num));
			global_ctx->previous = get_cur_time();
		}
		global_ctx->unlock();
		query_count = 0;
	}
}

void query_context::merge_global(){
	global_ctx->lock();
	global_ctx->found += found;
	global_ctx->profiler.merge(profiler);
	global_ctx->unlock();
}

void query_context::merge_object_pairs() {
    size_t local_size = object_pairs.size();
    if (local_size == 0) return;

	global_ctx->lock();
	size_t required_size = global_ctx->object_pairs.size() + local_size;
    if (required_size > global_ctx->object_pairs.capacity()) {
		size_t old_capacity = global_ctx->object_pairs.capacity();
		size_t new_capacity = old_capacity * 2;
		if (new_capacity < required_size || new_capacity < old_capacity) {
			new_capacity = required_size;
		}
		log("WARNING: Reserving global object_pairs %zu", old_capacity);
		global_ctx->object_pairs.reserve(new_capacity);
    }

	global_ctx->object_pairs.insert(global_ctx->object_pairs.end(),
	                                object_pairs.begin(),
	                                object_pairs.end());
	global_ctx->num_pairs = global_ctx->object_pairs.size();
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
	log("count-found:\t%zu",found);
	profiler.print();
}


void get_parameters(int argc, char **argv, query_context &global_ctx){
	po::options_description desc("query usage");
	desc.add_options()
		("help", "produce help message")
		("approximation,a", "use approximation")
		("hierarchy,h", "partition with hierarchical grid")

		("source,s", po::value<string>(&global_ctx.source_path), "path to the source")
		("target,t", po::value<string>(&global_ctx.target_path), "path to the target")
		("threads,n", po::value<int>(&global_ctx.num_threads), "number of threads")
		("vpr,v", po::value<int>(&global_ctx.vpr), "number of vertices per raster")
		("batch_size,b", po::value<size_t>(&global_ctx.batch_size), "batch size")
		("merge_threshold,m", po::value<float>(&global_ctx.merge_threshold), "merge threshold")
		("approx_confidence,c", po::value<float>(&global_ctx.approx_confidence), "approximation confidence threshold")
		("NLow,l", po::value<int>(&global_ctx.NLow), "NLow")
		("unroll_size,u", po::value<int>(&global_ctx.unroll_size), "unroll size")
		("within_distance,d", po::value<int>(&global_ctx.within_distance), "within distance")
		("bitwidth,w", po::value<int>(&global_ctx.bitwidth), "bits per raster status")
		;
	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
	}catch(...){
		cerr << "Undefined Option!" << endl;
		exit(EXIT_FAILURE);
	}
	if (vm.count("help")) {
		cout << desc << "\n";
		exit(0);
	}
	po::notify(vm);
	if(global_ctx.vpr <= 0){
		cerr << "vpr must be greater than 0" << endl;
		exit(EXIT_FAILURE);
	}
	if(global_ctx.bitwidth < 2 || global_ctx.bitwidth > 8){
		cerr << "bitwidth must be between 2 and 8" << endl;
		exit(EXIT_FAILURE);
	}
	if(global_ctx.unroll_size <= 0){
		cerr << "unroll size must be greater than 0" << endl;
		exit(EXIT_FAILURE);
	}

	global_ctx.use_approximation = vm.count("approximation");
	global_ctx.use_hierarchy = vm.count("hierarchy");

	assert(global_ctx.approx_confidence >= 0.0f && global_ctx.approx_confidence <= 1.0f);

}
