#ifndef SRC_GEOMETRY_QUERY_CONTEXT_H_
#define SRC_GEOMETRY_QUERY_CONTEXT_H_

#include <stdlib.h>
#include <stdio.h>
#include <boost/program_options.hpp>

#include "Point.h"
#include "Box.h"
#include "query_profiler.h"

namespace po = boost::program_options;
using namespace std;
class MyPolygon;
class Farm;
struct EdgeSeq;
struct RasterInfo;
struct FarmOffset;

enum QueryType
{
	intersect = 0,
	intersection = 1,
	within = 2
};

class query_context
{
public:
	using WorkerFunction = void *(*)(void *);

	int thread_id = 0;

	// configuration
	bool geography = true;
	int num_threads = 0;

	int vpr = 10;
	bool use_hierarchy = false;
	bool use_approximation = false;

	int mer_sample_round = 20;
	bool perform_refine = true;
	float sample_rate = 1.0;
	size_t batch_size = 0;
	int bitwidth = 4;
	float merge_threshold = 0.9;
	float approx_confidence = 0.9;
	int NLow = 1;
	int unroll_size = 16;

	QueryType query_type;
	int within_distance = 10;

	string source_path;
	string target_path;

	size_t max_num_polygons = INT_MAX;

	// shared staff, for multiple thread task assignment
	size_t index = 0;
	size_t index_end = 0;
	struct timeval previous = get_cur_time();
	// the gap between two reports, in ms
	int report_gap = 100;
	pthread_mutex_t lk;
	const char *report_prefix = "processed";

	// result
	double distance = 0;
	bool contain = false;
	double *areas = nullptr;

	// query statistic
	QueryProfiler profiler;

	size_t found = 0;
	size_t query_count = 0;

	// temporary storage for query processing
	vector<Farm *> source_objects;
	vector<Farm *> target_objects;
	vector<Farm *> referred_objects;
	void *target = NULL;
	void *target2 = NULL;
	query_context *global_ctx = NULL;
	size_t target_num = 0;
	size_t target_id = 0;

	// for gpu
	FarmOffset *h_farm_offset = nullptr;
	FarmOffset *d_farm_offset = nullptr;
	RasterInfo *h_info = nullptr;
	RasterInfo *d_info = nullptr;
	uint8_t *h_status = nullptr;
	uint8_t *d_status = nullptr;
	uint32_t *h_offset = nullptr;
	uint32_t *d_offset = nullptr;
	EdgeSeq *h_edge_sequences = nullptr;
	EdgeSeq *d_edge_sequences = nullptr;
	Point *h_vertices = nullptr;
	Point *d_vertices = nullptr;
	uint32_t *h_gridline_offset = nullptr;
	uint32_t *d_gridline_offset = nullptr;
	double *h_gridline_nodes = nullptr;
	double *d_gridline_nodes = nullptr;
	RasterInfo *h_layer_info = nullptr;
	RasterInfo *d_layer_info = nullptr;
	uint32_t *h_layer_offset = nullptr;
	uint32_t *d_layer_offset = nullptr;
	
	size_t num_polygons = 0;
	size_t num_status = 0;
	size_t num_offset = 0;
	size_t num_edge_sequences = 0;
	size_t num_vertices = 0;
	size_t num_gridline_offset = 0;
	size_t num_gridline_nodes = 0;

	float *h_degree_degree_per_kilometer_latitude = nullptr;
	float *h_degree_per_kilometer_longitude_arr = nullptr;
	float *d_degree_degree_per_kilometer_latitude = nullptr;
	float *d_degree_per_kilometer_longitude_arr = nullptr;

	char* d_BufferInput = nullptr;
	uint *d_bufferinput_size = nullptr;
	char* d_BufferOutput = nullptr;
	uint *d_bufferoutput_size = nullptr;

	uint *d_result = nullptr;

	// for hierachy
	int num_layers = 0;
	int max_layers = 25;

	// for index
	vector<pair<uint32_t, uint32_t>> object_pairs;
	pair<uint32_t, uint32_t>* h_candidate_pairs = nullptr;
	pair<uint32_t, uint32_t>* d_candidate_pairs = nullptr;
	size_t num_pairs = 0;

public:
	// functions
	query_context();
	~query_context();
	query_context(const query_context &) = delete;
	query_context &operator=(const query_context &) = delete;
	query_context(query_context &&) = delete;
	query_context &operator=(query_context &&) = delete;
	void lock();
	void unlock();

	// for multiple thread
	void report_progress(int eval_batch = 10);
	bool next_batch(int batch_num = 1);
	void run_worker_threads(WorkerFunction worker, void *worker_target = nullptr, void *worker_target2 = nullptr);

	void merge_global();
	void merge_object_pairs();

	void print_stats();

	// utility functions for query types
	bool is_within_query()
	{
		return query_type == QueryType::within;
	}

	bool within(double dist)
	{
		return is_within_query() && dist <= within_distance;
	}
};

void get_parameters(int argc, char **argv, query_context &global_ctx);

#endif /* SRC_GEOMETRY_QUERY_CONTEXT_H_ */
