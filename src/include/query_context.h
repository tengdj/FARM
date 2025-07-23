#ifndef SRC_GEOMETRY_QUERY_CONTEXT_H_
#define SRC_GEOMETRY_QUERY_CONTEXT_H_

#include <stdlib.h>
#include <stdio.h>
#include <map>
#include <boost/program_options.hpp>
#include <cfloat>

#ifdef USE_GPU
#include <../cuda/mygpu.h>
#endif

#include "Point.h"
#include "Box.h"

namespace po = boost::program_options;
using namespace std;
class MyPolygon;
class Ideal;
struct EdgeSeq;
struct RasterInfo;
struct IdealOffset;
struct IdealPair;
struct Segment;

enum QueryType
{
	contain = 0,
	contain_polygon = 1,
	within = 2,
	within_polygon = 3,
	intersection = 4,
	intersect = 5
};

class execute_step
{
public:
	size_t counter = 0;
	double execution_time = 0;
	execute_step &operator=(execute_step const &obj)
	{
		counter = obj.counter;
		execution_time = obj.counter;
		return *this;
	}
	void reset()
	{
		counter = 0;
		execution_time = 0;
	}

	execute_step &operator+=(const execute_step &rhs)
	{
		this->counter += rhs.counter;
		this->execution_time += rhs.execution_time;
		return *this;
	}
};

class configurations
{
public:
	int thread_id = 0;
	int num_threads = 0;
	int vpr = 10;
	int vpr_end = 10;
	bool use_ideal = false;
	bool use_convex_hull = false;
	bool use_mer = false;
	bool use_triangulate = false;
	int mer_sample_round = 20;
	bool perform_refine = true;
	bool gpu = false;
	bool collect_latency = false;
	float sample_rate = 1.0;
	int small_threshold = 500;
	int big_threshold = 1000000;
	bool sort_polygons = false;
	int distance_buffer_size = 10;
	QueryType query_type = QueryType::contain;
	string source_path;
	string target_path;
	string valid_path;
};

class query_context
{
public:
	int thread_id = 0;

	// configuration
	bool geography = true;
	int num_threads = 0;

	int vpr = 10;
	bool use_ideal = false;
	bool use_raster = false;
	bool use_vector = false;
	bool use_qtree = false;
	bool use_gpu = false;
	bool use_hierachy = false;

	int mer_sample_round = 20;
	bool perform_refine = true;
	bool collect_latency = false;
	float sample_rate = 1.0;
	double load_factor = 1.0;
	size_t batch_size = 0;
	int category_count = 20;
	float merge_threshold = 0.9;

	int small_threshold = 500;
	int big_threshold = 400000;

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
	double area = 0.0f;

	// query statistic
	double raster_filter_time = 0.0;
	double refine_time = 0.0;

	size_t found = 0;
	size_t query_count = 0;
	size_t refine_count = 0;

	execute_step object_checked;
	execute_step node_check;
	execute_step contain_check;
	execute_step pixel_evaluated;
	execute_step border_evaluated;
	execute_step border_checked;
	execute_step edge_checked;
	execute_step intersection_checked;

	// temporary storage for query processing
	vector<MyPolygon *> source_polygons;
	vector<MyPolygon *> target_polygons;
	vector<Ideal *> source_ideals;
	vector<Ideal *> target_ideals;
	Point *points = NULL;
	void *target = NULL;
	void *target2 = NULL;
	void *target3 = NULL;
	query_context *global_ctx = NULL;
	size_t target_num = 0;
	size_t target_id = 0;

	map<int, int> vertex_number;
	map<int, double> latency;

	// for gpu

	Point* d_points = nullptr;
	IdealOffset *h_idealoffset = nullptr;
	IdealOffset *d_idealoffset = nullptr;
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

	float *d_mean = nullptr;
	float *d_stddev = nullptr;

	float *d_degree_degree_per_kilometer_latitude = nullptr;
	float *d_degree_per_kilometer_longitude_arr = nullptr;

	char* d_BufferInput = nullptr;
	uint *d_bufferinput_size = nullptr;
	char* d_BufferOutput = nullptr;
	uint *d_bufferoutput_size = nullptr;

	uint *d_result = nullptr;

	// for hierachy
	double min_step_x = DBL_MAX;
	double min_step_y = DBL_MAX;
	box space = {10000.0, 10000.0, -10000.0, -10000.0};
	int num_layers = 0;
	int max_layers = 25;

	// for index
	vector<pair<uint32_t, uint32_t>> object_pairs;
	pair<uint32_t, uint32_t>* h_candidate_pairs = nullptr;
	pair<uint32_t, uint32_t>* d_candidate_pairs = nullptr;
	size_t num_pairs = 0;

	// for intersection
	Segment *segments = nullptr;
	uint num_segments = 0;
	uint8_t *pip = nullptr;

	vector<MyPolygon*> intersection_polygons;

public:
	// functions
	query_context();
	~query_context();
	query_context(query_context &t);
	void lock();
	void unlock();

	// for multiple thread
	void report_progress(int eval_batch = 10);
	bool next_batch(int batch_num = 1);

	// for query statistics
	void report_latency(int num_v, double latency);
	void load_points();
	void merge_global();

	void reset_stats()
	{
		// query statistic
		found = 0;
		query_count = 0;
		refine_count = 0;
		index = 0;

		object_checked.reset();
		pixel_evaluated.reset();
		border_evaluated.reset();
		border_checked.reset();
		edge_checked.reset();
		intersection_checked.reset();
		node_check.reset();
		contain_check.reset();
	}
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

	bool is_contain_query()
	{
		return query_type == QueryType::contain;
	}
};

query_context get_parameters(int argc, char **argv);

#endif /* SRC_GEOMETRY_QUERY_CONTEXT_H_ */
