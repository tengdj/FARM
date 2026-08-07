#ifndef FARM_H
#define FARM_H

#include "MyPolygon.h"
#include "MyRaster.h"
#include "Hraster.h"

enum Direction
{
	HORIZONTAL = 0,
	VERTICAL = 1
};

enum cross_type
{
	ENTER = 0,
	LEAVE = 1
};

class cross_info
{
public:
	cross_type type;
	int edge_id;
	cross_info(cross_type t, int e)
	{
		type = t;
		edge_id = e;
	}
};

struct FarmOffset
{
	uint status_start; // Byte offset into the packed GPU status array.
	uint offset_start;
	uint edge_sequences_start;
	uint vertices_start;
	uint gridline_offset_start;
	uint gridline_nodes_start;
	uint layer_start;
};

struct EdgeSeq
{
	uint start;
	uint length;
};

class Grid_line
{
	uint32_t *offset = nullptr;
	double *intersection_nodes = nullptr;

	size_t num_grid_lines = 0;
	size_t num_crosses = 0;
	size_t num_intersected_lines = 0;

public:
	Grid_line() = default;
	Grid_line(int size);
	~Grid_line();
	void init_intersection_node(int num_nodes);
	int get_num_nodes(int y) { return offset[y + 1] - offset[y]; }
	void add_node(int idx, double x) { intersection_nodes[idx] = x; }

	size_t get_num_grid_lines() { return num_grid_lines; }
	void set_num_crosses(size_t x) { num_crosses = x; }
	size_t get_num_crosses() { return num_crosses; }
	void set_num_intersected_lines(size_t x) { num_intersected_lines = x; }
	size_t get_num_intersected_lines() { return num_intersected_lines; }
	void set_offset(int id, int idx) { offset[id] = idx; }
	uint32_t get_offset(int id) { return offset[id]; }
	double get_intersection_nodes(int id) { return intersection_nodes[id]; }
	uint32_t *get_offset() { return offset; }
	double *get_intersection_nodes() { return intersection_nodes; }
};

class Farm : public MyPolygon, public MyRaster
{
public:
	bool use_hierarchy = false;
	size_t id = 0;

private:
	uint32_t *offset = nullptr;
	pair<uint32_t, uint32_t> *edge_sequences = nullptr;
	Grid_line *horizontal = nullptr;
	Grid_line *vertical = nullptr;
	uint32_t *layer_offset = nullptr; // Byte offsets of byte-aligned packed layers.
	RasterInfo *layer_info = nullptr;
	Hraster *layers = nullptr;
	double *areas = nullptr;

	uint len_edge_sequences = 0;
	uint num_layers = 0;

	pthread_mutex_t farm_partition_lock;
	void init_pixels();
	void evaluate_edges();
	void calculate_fullness();
	bool border_pixel_intersects_box(int pixel_id, box &overlap, query_context *ctx);

public:
	Farm()
	{
		pthread_mutex_init(&farm_partition_lock, NULL);
	}
	~Farm();
	void rasterization(int vertex_per_raster);
	void rasterization();

	void set_offset(int id, int idx) { offset[id] = idx; }
	uint32_t get_offset(int id) { return offset[id]; }
	uint32_t *get_offset() { return offset; }
	void process_pixels_null(int x, int y);
	void init_edge_sequences(int num_edge_seqs);
	void add_edge(int idx, int start, int end);
	pair<uint32_t, uint32_t> get_edge_sequence(int idx) { return edge_sequences[idx]; }
	pair<uint32_t, uint32_t> *get_edge_sequence() { return edge_sequences; }
	uint get_len_edge_sequences() { return len_edge_sequences; }
	uint32_t get_num_sequences(int id);
	void process_crosses_sparse(const vector<int> &pixel_ids, const vector<vector<cross_info>> &edges_info);
	void process_intersection(vector<vector<double>>& intersection_info, Direction direction);
	int count_intersection_nodes(Point &p);
	Grid_line *get_horizontal() { return horizontal; }
	Grid_line *get_vertical() { return vertical; }
	double get_areas(int id) { return areas[id]; }
	double *get_areas() { return areas; }
	double decodePixelArea(int id, bool isLow);
	uint8_t encodePixelArea(double area);
	uint8_t encodePixelArea(double area, double pixel_area_val);

	void layering(int NLow);
	Hraster *get_layers() { return layers; }
	uint get_num_layers() { return num_layers; }
	uint get_status_size() { return status_size; }
	RasterInfo *get_layer_info() { return layer_info; }
	uint32_t *get_layer_offset() { return layer_offset; }
	// double merge_area(box target, PartitionStatus &status);
	void merge_status(Hraster &parent, const Hraster &child);

	// statistic collection
	int num_edges_covered(int id);
	// size_t get_num_gridlines();
	// double get_num_intersection();

	// query functions
	bool contain(Point &p, query_context *ctx);
	bool intersect(Farm *target, query_context *ctx, bool approximation = false);
	// bool intersect(MyPolygon *target, query_context *ctx);
	void intersection(Farm *target, query_context *ctx, bool approximation = false);
	bool within(Farm *target, query_context *ctx, bool approximation = false);

};

// utility functions
void process_rasterization(query_context *ctx);
void preprocess(query_context *gctx);

// storage related functions
vector<Farm *> load_binary_file(const char *path, query_context &ctx);
	VertexSequence *read_vertices(const char *wkt, size_t &offset, bool clockwise);
	Farm *read_polygon(const char *wkt, size_t &offset);
	vector<Farm *> load_polygon_wkt(const char *path);
	void dump_to_file(const char *path, char *data, size_t size);
void dump_polygons_to_file(vector<Farm *> polygons, const char *path);

// gpu functions

#ifdef USE_GPU
void cuda_create_buffer(query_context *gctx);
void preprocess_for_gpu(query_context *gctx);
void cuda_intersect(query_context *gctx);
void cuda_intersection(query_context *gctx);
void cuda_within_polygon(query_context *gctx);
#endif

#endif // FARM_H
