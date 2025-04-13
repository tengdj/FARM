#ifndef IDEAL_H
#define IDEAL_H

#include "MyPolygon.h"
#include "MyRaster.h"
#include "Hraster.h"

#define BUFFER_SIZE 1024 * 1024 * 1024

enum Direction{
	HORIZONTAL = 0,
	VERTICAL = 1
};

enum cross_type{
	ENTER = 0,
	LEAVE = 1
};

class cross_info{
public:
	cross_type type;
	int edge_id;
	cross_info(cross_type t, int e){
		type = t;
		edge_id = e;
	}
};

struct IdealPair
{
	uint32_t source;
	uint32_t target;
	int pair_id;
};

struct IdealOffset{
	uint status_start;
	uint offset_start;
	uint edge_sequences_start;
	uint vertices_start;
	uint gridline_offset_start;
	uint gridline_nodes_start;
	uint layer_start;
};

struct EdgeSeq{
	uint start;
	uint length;
};

class Grid_line{
	uint32_t *offset = nullptr;
	double *intersection_nodes = nullptr;

	size_t num_grid_lines = 0;
	size_t num_crosses = 0;
public:
	Grid_line() = default;
	Grid_line(int size);
	~Grid_line();
	void init_intersection_node(int num_nodes);
	int get_num_nodes(int y) {return offset[y + 1] - offset[y];}
	void add_node(int idx, double x) {intersection_nodes[idx] = x;}

	size_t get_num_grid_lines() {return num_grid_lines; }
	void set_num_crosses(size_t x) {num_crosses = x;}
	size_t get_num_crosses() {return num_crosses;}
	void set_offset(int id, int idx) {offset[id] = idx;}
	uint32_t get_offset(int id) {return offset[id];}
	double get_intersection_nodes(int id) {return intersection_nodes[id];}
	uint32_t *get_offset() {return offset;}
	double *get_intersection_nodes() {return intersection_nodes;}
};


class Ideal : public MyPolygon, public MyRaster{
public:
	bool use_hierachy = false;
	size_t id = 0;

private:
	uint32_t *offset = nullptr;
	pair<uint32_t, uint32_t> *edge_sequences = nullptr;
	Grid_line *horizontal = nullptr;
	Grid_line *vertical = nullptr;
	uint32_t *layer_offset = nullptr;
    RasterInfo *layer_info = nullptr;
	Hraster *layers = nullptr;

	uint len_edge_sequences = 0;
	uint num_layers = 0;
	uint status_size = 0;


    pthread_mutex_t ideal_partition_lock;
	void init_pixels();
	void evaluate_edges();
	void scanline_reandering();

public:
    Ideal(){
        pthread_mutex_init(&ideal_partition_lock, NULL);
    }
	~Ideal();
    void rasterization(int vertex_per_raster);
	void rasterization();

	void set_offset(int id, int idx){offset[id] = idx;}
	uint32_t get_offset(int id) {return offset[id];}
	uint32_t *get_offset() {return offset; }
	void process_pixels_null(int x, int y);
	void init_edge_sequences(int num_edge_seqs);
	void add_edge(int idx, int start, int end);
	pair<uint32_t, uint32_t> get_edge_sequence(int idx){return edge_sequences[idx];}
	pair<uint32_t, uint32_t> *get_edge_sequence(){return edge_sequences;}
	uint get_len_edge_sequences() {return len_edge_sequences;}
	uint32_t get_num_sequences(int id);
	double get_possible_min(box *t_mbr, int core_x_low, int core_y_low, int core_x_high, int code_y_high, int step, bool geography = true);
	double get_possible_min(Point &p, int center, int step, bool geography = true);
	void process_crosses(map<int, vector<cross_info>> edge_info);
	void process_intersection(map<int, vector<double>> edge_intersection, Direction direction);
	int count_intersection_nodes(Point &p);
	Grid_line *get_vertical() {return vertical;}
	Hraster* get_layers() { return layers; }
	uint get_num_layers() { return num_layers; }
	uint get_status_size() { return status_size; }
	RasterInfo* get_layer_info() { return layer_info; }
	uint32_t* get_layer_offset() { return layer_offset; }


	// statistic collection
	int get_num_border_edge();
	int num_edges_covered(int id);
	// size_t get_num_gridlines();
	size_t get_num_crosses();
	// double get_num_intersection();
	

	// query functions
	bool contain(Point &p, query_context *ctx, bool profile = false);
	bool contain(Ideal *target, query_context *ctx, bool profile = false);
	// bool intersect(MyPolygon *target, query_context *ctx);
	double distance(Point &p, query_context *ctx, bool profile = false);
	double distance(Ideal *target, query_context *ctx);
	double distance(Ideal *target, int pix, query_context *ctx, bool profile = true);
};

//utility functions
void process_rasterization(query_context *ctx);
void preprocess(query_context *gctx);

// storage related functions
vector<Ideal *> load_binary_file(const char *path, query_context &ctx);
VertexSequence *read_vertices(const char *wkt, size_t &offset, bool clockwise);
Ideal *read_polygon(const char *wkt, size_t &offset);
vector<Ideal *> load_polygon_wkt(const char *path);
Point *load_point_wkt(const char *path, size_t &count, query_context *ctx);
void dump_to_file(const char *path, char *data, size_t size);
void dump_polygons_to_file(vector<Ideal *> polygons, const char *path);

// gpu functions
#ifdef USE_GPU
void indexFilter(query_context *gctx);
void cuda_create_buffer(query_context *gctx);
void preprocess_for_gpu(query_context *gctx);
void cuda_contain(query_context *gctx, bool polygon);
void cuda_contain_polygon(query_context *gctx);
uint cuda_within(query_context *gctx);
void cuda_within_polygon(query_context *gctx);
#endif

#endif // IDEAL_H