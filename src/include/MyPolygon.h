/*
 * MyPolygon.h
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#ifndef SRC_MYPOLYGON_H_
#define SRC_MYPOLYGON_H_

#include <vector>
#include <string>
#include <stdint.h>
#include <math.h>
#include <stack>
#include <map>
#include <bits/stdc++.h>

#include "../geometry/triangulate/poly2tri.h"
#include "util.h"
#include "../index/RTree.h"
#include "../index/QTree.h"
#include "BaseGeometry.h"
#include "Box.h"
#include "Point.h"
#include "query_context.h"
#include "geometry_computation.h"

using namespace std;
using namespace p2t;

const static char *multipolygon_char = "MULTIPOLYGON";
const static char *polygon_char = "POLYGON";

// the structured metadata of a polygon
typedef struct PolygonMeta_
{
	uint size;		   // size of the polygon in bytes
	uint num_vertices; // number of vertices in the boundary (hole excluded)
	size_t offset;	   // the offset in the file
	box mbr;		   // the bounding boxes
} PolygonMeta;

class VertexSequence
{
public:
	int num_vertices = 0;
	Point *p = NULL;

public:
	VertexSequence() {};
	VertexSequence(int nv);
	VertexSequence(int nv, Point *pp);
	size_t encode(char *dest);
	size_t decode(char *source);
	size_t get_data_size();

	~VertexSequence();
	vector<Vertex *> pack_to_polyline();
	VertexSequence *clone();
	VertexSequence *convexHull();
	box *getMBR();
	void print(bool complete_ring = false);
	bool clockwise();
	void reverse();
	double area();
	bool contain(Point &p);
	double distance(Point &p, bool geography = true);
	// fix the collinear problem
	void fix();
};

class MyPolygon : virtual public BaseGeometry
{
private:
	size_t id = 0;
	size_t hc_id = 0;
	vector<VertexSequence *> holes;

	box *mer = NULL;

	VertexSequence *convex_hull = NULL;

	// for triangulation
	Point *triangles = NULL;
	size_t triangle_num = 0;

	RTNode *rtree = NULL;
	QTNode *qtree = NULL;

	pthread_mutex_t qtree_partition_lock;

protected:
	VertexSequence *boundary = NULL;

public:
	MyPolygon()
	{
		pthread_mutex_init(&qtree_partition_lock, NULL);
	}
	~MyPolygon();

	VertexSequence *get_boundary() { return boundary; }
	VertexSequence *get_boundary(int num_vertices)
	{
		if (boundary)
			return boundary;
		boundary = new VertexSequence(num_vertices);
		return boundary;
	}
	void set_boundary(VertexSequence *_boundary) { boundary = _boundary; }
	vector<VertexSequence *> get_holes() { return holes; }
	MyPolygon *clone();

	//  R-tree on triangles
	// size_t get_triangle_size();
	// RTNode *get_rtree() {return rtree;}
	size_t get_rtree_size();
	void triangulate();
	void build_rtree();

	static VertexSequence *read_vertices(const char *wkt, size_t &offset, bool clockwise = true);
	static MyPolygon *read_polygon(const char *wkt, size_t &offset);
	static bool validate_polygon(const char *wkt, size_t &offset, size_t &len);
	static bool validate_vertices(const char *wkt, size_t &offset, size_t &len);

	static MyPolygon *gen_box(double minx, double miny, double maxx, double maxy);
	static MyPolygon *gen_box(box &pix);
	// static MyPolygon *read_one_polygon();

	// some query functions
	bool contain(Point &p); // brute-forcely check containment
	bool contain_rtree(RTNode *node, Point &p, query_context *ctx);
	double distance_rtree(Point &p, query_context *ctx);
	double distance_rtree(Point &start, Point &end, query_context *ctx);
	bool contain(Point &p, query_context *ctx, bool profile = false);
	bool contain(MyPolygon *target, query_context *ctx);
	double distance(Point &p, query_context *ctx, bool profile = false);
	double distance(MyPolygon *target, query_context *ctx, bool profile = false);

	// bool intersect_box(box *target);

	// double distance_rtree(Point &p, query_context *ctx);
	// double distance_rtree(Point &start, Point &end, query_context *ctx);

	// some utility functions
	void print_without_head(bool print_hole = false, bool complete_ring = false);
	void print(bool print_id = true, bool print_hole = false);
	// void print_triangles();
	// void print_without_return(bool print_hole=false, bool complete_ring=false);
	// string to_string(bool clockwise = false, bool complete_ring=false);

	// for filtering
	box *getMBB();
	box *getMER(query_context *ctx = NULL);
	VertexSequence *get_convex_hull();
	QTNode *partition_qtree(const int vpr);
	box *get_mer() { return mer; }
	QTNode *get_qtree() { return qtree; }

	inline int get_num_vertices()
	{
		if (!boundary)
		{
			return 0;
		}
		return boundary->num_vertices;
	}
	inline double getx(int index)
	{
		assert(boundary && index < boundary->num_vertices);
		return boundary->p[index].x;
	}
	inline double gety(int index)
	{
		assert(boundary && index < boundary->num_vertices);
		return boundary->p[index].y;
	}
	inline Point *get_point(int index)
	{
		assert(boundary && index < boundary->num_vertices);
		return &boundary->p[index];
	}
	inline size_t getid() { return id; }
	inline void setid(size_t iid) { id = iid; }
	double area()
	{
		assert(boundary);
		return boundary->area();
	}

	PolygonMeta get_meta();
	size_t get_data_size();
	size_t encode(char *target);
	size_t decode(char *source);
};

class MyMultiPolygon
{
	vector<MyPolygon *> polygons;

public:
	static MyMultiPolygon *read_multipolygon();
	static MyPolygon *read_one_polygon();
	static bool validate_wkt(string &wkt_str);

	MyMultiPolygon(const char *wkt);
	MyMultiPolygon() {};
	~MyMultiPolygon();
	int num_polygons()
	{
		return polygons.size();
	}
	void print();
	void print_separate();
	vector<MyPolygon *> get_polygons()
	{
		return polygons;
	}
	MyPolygon *get_polygon(int index)
	{
		return index >= polygons.size() ? NULL : polygons[index];
	}
	size_t insert_polygon(MyPolygon *p)
	{
		polygons.push_back(p);
		return polygons.size();
	}
	size_t insert_polygon(vector<MyPolygon *> ps)
	{
		for (MyPolygon *p : ps)
		{
			polygons.push_back(p);
		}
		return polygons.size();
	}
	size_t get_data_size()
	{
		size_t ds = 0;
		for (MyPolygon *p : polygons)
		{
			ds += p->get_data_size();
		}
		return ds;
	}
};

class MyMultiPoint
{
	vector<Point *> points;

public:
	void insert(vector<Point *> &origin_points);
	void insert(Point *p);
	void print(size_t max_num = INT_MAX);
};

// utility functions
//  void process_rasterization(query_context *ctx);
//  void process_convex_hull(query_context *ctx);
//  void process_mer(query_context *ctx);
//  void process_internal_rtree(query_context *gctx);
//  void preprocess(query_context *gctx);

void print_boxes(vector<box *> boxes);

// storage related functions
size_t load_points_from_path(const char *path, Point **points);
vector<MyPolygon *> load_polygons_from_path(const char *path, query_context &ctx);
size_t load_mbr_from_file(const char *path, box **);
size_t load_polygonmeta_from_file(const char *path, PolygonMeta **pmeta);

size_t number_of_objects(const char *path);
box universe_space(const char *path);

#endif /* SRC_MYPOLYGON_H_ */