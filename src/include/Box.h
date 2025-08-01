#ifndef BOX_H
#define BOX_H

#include "Point.h"

#define BOX_GLOBAL_MIN 100000.0
#define BOX_GLOBAL_MAX -100000.0

class box{
public:
	double low[2] = {BOX_GLOBAL_MIN,BOX_GLOBAL_MIN};
	double high[2] = {BOX_GLOBAL_MAX,BOX_GLOBAL_MAX};

	CUDA_HOSTDEV box(){}

	CUDA_HOSTDEV box(box *b){
		low[0] = b->low[0];
		high[0] = b->high[0];
		low[1] = b->low[1];
		high[1] = b->high[1];
	}
	
	CUDA_HOSTDEV box (double lowx, double lowy, double highx, double highy){
		low[0] = lowx;
		low[1] = lowy;
		high[0] = highx;
		high[1] = highy;
	}

	bool valid();

	box get_union(box &b);
	box get_intersection(box &b);


	void update(Point &p);
	void update(box &b);

	double height();
	double width();
	double area();

	bool intersect(Point &start, Point &end);
	bool intersect(box &target);
	bool contain(box &target);
	bool contain(Point &p);

	// distance to box
	double distance(box &target, bool geography);
	double max_distance(box &target, bool geography);

	// distance to point
	double distance(Point &p, bool geography);
	double max_distance(Point &p, bool geography);

	// distance to segment
	double distance(Point &start, Point &end, bool geography);
	double max_distance(Point &start, Point &end, bool geography);

	box expand(double expand_buffer, bool geography);

	Point centroid();

	void print_vertices();
	void print();
	void to_array(Point *p);
};

#endif // BOX_H