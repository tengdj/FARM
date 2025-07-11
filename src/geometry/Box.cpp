#include "../include/Box.h"
#include "../include/geometry_computation.h"

bool print_debug = false;

bool box::valid(){
	return low[0] <= high[0] && low[1] <= high[1];

}

void box::update(Point &p){
	low[0] = min(low[0], (double)p.x);
	low[1] = min(low[1], (double)p.y);
	high[0] = max(high[0], (double)p.x);
	high[1] = max(high[1], (double)p.y);
}

void box::update(box &b){
	low[0] = min(low[0], b.low[0]);
	low[1] = min(low[1], b.low[1]);
	high[0] = max(high[0], b.high[0]);
	high[1] = max(high[1], b.high[1]);
}

box box::get_intersection(box &b){
	assert(this->intersect(b));
	return box(max(low[0],b.low[0]), max(low[1], b.low[1]), min(high[0],b.high[0]), min(high[1], b.high[1]));;
}

box box::get_union(box &b){
	box unioned = *this;
	unioned.update(b);
	return unioned;
}



bool box::intersect(Point &start, Point &end){
	if(segment_intersect(start, end, Point(low[0], low[1]), Point(high[0], low[1]))){
		return true;
	}
	if(segment_intersect(start, end, Point(high[0], low[1]), Point(high[0], high[1]))){
		return true;
	}
	if(segment_intersect(start, end, Point(high[0], high[1]), Point(low[0], high[1]))){
		return true;
	}
	if(segment_intersect(start, end, Point(low[0], high[1]), Point(low[0], low[1]))){
		return true;
	}
	return false;
}
bool box::intersect(box &target){
	return !(target.low[0]>high[0]||
			 target.high[0]<low[0]||
			 target.low[1]>high[1]||
			 target.high[1]<low[1]);
}
bool box::contain(box &target){
	return target.low[0]>=low[0]&&
		   target.high[0]<=high[0]&&
		   target.low[1]>=low[1]&&
		   target.high[1]<=high[1];
}
bool box::contain(Point &p){
	return p.x>=low[0]&&
		   p.x<=high[0]&&
		   p.y>=low[1]&&
		   p.y<=high[1];
}

double box::area(){
	return (high[0]-low[0])*(high[1]-low[1]);
}

double box::height(){
	return (high[1]-low[1]);
}

double box::width(){
	return (high[0]-low[0]);
}


/*
 * distance related functions
 * */

// box to box
double box::distance(box &t, bool geography){
	if(this->intersect(t)){
		return 0;
	}
	double dx = 0;
	double dy = 0;

	if(t.low[1]>high[1]){
		dy = t.low[1] - high[1];
	}else if(t.high[1]<low[1]){
		dy = low[1] - t.high[1];
	}else{
		dy = 0;
	}

	if(t.low[0]>high[0]){
		dx = t.low[0] - high[0];
	}else if(t.high[0]<low[0]){
		dx = low[0] - t.high[0];
	}else{
		dx = 0;
	}

	if(geography){
		dy = dy/degree_per_kilometer_latitude;
		dx = dx/degree_per_kilometer_longitude(low[1]);
	}
	return sqrt(dx * dx + dy * dy);
}

double box::max_distance(box &target, bool geography){
	double highx = max(high[0],target.high[0]);
	double highy = max(high[1],target.high[1]);
	double lowx = min(low[0], target.low[0]);
	double lowy = min(low[1], target.low[1]);
	double dx = highx - lowx;
	double dy = highy - lowy;
	if(geography){
		dy = dy/degree_per_kilometer_latitude;
		dx = dx/degree_per_kilometer_longitude(low[1]);
	}
	return sqrt(dx * dx + dy * dy);
}

// point to box
double box::distance(Point &p, bool geography){
	if(this->contain(p)){
		return 0;
	}
	double dx = max(abs(p.x-(low[0]+high[0])/2) - (high[0]-low[0])/2, 0.0);
	double dy = max(abs(p.y-(low[1]+high[1])/2) - (high[1]-low[1])/2, 0.0);
	if(geography){
		dy = dy/degree_per_kilometer_latitude;
		dx = dx/degree_per_kilometer_longitude(p.y);
	}
	return sqrt(dx * dx + dy * dy);
}

double box::max_distance(Point &p, bool geography){

	double dx = max(abs(p.x-low[0]), abs(p.x-high[0]));
	double dy = max(abs(p.y-low[1]), abs(p.y-high[1]));

	if(geography){
		dy = dy/degree_per_kilometer_latitude;
		dx = dx/degree_per_kilometer_longitude(p.y);
	}
	return sqrt(dx*dx+dy*dy);
}

// segment to box
double box::distance(Point &start, Point &end, bool geography){
	Point vertex_array[5];
	to_array(vertex_array);
	double dist1 = segment_to_segment_distance(start, end, vertex_array[0], vertex_array[1], geography);
	double dist2 = segment_to_segment_distance(start, end, vertex_array[1], vertex_array[2], geography);
	double dist3 = segment_to_segment_distance(start, end, vertex_array[2], vertex_array[3], geography);
	double dist4 = segment_to_segment_distance(start, end, vertex_array[3], vertex_array[4], geography);

	return min(dist1, min(dist2, min(dist3, dist4)));
}
double box::max_distance(Point &start, Point &end, bool geography){
	Point vertex_array[5];
	to_array(vertex_array);
	double dist1 = segment_to_segment_distance(start, end, vertex_array[0], vertex_array[1], geography);
	double dist2 = segment_to_segment_distance(start, end, vertex_array[1], vertex_array[2], geography);
	double dist3 = segment_to_segment_distance(start, end, vertex_array[2], vertex_array[3], geography);
	double dist4 = segment_to_segment_distance(start, end, vertex_array[3], vertex_array[4], geography);

	return max(dist1, max(dist2, max(dist3, dist4)));
}


box box::expand(double expand_buffer, bool geography){
	box b = *this;
	double shiftx = expand_buffer;
	double shifty = expand_buffer;
	if(geography){
		shiftx *= degree_per_kilometer_longitude(b.low[1]);
		shifty *= degree_per_kilometer_latitude;
	}

	b.low[0] -= shiftx;
	b.high[0] += shiftx;
	b.low[1] -= shifty;
	b.high[1] += shifty;
	return b;
}

Point box::centroid(){
	return Point((high[0]+low[0])/2, (high[1]+low[1])/2);
}

void box::to_array(Point *p){
	p[0].x = low[0];
	p[0].y = low[1];
	p[1].x = high[0];
	p[1].y = low[1];
	p[2].x = high[0];
	p[2].y = high[1];
	p[3].x = low[0];
	p[3].y = high[1];
	p[4].x = low[0];
	p[4].y = low[1];
}

/*
 * print functions
 * */
void box::print_vertices(){
	printf("%f %f, %f %f, %f %f, %f %f, %f %f",
				low[0],low[1],
				high[0],low[1],
				high[0],high[1],
				low[0],high[1],
				low[0],low[1]);
}

void box::print(){
	printf("POLYGON((");
	print_vertices();
	printf("))\n");
}