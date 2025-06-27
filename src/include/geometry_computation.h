/*
 * geometry_computation.h
 *
 *  Created on: Apr 28, 2022
 *      Author: teng
 */

#ifndef SRC_GEOMETRY_GEOMETRY_COMPUTATION_H_
#define SRC_GEOMETRY_GEOMETRY_COMPUTATION_H_

#include <math.h>
#include "util.h"
#include "Point.h"
#include <float.h>

/*
 *
 * some utility functions for geometry computations
 *
 * */


inline bool collinear(Point &p1, Point &p2, Point &p3){
	double a = p1.x * (p2.y - p3.y) +
			   p2.x * (p3.y - p1.y) +
			   p3.x * (p1.y - p2.y);
	return double_zero(a);
}

inline double getSlope(Point &A, Point &B)
{
	return ((B.y - A.y) / (B.x - A.x));
}

inline bool isInsideHorizontalDual(double Yi, double Yi1, double &py)
{
	if (py < Yi || py > Yi1)
	{
		return false;
	}
	return true;
}
inline bool isInsideVerticalDual(double Xi, double Xi1, double &px)
{
	if (px < Xi || px > Xi1)
	{
		return false;
	}
	return true;
}

/*
 * distance related
 * */

inline double point_to_segment_distance(const Point &p, const Point &p1, const Point &p2, bool geography) {

  double A = p.x - p1.x;
  double B = p.y - p1.y;
  double C = p2.x - p1.x;
  double D = p2.y - p1.y;

  double dot = A * C + B * D;
  double len_sq = C * C + D * D;
  double param = -1;
  if (len_sq != 0) //in case of 0 length line
      param = dot / len_sq;

  double xx, yy;

  if (param < 0) {
    xx = p1.x;
    yy = p1.y;
  } else if (param > 1) {
    xx = p2.x;
    yy = p2.y;
  } else {
    xx = p1.x + param * C;
    yy = p1.y + param * D;
  }

  double dx = p.x - xx;
  double dy = p.y - yy;
  if(geography){
      dx = dx/degree_per_kilometer_longitude(p.y);
      dy = dy/degree_per_kilometer_latitude;
  }
  return sqrt(dx * dx + dy * dy);
}

inline double point_to_segment_sequence_distance(Point &p, Point *vs, size_t seq_len, bool geography){
    double mindist = DBL_MAX;
    for (int i = 0; i < seq_len-1; i++) {
        double dist = point_to_segment_distance(p, vs[i], vs[i+1], geography);
        if(dist<mindist){
            mindist = dist;
        }
    }
    return mindist;
}

inline double segment_to_segment_distance(Point &s1, Point &e1, Point &s2, Point &e2, bool geography){
	double dist1 = point_to_segment_distance(s1,s2,e2,geography);
	double dist2 = point_to_segment_distance(e1,s2,e2,geography);
	double dist3 = point_to_segment_distance(s2,s1,e1,geography);
	double dist4 = point_to_segment_distance(e2,s1,e1,geography);
	return min(dist1, min(dist2, min(dist3, dist4)));
}

inline double segment_sequence_distance(Point *vs1, Point *vs2, size_t s1, size_t s2, bool geography){
    double mindist = DBL_MAX;
	for(int i=0;i<s1;i++){
		double dist = point_to_segment_sequence_distance(vs1[i], vs2, s2, geography);
		if(dist<mindist){
			mindist = dist;
		}
	}
	for(int i=0;i<s2;i++){
		double dist = point_to_segment_sequence_distance(vs2[i], vs1, s1, geography);
		if(dist<mindist){
			mindist = dist;
		}
	}
	return mindist;
}


inline double point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len, double within_distance, bool geography, size_t &checked){
    double mindist = DBL_MAX;
    for (int i = 0; i < seq_len-1; i++) {
    	checked++;
        double dist = point_to_segment_distance(p, vs[i], vs[i+1], geography);
        if(dist<mindist){
            mindist = dist;
        }
		if(mindist <= within_distance){
			return mindist;
		}
    }
    return mindist;
}

inline double segment_to_segment_within_batch(Point *vs1, Point *vs2, size_t s1, size_t s2, double within_distance, bool geography, size_t &checked){
    double mindist = DBL_MAX;
	for(int i=0;i<s1;i++){
		double dist = point_to_segment_within_batch(vs1[i], vs2, s2, within_distance, geography, checked);
		if(dist<mindist){
			mindist = dist;
		}
		if(mindist <= within_distance){
			return mindist;
		}
	}
	for(int i=0;i<s2;i++){
		double dist = point_to_segment_within_batch(vs2[i], vs1, s1, within_distance, geography, checked);
		if(dist<mindist){
			mindist = dist;
		}
		if(mindist <= within_distance){
			return mindist;
		}
	}
	return mindist;
}

/*
 *
 * topology related
 *
 * */

inline int sgn(const double& x) {
	return x >= 0 ? x ? 1 : 0 : -1;
}

inline bool inter1(double a, double b, double c, double d) {

	double tmp;
    if (a > b){
    	tmp = a;
    	a = b;
    	b = tmp;
    }
    if (c > d){
    	tmp = c;
    	c = d;
    	d = tmp;
    }
    return max(a, c) <= min(b, d);
}

// checking whether two segments intersect
inline bool segment_intersect(const Point& a, const Point& b, const Point& c, const Point& d) {
    if (c.cross(a, d) == 0 && c.cross(b, d) == 0)
        return inter1(a.x, b.x, c.x, d.x) && inter1(a.y, b.y, c.y, d.y);
    return sgn(a.cross(b, c)) != sgn(a.cross(b, d)) &&
           sgn(c.cross(d, a)) != sgn(c.cross(d, b));
}

// checking whether two segment sequences intersect
inline bool segment_intersect_batch(Point *p1, Point *p2, int s1, int s2){
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			if(segment_intersect(p1[i],p1[i+1],p2[j],p2[j+1])){
				return true;
			}
		}
	}
	return false;
}

inline void segment_intersect_batch(Point *p1, Point *p2, int s1, int s2, int e1, int e2, std::vector<Intersection>& inters){
	for(int i = s1; i < e1; i ++){
		for(int j = s2; j < e2; j ++){	
			Point d1 = p1[i + 1] - p1[i];
			Point d2 = p2[j + 1] - p2[j];
			Point r = p2[j] - p1[i];

			double denom = d1.cross(d2);

			if (std::abs(denom) < 1e-9) continue;

			double t = r.cross(d2) / denom;
			double u = r.cross(d1) / denom;

			if (t >= -1e-9 && t <= 1 + 1e-9 && u >= -1e-9 && u <= 1 + 1e-9) {
				Point intersect_p = p1[i] + d1 * t;	
				Intersection inter = {intersect_p, 0, i, j, t, u};
				inters.push_back(inter);
			}
		}
	}
	return;
}

/*
 *
 * area related
 *
 */

inline double computePolygonArea(vector<Point> &polygon)
{
	int n = polygon.size();
	if (n < 3)
		return 0.0;

	double area = 0.0;
	for (int i = 0; i < n - 1; ++i)
	{
		const Point &p1 = polygon[i];
		const Point &p2 = polygon[i + 1];
		area += (p1.x * p2.y - p2.x * p1.y);
	}
	return std::abs(area) / 2.0;
}

inline uint8_t classifySubpolygon(double area, double pixelArea, int count)
{
	double ratio = area / pixelArea;
	if (fabs(ratio - 1.0) < 1e-6)
	{
		// full
		return count - 1;
	}

	if (fabs(ratio) < 1e-6)
	{
		// empty
		return 0;
	}

	int idx = static_cast<int>((ratio * (count - 2)) + 1);
	if (idx >= count)
		idx = count - 1; // 防止越界

	// int idx = static_cast<int>(ceil(ratio * (count - 2)));
	assert(idx < 256);
	return idx;
}


#endif /* SRC_GEOMETRY_GEOMETRY_COMPUTATION_H_ */