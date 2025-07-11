/*
 * Point.h
 *
 *  Created on: Jan 1, 2021
 *      Author: teng
 */

#ifndef SRC_GEOMETRY_POINT_H_
#define SRC_GEOMETRY_POINT_H_
#include "util.h"
#include <stdio.h>
#include <string>

using namespace std;

const static char *point_char = "POINT";
class Edge;

class Point {
public:
    float x;
    float y;

    CUDA_HOSTDEV Point() : x(0), y(0) {}
    CUDA_HOSTDEV Point(float xx, float yy) : x(xx), y(yy) {}

    /// Set this point to all zeros.
    void set_zero() {
        x = 0.0;
        y = 0.0;
    }

    /// Set this point to some specified coordinates.
    CUDA_HOSTDEV void set(float x_, float y_) {
        x = x_;
        y = y_;
    }

    CUDA_HOSTDEV bool operator==(const Point &p) const { return fabs(x-p.x) < 1e-9 && fabs(y-p.y) < 1e-9; }
    CUDA_HOSTDEV bool operator!=(const Point &p) const { return fabs(x-p.x) >= 1e-9 || fabs(y-p.y) >= 1e-9; }
    CUDA_HOSTDEV bool operator<(const Point &p) const { 
        if(fabs(x-p.x) >= 1e-9){
            return p.x > x;
        }else if(fabs(y-p.y) >= 1e-9){
            return p.y < y;
        }else{
            return false;
        }
    }
    CUDA_HOSTDEV bool operator<=(const Point &p) const { 
        if(fabs(x-p.x) >= 1e-9){
            return p.x > x;
        }else if(fabs(y-p.y) >= 1e-9){
            return p.y < y;
        }else{
            return true;
        }
    }

    /// Negate this point.
    CUDA_HOSTDEV Point operator-() const {
        Point v;
        v.set(-x, -y);
        return v;
    }

    CUDA_HOSTDEV Point operator+(const Point &p) const { return Point(x + p.x, y + p.y); }
    CUDA_HOSTDEV Point operator-(const Point &p) const { return Point(x - p.x, y - p.y); }
    CUDA_HOSTDEV Point operator*(const float &t) const { return Point(t * x, t * y); }

    /// Add a point to this point.
    CUDA_HOSTDEV void operator+=(const Point &v) {
        x += v.x;
        y += v.y;
    }

    /// Subtract a point from this point.
    CUDA_HOSTDEV void operator-=(const Point &v) {
        x -= v.x;
        y -= v.y;
    }

    /// Multiply this point by a scalar.
    CUDA_HOSTDEV void operator*=(float a) {
        x *= a;
        y *= a;
    }

    /// Get the length of this point (the norm).
    float Length() const { return sqrt(x * x + y * y); }

    /// Convert this point into a unit point. Returns the Length.
    float Normalize() {
        float len = Length();
        x /= len;
        y /= len;
        return len;
    }

    CUDA_HOSTDEV float cross(const Point &p) const { return x * p.y - y * p.x; }

    CUDA_HOSTDEV float cross(const Point &a, const Point &b) const {
        return (a - *this).cross(b - *this);
    }

    void print() { printf("POINT (%.12f %.12f)\n", x, y); }
    
    void print_without_return() { printf("POINT (%f %f)", x, y); }
    
    string to_string() {
        char double_str[200];
        sprintf(double_str, "POINT(%f %f)", x, y);
        return string(double_str);
    }
    
    static Point *read_one_point(string &input_line) {

        if (input_line.size() == 0) {
            return NULL;
        }
        const char *wkt = input_line.c_str();
        size_t offset = 0;
        // read the symbol MULTIPOLYGON
        while (wkt[offset] != 'P') {
            offset++;
        }
        for (int i = 0; i < strlen(point_char); i++) {
            assert(wkt[offset++] == point_char[i]);
        }
        skip_space(wkt, offset);
        Point *p = new Point();
        p->x = read_double(wkt, offset);
        p->y = read_double(wkt, offset);

        return p;
    }
};

struct Intersection{
	Point p;
    int pair_id;
    int edge_source_id;     
    int edge_target_id;     
    double t;
    double u;

	// Intersection(){}
	// Intersection(Point _p, int _pair_id, int _edge_source_id, int _edge_target_id, double _t, double _u) : p(_p), pair_id(_pair_id), edge_source_id(_edge_source_id), edge_target_id(_edge_target_id), t(_t), u(_u){}

	void print(){
		printf("POINT(%lf %lf), %d %u %u %lf %lf\n", p.x, p.y, pair_id, edge_source_id, edge_target_id, t, u);
	}
};


class Vertex : public Point {
  public:
    /// The edges this point constitutes an upper ending point
    std::vector<Edge *> edge_list;

    Vertex(float xx, float yy) {
        x = xx;
        y = yy;
    }
};

// Represents a simple polygon's edge
class Edge {
  public:
    Vertex *p, *q;

    /// Constructor
    Edge(Vertex *p1, Vertex *p2) : p(p1), q(p2) {
        if (p1->y > p2->y) {
            q = p1;
            p = p2;
        } else if (p1->y == p2->y) {
            if (p1->x > p2->x) {
                q = p1;
                p = p2;
            } else if (p1->x == p2->x) {
                // Repeat points
                assert(false);
            }
        }

        q->edge_list.push_back(this);
    }
};

#endif /* SRC_GEOMETRY_POINT_H_ */
