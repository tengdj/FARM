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
    double x;
    double y;
    CUDA_HOSTDEV Point() {
        // x = 0;
        // y = 0;
    }
    CUDA_HOSTDEV Point(double xx, double yy) {
        x = xx;
        y = yy;
    }
    void print() { printf("POINT (%f %f)\n", x, y); }
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
    /// Set this point to all zeros.
    void set_zero() {
        x = 0.0;
        y = 0.0;
    }

    /// Set this point to some specified coordinates.
    CUDA_HOSTDEV void set(double x_, double y_) {
        x = x_;
        y = y_;
    }

    //	  /// Negate this point.
    //	  Point operator =(const Point &v) const
    //	  {
    //	    Point r;
    //	    r.x = v.x;
    //	    r.y = v.y;
    //	    return r;
    //	  }

    CUDA_HOSTDEV bool operator==(const Point &p) const { return x == p.x && y == p.y; }
    

    /// Negate this point.
    CUDA_HOSTDEV Point operator-() const {
        Point v;
        v.set(-x, -y);
        return v;
    }

    CUDA_HOSTDEV Point operator-(const Point &p) const { return Point(x - p.x, y - p.y); }

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
    CUDA_HOSTDEV void operator*=(double a) {
        x *= a;
        y *= a;
    }



    /// Get the length of this point (the norm).
    double Length() const { return sqrt(x * x + y * y); }

    /// Convert this point into a unit point. Returns the Length.
    double Normalize() {
        double len = Length();
        x /= len;
        y /= len;
        return len;
    }

    CUDA_HOSTDEV double cross(const Point &p) const { return x * p.y - y * p.x; }

    CUDA_HOSTDEV double cross(const Point &a, const Point &b) const {
        return (a - *this).cross(b - *this);
    }
};

class Vertex : public Point {
  public:
    /// The edges this point constitutes an upper ending point
    std::vector<Edge *> edge_list;

    Vertex(double xx, double yy) {
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
