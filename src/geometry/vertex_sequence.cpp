/*
 * vertex_sequence.cpp
 *
 *  Created on: Apr 25, 2023
 *      Author: teng
 */
#include "../include/MyPolygon.h"

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
inline int orientation(Point p, Point q, Point r){
	double val = (q.y - p.y) * (r.x - q.x) -
			  (q.x - p.x) * (r.y - q.y);
	if (val == 0.0) return 0;  // colinear
	return (val > 0)? 1: 2; // clock or counterclock wise
}

inline int orientation(double px, double py, double qx, double qy, double rx, double ry){
	double val = (qy - py) * (rx - qx) -
			  (qx - px) * (ry - qy);
	if (val == 0.0) return 0;  // colinear
	return (val > 0)? 1: 2; // clock or counterclock wise
}

VertexSequence *VertexSequence::convexHull() {

    int n = num_vertices - 1;
    // There must be at least 3 points
    if (n < 3) {
        return NULL;
    }
    // Initialize Result
    vector<int> hull;

    // Find the leftmost point
    int lp = 0;
    for (int i = 1; i < n; i++)
        if (p[i].x < p[lp].x)
            lp = i;

    // Start from leftmost point, keep moving counterclockwise
    // until reach the start point again.  This loop runs O(h)
    // times where h is number of points in result or output.
    int idx = 0;
    bool complete = false;
    do {
        // Add current point to result
        hull.push_back(lp);

        // Search for a point 'q' such that orientation(lp, x,
        // q) is counterclockwise for all points 'x'. The idea
        // is to keep track of last visited most counterclock-
        // wise point in q. If any point 'i' is more counterclock-
        // wise than q, then update q.
        int q = (lp + 1) % n;
        for (int i = 0; i < n; i++) {
            // If i is more counterclockwise than current q, then
            // update q
            if (q != i && orientation(p[lp].x, p[lp].y, p[i].x, p[i].y, p[q].x,
                                      p[q].y) == 2)
                q = i;
        }

        // Now q is the most counterclockwise with respect to lp
        // Set lp as q for next iteration, so that q is added to
        // result 'hull'
        lp = q;
        for (int ih = 0; ih < hull.size() && !complete; ih++) {
            complete = (lp == hull[ih]);
        }
    } while (!complete); // While we don't come to first point

    VertexSequence *ch = new VertexSequence(hull.size() + 1);
    for (int i = 0; i < hull.size(); i++) {
        ch->p[i].x = p[hull[i]].x;
        ch->p[i].y = p[hull[i]].y;
    }
    ch->p[hull.size()].x = ch->p[0].x;
    ch->p[hull.size()].y = ch->p[0].y;
    hull.clear();
    return ch;
}

void VertexSequence::print(bool complete_ring){
	cout<<"(";
	for(int i=0;i<num_vertices;i++){
		if(i!=0){
			cout<<",";
		}
		printf("%.12f ",p[i].x);
		printf("%.12f",p[i].y);
	}
	// the last vertex should be the same as the first one for a complete ring
	if(complete_ring){
		if(p[0].x!=p[num_vertices-1].x||p[0].y!=p[num_vertices-1].y){
			cout<<",";
			printf("%f ",p[0].x);
			printf("%f",p[0].y);
		}
	}
	cout<<")";
}

double VertexSequence::area(){
	double a = 0.0f;
	double b = 0.0f;

	puts("------------------------------------------------------------------------");
	for(int i = 0; i < num_vertices - 1; i ++){
		int j = i + 1;
        a += p[i].x * p[j].y;
        b += p[i].y * p[j].x;
		printf("%lf %lf\n", a, b);
	}
	puts("------------------------------------------------------------------------");
	return abs(a - b) / 2.0;
}

bool VertexSequence::clockwise(){
	return area()<0;
}

void VertexSequence::reverse(){
	Point tmp;
	for(int i=0;i<num_vertices/2;i++){
		tmp = p[i];
		p[i] = p[num_vertices-1-i];
		p[num_vertices-1-i] = tmp;
	}
}

VertexSequence::VertexSequence(int nv){
	p = new Point[nv];
	num_vertices = nv;
}
VertexSequence::VertexSequence(int nv, Point *pp){
	assert(nv>0);
	num_vertices = nv;
	p = new Point[nv];
	memcpy((char *)p,(char *)pp,num_vertices*sizeof(Point));
};

VertexSequence::~VertexSequence(){
	if(p){
		delete []p;
	}
}

vector<Vertex *> VertexSequence::pack_to_polyline(){
	vector<Vertex *> polyline;
	polyline.resize(num_vertices-1);
	for(int i=0;i<num_vertices-1;i++){
		polyline[i] = new Vertex(p[i].x,p[i].y);
	}
	return polyline;
}

box *VertexSequence::getMBR(){
	box *mbr = new box();
	float bias = 0.00001;
	for(int i=0;i<num_vertices;i++){
		mbr->low[0] = min(mbr->low[0], p[i].x);
		mbr->high[0] = max(mbr->high[0], p[i].x);
		mbr->low[1] = min(mbr->low[1], p[i].y);
		mbr->high[1] = max(mbr->high[1], p[i].y);
	}
	mbr->low[0] -= bias;
	mbr->high[0] += bias;
	mbr->low[1] -= bias;
	mbr->high[1] += bias;

	return mbr;
}

void VertexSequence::fix(){
	if(num_vertices<=3){
		return;
	}
	int cur = 0;
	int next = 1;
	unordered_map<double, double> exist;
	exist[p[cur].x+p[cur].y] = p[cur].x;
	while(next<num_vertices){
		// next vertex appeared before, skip this one
		if(exist.find(p[next].x+p[next].y)!=exist.end() && exist[p[next].x+p[next].y] == p[next].x){
			next++;
			continue;
		}
		// not cllinear and shown first time, move cur forward
		if(!collinear(p[cur],p[next],p[(next+1)%num_vertices])){
			p[++cur] = p[next];
			// register the new one
			exist[p[cur].x+p[cur].y] = p[cur].x;
		}
		// the next vertex is valid
		next++;
	}
	exist.clear();
	num_vertices = cur+1;
}

size_t VertexSequence::encode(char *dest){
	assert(num_vertices>0);
	size_t encoded = 0;
	((long *)dest)[0] = num_vertices;
	encoded += sizeof(size_t);
	memcpy(dest+encoded,(char *)p,num_vertices*sizeof(Point));
	encoded += num_vertices*sizeof(Point);
	return encoded;
}

size_t VertexSequence::decode(char *source){
	size_t decoded = 0;
	num_vertices = ((size_t *)source)[0];
	assert(num_vertices>0);
	p = new Point[num_vertices + 1];
	decoded += sizeof(size_t);
	memcpy((char *)p,source+decoded,num_vertices*sizeof(Point));
	decoded += num_vertices*sizeof(Point);
	num_vertices ++;
	p[num_vertices - 1] = p[0]; 
	assert(p[0] == p[num_vertices - 1]);
	return decoded;
}

size_t VertexSequence::get_data_size(){
	return sizeof(size_t)+num_vertices*sizeof(Point);
}


VertexSequence *VertexSequence::clone(){
	VertexSequence *ret = new VertexSequence(num_vertices);
	memcpy((void *)ret->p,(void *)p,sizeof(Point)*num_vertices);
	return ret;
}

/*
 *
 * containment test
 *
 * */
bool VertexSequence::contain(Point &point) {
	bool ret = false;
	for(int i = 0,j=num_vertices-1; i < num_vertices; j=i++) {
		if( ( (p[i].y >= point.y ) != (p[j].y >= point.y) ) &&
				(point.x <= (p[j].x - p[i].x) * (point.y - p[i].y) / (p[j].y - p[i].y) + p[i].x)){
			ret = !ret;
		}
	}
	return ret;
}

double VertexSequence::distance(Point &point, bool geography) {
	return point_to_segment_sequence_distance(point, p, num_vertices, geography);
}
