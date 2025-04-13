#pragma once

#include "cuda_util.h"
#include "Ideal.h"

#define BLOCK_SIZE 256
#define WITHIN_DISTANCE 10

const double EARTH_RADIUS_KM = 6371.0;

struct PixMapping
{
	int pair_id = 0;
	int pix_id = 0;
};

struct PixPair
{
	int source_pixid = 0;
	int target_pixid = 0;
	int pair_id = 0;
};

__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    
    do {
        expected = old;
        int new_val;
        if (__int_as_float(expected) <= val) {
            break;
        }
        new_val = __float_as_int(val);
        old = atomicCAS(address_as_int, expected, new_val);
    } while (old != expected);
    return __int_as_float(old);
}

__device__ __forceinline__ double atomicMinDouble(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(fmin(val, __longlong_as_double(assumed))));
	} while (assumed != old);

	return __longlong_as_double(old);
}

__device__ __forceinline__ double atomicExchDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    unsigned long long int val_as_ull = __double_as_longlong(val);

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, val_as_ull);
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__device__ __forceinline__ int gpu_get_id(int x, int y, int dimx)
{
	return y * (dimx + 1) + x;
}

// from id to pixel x
__device__ __forceinline__ int gpu_get_x(int id, int dimx)
{
	return id % (dimx + 1);
}

// from id to pixel y
__device__ __forceinline__ int gpu_get_y(int id, int dimx, int dimy)
{
	assert((id / (dimx + 1)) <= dimy);
	return id / (dimx + 1);
}

__device__ __forceinline__ int gpu_double_to_int(double val)
{
    int vi = (int)val; 
    double diff = abs(1.0 * (vi + 1) - val);
    int adjust = (diff < 0.00000001); 
    return vi + adjust; 
}

__device__ __forceinline__ int gpu_get_offset_x(double s_xval, double t_xval, double step_x, int dimx)
{
	int x = gpu_double_to_int((t_xval - s_xval) / step_x);
	return min(max(x, 0), dimx);
}

__device__ __forceinline__ int gpu_get_offset_y(double s_yval, double t_yval, double step_y, int dimy)
{
	int y = gpu_double_to_int((t_yval - s_yval) / step_y);
	return min(max(y, 0), dimy);
}

// __device__ __forceinline__ PartitionStatus gpu_show_status(uint8_t *status, uint &start, int &id)
// {
// 	uint8_t st = (status + start)[id / 4];
// 	int pos = id % 4 * 2; // The multiplication by 2 is because each status occupies 2 bits.
// 	st &= ((uint8_t)3 << pos);
// 	st >>= pos;
// 	if (st == 0)
// 		return OUT;
// 	if (st == 3)
// 		return IN;
// 	return BORDER;
// }

__device__ __forceinline__ PartitionStatus gpu_show_status(uint8_t *status, uint start, int id, uint32_t offset = 0)
{
	uint8_t st = (status + start + offset)[id];
    PartitionStatus result = static_cast<PartitionStatus>((st > 1) ? 2 : st);
    return result;
}

__device__ __forceinline__ box gpu_get_pixel_box(int x, int y, double bx_lowx, double bx_lowy, double step_x, double step_y)
{
	const double start_x = bx_lowx;
	const double start_y = bx_lowy;

	double lowx = start_x + x * step_x;
	double lowy = start_y + y * step_y;
	double highx = start_x + (x + 1) * step_x;
	double highy = start_y + (y + 1) * step_y;

	return box(lowx, lowy, highx, highy);
}

__device__ __forceinline__ double degreesToRadians(double degrees)
{
	return degrees * M_PI / 180.0;
}

__device__ __forceinline__ double haversine(double lon1, double lat1, double lon2, double lat2)
{
	// 将经纬度从度转换为弧度
	lon1 = degreesToRadians(lon1);
	lat1 = degreesToRadians(lat1);
	lon2 = degreesToRadians(lon2);
	lat2 = degreesToRadians(lat2);

	// 差值
	double dlon = lon2 - lon1;
	double dlat = lat2 - lat1;

	// 哈弗赛因公式
	double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
			   std::cos(lat1) * std::cos(lat2) *
				   std::sin(dlon / 2) * std::sin(dlon / 2);
	double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));

	// 计算距离
	double distance = EARTH_RADIUS_KM * c;

	return distance;
}

// distance related

__device__ __forceinline__ float gpu_degree_per_kilometer_longitude(float latitude, float *degree_per_kilometer_longitude_arr){
	float absla = abs(latitude);
	assert(absla<=90.0);
	if(absla==90.0){
		absla = 89.9;
	}
	return degree_per_kilometer_longitude_arr[(int)(absla*10.0)];
}


// point to box max distance
__device__ __forceinline__ float gpu_max_distance(Point &p, box &bx, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
	float dx = fmax(abs(p.x - bx.low[0]), abs(p.x - bx.high[0]));
	float dy = fmax(abs(p.y - bx.low[1]), abs(p.y - bx.high[1]));

	dx = dx / gpu_degree_per_kilometer_longitude(p.y, degree_per_kilometer_longitude_arr);
	dy = dy / *degree_per_kilometer_latitude;

	return sqrt(dx*dx+dy*dy);
}

// point to box min distance
__device__ __forceinline__ double gpu_distance(box &bx, Point &p, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
	if(p.x >= bx.low[0] && p.x <= bx.high[0]&&
	   p.y >= bx.low[1] && p.y <= bx.high[1])
	{
		return 0.0;
    }

	float dx = fmax(abs(p.x - (bx.low[0] + bx.high[0]) / 2) - (bx.high[0] - bx.low[0]) / 2, 0.0);
	double dy = fmax(abs(p.y - (bx.low[1] + bx.high[1]) / 2) - (bx.high[1] - bx.low[1]) / 2, 0.0);

	dx = dx / gpu_degree_per_kilometer_longitude(p.y, degree_per_kilometer_longitude_arr);
	dy = dy / *degree_per_kilometer_latitude;

	return sqrt(dx*dx+dy*dy);
}

__device__ __forceinline__ float gpu_max_distance(box &s_box, box &t_box, float *degree_per_kilometer_latitude, float  *degree_per_kilometer_longitude_arr)
{
	float dx = fmax(s_box.high[0], t_box.high[0]) - fmin(s_box.low[0], t_box.low[0]);
	float dy = fmax(s_box.high[1], t_box.high[1]) - fmin(s_box.low[1], t_box.low[1]);;

	dx = dx / gpu_degree_per_kilometer_longitude(s_box.low[1], degree_per_kilometer_longitude_arr);
	dy = dy / *degree_per_kilometer_latitude;

	return sqrt(dx*dx+dy*dy);
}

// box to box
__device__ __forceinline__ float gpu_distance(box &s, box &t, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    const bool overlap_x = !(t.low[0] > s.high[0] || t.high[0] < s.low[0]);
    const bool overlap_y = !(t.low[1] > s.high[1] || t.high[1] < s.low[1]);
    
    const bool boxes_intersect = overlap_x && overlap_y;
    
    // For x-axis
    const float dx_case1 = t.low[0] - s.high[0];  // when t is to the right of s
    const float dx_case2 = s.low[0] - t.high[0];  // when t is to the left of s
    float dx = (t.low[0] > s.high[0]) * dx_case1 + 
               (t.high[0] < s.low[0]) * dx_case2;
    
    // For y-axis
    const float dy_case1 = t.low[1] - s.high[1];  // when t is above s
    const float dy_case2 = s.low[1] - t.high[1];  // when t is below s
    float dy = (t.low[1] > s.high[1]) * dy_case1 + 
               (t.high[1] < s.low[1]) * dy_case2;
    
    const float longitude_factor = gpu_degree_per_kilometer_longitude(s.low[1], degree_per_kilometer_longitude_arr);
    dx = dx / longitude_factor;
    dy = dy / *degree_per_kilometer_latitude;
    
    return boxes_intersect ? 0.0f : sqrt(dx * dx + dy * dy);
}

// point to point
__device__ __forceinline__ double gpu_point_to_point_distance(const Point *p1, const Point *p2)
{
	return haversine(p1->x, p1->y, p2->x, p2->y);
}

__device__ __forceinline__ double gpu_point_to_segment_distance(const Point &p, const Point &p1, const Point &p2, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    double A = p.x - p1.x;
    double B = p.y - p1.y;
    double C = p2.x - p1.x;
    double D = p2.y - p1.y;

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    
    double epsilon = 1e-10;
    len_sq = max(len_sq, epsilon);
    
    double param = dot / len_sq;
    
    param = max(0.0, min(1.0, param));
    
    double xx = p1.x + param * C;
    double yy = p1.y + param * D;
    
    double dx = p.x - xx;
    double dy = p.y - yy;
    dx = dx / gpu_degree_per_kilometer_longitude(p.y, degree_per_kilometer_longitude_arr);
    dy = dy / *degree_per_kilometer_latitude;

    return sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ double gpu_point_to_segment_distance(const double &p_x, const double &p_y, const double &p1_x, const double &p1_y, const double &p2_x, const double &p2_y)
{
	double A = p_x - p1_x;
	double B = p_y - p1_y;
	double C = p2_x - p1_x;
	double D = p2_y - p1_y;

	double dot = A * C + B * D;
	double len_sq = C * C + D * D;
	double param = -1;
	if (len_sq != 0) // in case of 0 length line
		param = dot / len_sq;

	double xx, yy;

	if (param < 0)
	{
		xx = p1_x;
		yy = p1_y;
	}
	else if (param > 1)
	{
		xx = p2_x;
		yy = p2_y;
	}
	else
	{
		xx = p1_x + param * C;
		yy = p1_y + param * D;
	}

	return haversine(p_x, p_y, xx, yy);
}

__device__ __forceinline__ double gpu_segment_to_segment_distance(Point &s1, Point &e1, double s2_x, double s2_y, double e2_x, double e2_y)
{
	double dist1 = gpu_point_to_segment_distance(s1.x, s1.y, s2_x, s2_y, e2_x, e2_y);
	double dist2 = gpu_point_to_segment_distance(e1.x, e1.y, s2_x, s2_y, e2_x, e2_y);
	double dist3 = gpu_point_to_segment_distance(s2_x, s2_y, s1.x, s1.y, e1.x, e1.y);
	double dist4 = gpu_point_to_segment_distance(e2_x, e2_y, s1.x, s1.y, e1.x, e1.y);
	return min(dist1, min(dist2, min(dist3, dist4)));
}

__device__ __forceinline__ double gpu_box_to_segment_distance(box &bx, Point &p1, Point &p2)
{

	double dist1 = gpu_segment_to_segment_distance(p1, p2, bx.low[0], bx.low[1], bx.high[0], bx.low[1]);
	double dist2 = gpu_segment_to_segment_distance(p1, p2, bx.high[0], bx.low[1], bx.high[0], bx.high[1]);
	double dist3 = gpu_segment_to_segment_distance(p1, p2, bx.high[0], bx.high[1], bx.low[0], bx.high[1]);
	double dist4 = gpu_segment_to_segment_distance(p1, p2, bx.low[0], bx.high[1], bx.low[0], bx.low[1]);

	return min(dist1, min(dist2, min(dist3, dist4)));
}

__device__ __forceinline__ double gpu_get_step(box &bx, int dimx, int dimy)
{
	Point a(bx.low[0], bx.low[1]);
	Point b(bx.high[0], bx.low[1]);
	Point c(bx.low[0], bx.high[1]);

	return min(haversine(a.x, a.y, b.x, b.y) / dimx, haversine(a.x, a.y, c.x, c.y) / dimy);
}

__device__ __forceinline__ double gpu_point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    double mindist = DBL_MAX;
    
    // Process all segments without early termination
    for (int i = 0; i < seq_len - 1; i++)
    {
        double dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
        mindist = min(mindist, dist);
    }

    return mindist;
}

// __device__ __forceinline__ double gpu_point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len, double *degree_per_kilometer_latitude, double *degree_per_kilometer_longitude_arr)
// {
//     double mindist = DBL_MAX;
    
//     // Phase 1: Check a few segments to see if we can terminate early
//     // This is a performance optimization for common cases
//     const int EARLY_CHECK_COUNT = 4;  // Adjust based on your data characteristics
//     for (int i = 0; i < min((int)seq_len - 1, EARLY_CHECK_COUNT); i++)
//     {
//         double dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
//         mindist = min(mindist, dist);
//     }
    
//     // Use a warp-level vote to see if all threads can terminate early
//     if (__all_sync(__activemask(), mindist <= WITHIN_DISTANCE))
//     {
//         return mindist;
//     }
    
//     // Phase 2: Process remaining segments
//     for (int i = EARLY_CHECK_COUNT; i < seq_len - 1; i++)
//     {
//         double dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
//         mindist = min(mindist, dist);
//     }
    
//     return mindist;
// }

__device__ __forceinline__ double gpu_segment_to_segment_within_batch(Point *vs1, Point *vs2, size_t s1, size_t s2, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
	double mindist = DBL_MAX;
	double dist;
	for (int i = 0; i < s1; i++)
	{
		dist = gpu_point_to_segment_within_batch(vs1[i], vs2, s2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
		if (dist < mindist)
		{
			mindist = dist;
		}
	}
	for (int i = 0; i < s2; i++)
	{
		dist = gpu_point_to_segment_within_batch(vs2[i], vs1, s1, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
		if (dist < mindist)
		{
			mindist = dist;
		}
	}
	return mindist;
}

// intersection related

__device__ inline int gpu_sgn(const double& x) {
    return (x > 0) - (x < 0);
}

__device__ inline bool gpu_inter1(double a, double b, double c, double d)
{
   
    double min_ab = fmin(a, b); 
    double max_ab = fmax(a, b); 
    double min_cd = fmin(c, d); 
    double max_cd = fmax(c, d); 

    return fmax(min_ab, min_cd) <= fmin(max_ab, max_cd);
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

__device__ inline bool gpu_segment_intersect(const Point& a, const Point& b, const Point& c, const Point& d) {
    if (c.cross(a, d) == 0 && c.cross(b, d) == 0)
        return gpu_inter1(a.x, b.x, c.x, d.x) && gpu_inter1(a.y, b.y, c.y, d.y);
    return gpu_sgn(a.cross(b, c)) != gpu_sgn(a.cross(b, d)) && gpu_sgn(c.cross(d, a)) != gpu_sgn(c.cross(d, b));
}

// __device__ inline bool gpu_segment_intersect(const Point& a, const Point& b, const Point& c, const Point& d) {
// 	double cad = c.cross(a, d);
// 	double cbd = c.cross(b, d);
// 	bool is_collinear = (cad == 0) && (cbd == 0);

// 	bool overlap_x = gpu_inter1(a.x, b.x, c.x, d.x);
// 	bool overlap_y = gpu_inter1(a.y, b.y, c.y, d.y);
// 	bool collinear_result = overlap_x && overlap_y;

// 	int sgn_abc = gpu_sgn(a.cross(b, c));
// 	int sgn_abd = gpu_sgn(a.cross(b, d));
// 	int sgn_cda = gpu_sgn(c.cross(d, a));
// 	int sgn_cdb = gpu_sgn(c.cross(d, b));
// 	bool non_collinear_result = (sgn_abc != sgn_abd) && (sgn_cda != sgn_cdb);
	
// 	return (is_collinear && collinear_result) || (!is_collinear && non_collinear_result);
// }

__device__ inline bool gpu_segment_intersect_batch(Point *p1, Point *p2, int s1, int s2)
{
    bool has_intersection = false;

    for (int i = 0; i < s1; i++) {
        for (int j = 0; j < s2; j++) {
            bool intersects = gpu_segment_intersect(p1[i], p1[i + 1], p2[j], p2[j + 1]);
            has_intersection = has_intersection || intersects;
        }
    }

    return has_intersection;
}