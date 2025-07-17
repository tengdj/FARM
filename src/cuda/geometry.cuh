#pragma once

#include "cuda_util.h"
#include "Ideal.h"

#define BLOCK_SIZE 256
#define WITHIN_DISTANCE 10

const float EARTH_RADIUS_KM = 6371.0;

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

__device__ __forceinline__ int gpu_get_id(int x, int y, int dimx)
{
	return y * dimx + x;
}

// from id to pixel x
__device__ __forceinline__ int gpu_get_x(int id, int dimx)
{
	return id % dimx;
}

// from id to pixel y
__device__ __forceinline__ int gpu_get_y(int id, int dimx, int dimy)
{
	assert((id / dimx) < dimy);
	return id / dimx;
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
	return min(max(x, 0), dimx - 1);
}

__device__ __forceinline__ int gpu_get_offset_y(double s_yval, double t_yval, double step_y, int dimy)
{
	int y = gpu_double_to_int((t_yval - s_yval) / step_y);
	return min(max(y, 0), dimy - 1);
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

__device__ __forceinline__ PartitionStatus gpu_show_status(uint8_t *status, uint start, int id, uint8_t category_count, uint32_t offset = 0)
{
	uint8_t st = (status + start + offset)[id];
	return (PartitionStatus)((st > 0) + (st >= category_count - 1));
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

// distance related

__device__ __forceinline__ float gpu_degree_per_kilometer_longitude(float latitude, float *degree_per_kilometer_longitude_arr){
	float absla = abs(latitude);
	// assert(absla<=90.0);
	if(absla >= 90.0){
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
__device__ __forceinline__ float gpu_distance(box &bx, Point &p, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
	if(p.x >= bx.low[0] && p.x <= bx.high[0]&&
	   p.y >= bx.low[1] && p.y <= bx.high[1])
	{
		return 0.0;
    }

	float dx = fmax(abs(p.x - (bx.low[0] + bx.high[0]) / 2) - (bx.high[0] - bx.low[0]) / 2, 0.0);
	float dy = fmax(abs(p.y - (bx.low[1] + bx.high[1]) / 2) - (bx.high[1] - bx.low[1]) / 2, 0.0);

	dx = dx / gpu_degree_per_kilometer_longitude(p.y, degree_per_kilometer_longitude_arr);
	dy = dy / *degree_per_kilometer_latitude;

	return sqrt(dx*dx+dy*dy);
}

__device__ __forceinline__ float gpu_max_distance(box &s_box, box &t_box, float *degree_per_kilometer_latitude, float  *degree_per_kilometer_longitude_arr)
{
	float dx = fmax(s_box.high[0], t_box.high[0]) - fmin(s_box.low[0], t_box.low[0]);
	float dy = fmax(s_box.high[1], t_box.high[1]) - fmin(s_box.low[1], t_box.low[1]);

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

__device__ __forceinline__ float gpu_point_to_segment_distance(const Point &p, const Point &p1, const Point &p2, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    float A = p.x - p1.x;
    float B = p.y - p1.y;
    float C = p2.x - p1.x;
    float D = p2.y - p1.y;

    float dot = A * C + B * D;
    float len_sq = C * C + D * D;
    
    float epsilon = 1e-10;
    len_sq = max(len_sq, epsilon);
    
    float param = dot / len_sq;
    
    param = max(0.0, min(1.0, param));
    
    float xx = p1.x + param * C;
    float yy = p1.y + param * D;
    
    float dx = p.x - xx;
    float dy = p.y - yy;
    dx = dx / gpu_degree_per_kilometer_longitude(p.y, degree_per_kilometer_longitude_arr);
    dy = dy / *degree_per_kilometer_latitude;

    return sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ float gpu_point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    float mindist = DBL_MAX;
    
    // Process all segments without early termination
    for (int i = 0; i < seq_len - 1; i++)
    {
        float dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
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

__device__ __forceinline__ float gpu_segment_to_segment_within_batch(Point *vs1, Point*vs2, size_t s1, size_t s2, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
	float mindist = FLT_MAX;
	float dist;
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

__device__ inline int gpu_sgn(const float& x) {
    return (x > 0) - (x < 0);
}

__device__ inline bool gpu_inter1(float a, float b, float c, float d)
{
   
    float min_ab = fmin(a, b); 
    float max_ab = fmax(a, b); 
    float min_cd = fmin(c, d); 
    float max_cd = fmax(c, d); 

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

__device__ inline void gpu_segment_intersect_batch(Point *p, int s1, int s2, int e1, int e2, int pair_id, Intersection* intersections, uint* num)
{
	for (int i = s1; i < e1; i++) {
        Point d1 = p[i + 1] - p[i];
        Point p_i = p[i];

    	for (int j = s2; j < e2; j++) {
			Point d2 = p[j + 1] - p[j];
			Point r = p[j] - p_i;

			float denom = d1.cross(d2);

			if (abs(denom) < 1e-9) continue;
			
			float inv_denom = 1.0 / denom;
			float t = r.cross(d2) * inv_denom;
			float u = r.cross(d1) * inv_denom;
			
			if (t >= -1e-9 && t <= 1 + 1e-9 && u >= -1e-9 && u <= 1 + 1e-9) {
                Point intersect_p = p_i + d1 * t;
                uint idx = atomicAdd(num, 1U);
                intersections[idx] = {intersect_p, pair_id, i, j, t, u};
			}
        }
    }
    return;
}

__device__ inline double gpu_decode_fullness(uint8_t fullness, double pixelArea, int category_count, bool isLow)
{
	if (fullness == 0)
    {
        return 0.0f;
    }
    else if (fullness == category_count - 1)
    {
        return pixelArea;
    }
    else
    {
        return (1.0 * fullness - isLow) / (category_count - 2) * pixelArea;
    }
}

__device__ inline uint8_t gpu_encode_fullness(double area, double pixelArea, int count){
	double ratio = area / pixelArea;
	// area calculation has precision error
	if (fabs(ratio - 1.0) < 1e-9)
	{
		// full
		return count - 1;
	}

	if (fabs(ratio) < 1e-9)
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

__device__ inline uint8_t gpu_encode_fullness(double area1, double pixelArea1, double area2, double pixelArea2, int count){
	double ratio = (area1 / pixelArea1 + area2 / pixelArea2) / 2;
	// area calculation has precision error
	if (fabs(ratio - 1.0) < 1e-9)
	{
		// full
		return count - 1;
	}

	if (fabs(ratio) < 1e-9)
	{
		// empty
		return 0;
	}

	int idx = static_cast<int>(ceil(ratio * (count - 2)));
	if (idx >= count)
		idx = count - 1; // 防止越界

	assert(idx < 256);
	return idx;
}


// __global__ void kernel_filter_segment_contain(Segment *segments, pair<uint32_t,uint32_t> *pairs,
// 											  IdealOffset *idealoffset, RasterInfo *info, 
// 											  uint8_t *status, Point *vertices,  uint size, uint8_t *flags, 
// 											  PixMapping *ptpixpairs, uint *pp_size);

// __global__ void kernel_refinement_segment_contain(PixMapping *ptpixpairs, Segment *segments, 
// 												pair<uint32_t, uint32_t> *pairs,
// 												IdealOffset *idealoffset, RasterInfo *info,
// 												uint32_t *es_offset, EdgeSeq *edge_sequences,
// 												Point *vertices, uint32_t *gridline_offset,
// 												double *gridline_nodes, uint *size, uint8_t *flags);