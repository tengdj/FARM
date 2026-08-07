#pragma once

#include "cuda_util.h"
#include "farm.h"

#define BLOCK_SIZE 256
#define MAX_SIZE 16

struct PixPair
{
	int pixid_a = 0;
	int pixid_b = 0;
	int pair_id = 0;
};

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    // Segment counts; each task range contains length + 1 vertices.
    uint s_length = 0;
    uint t_length = 0;
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

__device__ __forceinline__ uint8_t gpu_get_fullness(
	const uint8_t *status, uint32_t start, int id, uint8_t bitwidth,
	uint32_t byte_offset = 0)
{
	return read_packed_status(status + start + byte_offset,
		static_cast<uint32_t>(id), bitwidth);
}

__device__ __forceinline__ PartitionStatus gpu_show_status(
	const uint8_t *status, uint32_t start, int id, uint8_t bitwidth,
	uint32_t byte_offset = 0)
{
	const uint8_t st = gpu_get_fullness(
		status, start, id, bitwidth, byte_offset);
	return static_cast<PartitionStatus>(
		(st > 0) + (st >= status_max_value(bitwidth)));
}

__device__ __forceinline__ box gpu_get_pixel_box(int x, int y, double bx_lowx, double bx_lowy, double step_x, double step_y)
{
	double lowx = bx_lowx + x * step_x;
	double lowy = bx_lowy + y * step_y;
	double highx = bx_lowx + (x + 1) * step_x;
	double highy = bx_lowy + (y + 1) * step_y;

	return box(lowx, lowy, highx, highy);
}

struct FloatBox
{
    float low_x;
    float low_y;
    float high_x;
    float high_y;
};

__device__ __forceinline__ FloatBox make_float_pixel_box(
    int x, int y, float origin_x, float origin_y, float step_x, float step_y)
{
    const float low_x = fmaf(static_cast<float>(x), step_x, origin_x);
    const float low_y = fmaf(static_cast<float>(y), step_y, origin_y);
    return {low_x, low_y, low_x + step_x, low_y + step_y};
}

__device__ __forceinline__ float float_box_min_distance_sq(
    const FloatBox &source, const FloatBox &target, float inv_latitude, float inv_longitude)
{
    float dx = fmaxf(fmaxf(target.low_x - source.high_x, source.low_x - target.high_x), 0.0f);
    float dy = fmaxf(fmaxf(target.low_y - source.high_y, source.low_y - target.high_y), 0.0f);
    dx *= inv_longitude;
    dy *= inv_latitude;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ float float_box_max_distance_sq(
    const FloatBox &source, const FloatBox &target, float inv_latitude, float inv_longitude)
{
    float dx = (fmaxf(source.high_x, target.high_x) - fminf(source.low_x, target.low_x)) * inv_longitude;
    float dy = (fmaxf(source.high_y, target.high_y) - fminf(source.low_y, target.low_y)) * inv_latitude;
    return fmaf(dx, dx, dy * dy);
}

// distance related

__device__ __forceinline__ float gpu_degree_per_kilometer_longitude(
	float latitude, const float *__restrict__ degree_per_kilometer_longitude_arr){
	float absla = fabsf(latitude);
	// assert(absla<=90.0);
	if(absla >= 90.0f){
		absla = 89.9f;
	}
	return degree_per_kilometer_longitude_arr[static_cast<int>(absla * 10.0f)];
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

__device__ __forceinline__ float gpu_point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr, float within_distance)
{
	float mindist = FLT_MAX;

	for (int i = 0; i < seq_len - 1; i++)
	{
		float dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
		mindist = min(mindist, dist);
		if (mindist <= within_distance)
		{
			return mindist;
		}
	}

    return mindist;
}

__device__ __forceinline__ bool gpu_segment_intersect(Point a, Point b, Point c, Point d);

__device__ __forceinline__ float gpu_segment_to_segment_within_batch(Point *vs1, Point*vs2, size_t s1, size_t s2, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr, float within_distance)
{
	float mindist = FLT_MAX;
	if (s1 < 2 || s2 < 2)
	{
		return mindist;
	}
	for (size_t i = 0; i + 1 < s1; i++)
	{
		for (size_t j = 0; j + 1 < s2; j++)
		{
			if (gpu_segment_intersect(vs1[i], vs1[i + 1], vs2[j], vs2[j + 1]))
			{
				return 0.0f;
			}
			float dist = gpu_point_to_segment_distance(vs1[i], vs2[j], vs2[j + 1], degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
			if (dist < mindist)
			{
				mindist = dist;
			}
			if (mindist <= within_distance)
			{
				return mindist;
			}
		}
	}
	float dist = gpu_point_to_segment_within_batch(vs1[s1 - 1], vs2, s2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr, within_distance);
	if (dist <= within_distance)
	{
		return dist;
	}
	if (dist < mindist)
	{
		mindist = dist;
	}
	for (size_t i = 0; i < s2; i++)
	{
		dist = gpu_point_to_segment_within_batch(vs2[i], vs1, s1, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr, within_distance);
		if (dist <= within_distance)
		{
			return dist;
		}
		if (dist < mindist)
		{
			mindist = dist;
		}
	}
	return mindist;
}

// intersection related

__device__ __forceinline__ float cross_product(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ __forceinline__ bool is_on_segment(Point p, Point a, Point b) {
    return p.x >= fminf(a.x, b.x) && p.x <= fmaxf(a.x, b.x) &&
           p.y >= fminf(a.y, b.y) && p.y <= fmaxf(a.y, b.y);
}

__device__ __forceinline__ bool gpu_segment_intersect(Point a, Point b, Point c, Point d) {
    if (fmaxf(a.x, b.x) < fminf(c.x, d.x) || fmaxf(c.x, d.x) < fminf(a.x, b.x) ||
        fmaxf(a.y, b.y) < fminf(c.y, d.y) || fmaxf(c.y, d.y) < fminf(a.y, b.y)) {
        return false;
    }

    float cp1 = cross_product(a, b, c);
    float cp2 = cross_product(a, b, d);
    float cp3 = cross_product(c, d, a);
    float cp4 = cross_product(c, d, b);

    if (((cp1 > 0 && cp2 < 0) || (cp1 < 0 && cp2 > 0)) &&
        ((cp3 > 0 && cp4 < 0) || (cp3 < 0 && cp4 > 0))) {
        return true;
    }

    if (cp1 == 0 && is_on_segment(c, a, b)) return true;
    if (cp2 == 0 && is_on_segment(d, a, b)) return true;
    if (cp3 == 0 && is_on_segment(a, c, d)) return true;
    if (cp4 == 0 && is_on_segment(b, c, d)) return true;

    return false;
}

__device__ inline double gpu_decode_fullness(
	uint8_t fullness, double pixelArea, uint8_t bitwidth, bool isLow)
{
	const int category_count = status_category_count(bitwidth);
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
        return (double)(fullness - (uint8_t)isLow) / (category_count - 2) * pixelArea;
    }
}

__device__ inline uint8_t gpu_encode_fullness(
	double area1, double pixelArea1, double area2, double pixelArea2,
	uint8_t bitwidth){
	const int count = status_category_count(bitwidth);
	double ratio = (area1 + area2) / (pixelArea1 + pixelArea2);
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
		idx = count - 1; 

	assert(idx < 256);
	return idx;
}
