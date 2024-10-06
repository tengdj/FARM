#pragma once

#include "cuda_util.h"
#include "Ideal.h"

#define BLOCK_SIZE 1024
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

struct PolygonPair
{
	IdealOffset source;
	IdealOffset target;
	uint s_level = 0;
	uint t_level = 0;
};

struct PointPolygonPair
{
	IdealOffset source;
	Point target;
	uint level = 0;
};

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

__device__ __forceinline__ double atomicMaxDouble(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(__longlong_as_double(assumed), val)));

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
	if (abs(1.0 * (vi + 1) - val) < 0.00000001)
	{
		vi++;
	}
	return vi;
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

__device__ __forceinline__ PartitionStatus gpu_show_status(uint8_t *status, uint &start, int &id, uint16_t offset = 0)
{
	uint8_t st = (status + start + offset)[id];
	if(st == 0) return OUT;
	else if(st == 1) return BORDER;
	else return IN;
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

__device__ __forceinline__ int gpu_get_pixel_id(Point &p, box &s_mbr, double step_x, double step_y, int dimx, int dimy)
{
	int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, step_x, dimx);
	int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, step_y, dimy);
	assert(xoff <= dimx);
	assert(yoff <= dimy);
	return gpu_get_id(xoff, yoff, dimx);
}

__device__ __forceinline__ int gpu_get_closest_pixel(int pixx, int pixy, int dimx, int dimy)
{
	if (pixx < 0)
	{
		pixx = 0;
	}
	if (pixx > dimx)
	{
		pixx = dimx;
	}
	if (pixy < 0)
	{
		pixy = 0;
	}
	if (pixy > dimy)
	{
		pixy = dimy;
	}
	return gpu_get_id(pixx, pixy, dimx);
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

__device__ __forceinline__ double gpu_max_distance(Point &p, box &bx)
{
	Point q;
	q.x = (abs(p.x - bx.low[0]) < abs(p.x - bx.high[0])) ? bx.high[0] : bx.low[0];
	q.y = (abs(p.y - bx.low[1]) < abs(p.y - bx.high[1])) ? bx.high[1] : bx.low[1];

	return haversine(p.x, p.y, q.x, q.y);
}

__device__ __forceinline__ double gpu_max_distance(box &s_box, box &t_box)
{
	Point p, q;
	p.x = min(s_box.low[0], t_box.low[0]);
	p.y = min(s_box.low[1], t_box.low[1]);
	q.x = max(s_box.high[0], t_box.high[0]);
	q.y = max(s_box.high[1], t_box.high[1]);

	return haversine(p.x, p.y, q.x, q.y);
}

// point to box
__device__ __forceinline__ double gpu_distance(box &s, Point &t)
{
	if(t.x >= s.low[0] && t.x <= s.high[0]&&
	   t.y >= s.low[1] && t.y <= s.high[1])
	{
		return 0.0;
    }
	Point p;

	if (t.y > s.high[1])
	{
		p.y = s.high[1];
	}
	else if (s.low[1] > t.y)
	{
		p.y = s.low[1];
	}
	else
	{
		p.y = t.y;
	}

	if (t.x > s.high[0])
	{
		p.x = s.high[0];
	}
	else if (s.low[0] > t.x)
	{
		p.x = s.low[0];
	}
	else
	{
		p.x = t.x;
	}

	return haversine(p.x, p.y, t.x, t.y);
}

// box to box
__device__ __forceinline__ double gpu_distance(box &s, box &t)
{
	// s intersect t
	if (!(t.low[0] > s.high[0] || t.high[0] < s.low[0] ||
		  t.low[1] > s.high[1] || t.high[1] < s.low[1]))
	{
		return 0.0;
	}

	Point p, q;

	if (t.low[1] > s.high[1])
	{
		q.y = t.low[1];
		p.y = s.high[1];
	}
	else if (s.low[1] > t.high[1])
	{
		q.y = s.low[1];
		p.y = t.high[1];
	}
	else
	{
		q.y = s.low[1];
		p.y = s.low[1];
	}

	if (t.low[0] > s.high[0])
	{
		q.x = t.low[0];
		p.x = s.high[0];
	}
	else if (s.low[0] > t.high[0])
	{
		q.x = s.low[0];
		p.x = t.high[0];
	}
	else
	{
		q.x = s.low[0];
		p.x = s.low[0];
	}

	return haversine(p.x, p.y, q.x, q.y);
}

__device__ __forceinline__ double gpu_point_to_segment_distance(const Point &p, const Point &p1, const Point &p2)
{
	double A = p.x - p1.x;
	double B = p.y - p1.y;
	double C = p2.x - p1.x;
	double D = p2.y - p1.y;

	double dot = A * C + B * D;
	double len_sq = C * C + D * D;
	double param = -1;
	if (len_sq != 0) // in case of 0 length line
		param = dot / len_sq;

	double xx, yy;

	if (param < 0)
	{
		xx = p1.x;
		yy = p1.y;
	}
	else if (param > 1)
	{
		xx = p2.x;
		yy = p2.y;
	}
	else
	{
		xx = p1.x + param * C;
		yy = p1.y + param * D;
	}

	return haversine(p.x, p.y, xx, yy);
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

__device__ __forceinline__ double gpu_segment_to_segment_distance(Point &s1, Point &e1, Point s2, Point e2)
{
	double dist1 = gpu_point_to_segment_distance(s1, s2, e2);
	double dist2 = gpu_point_to_segment_distance(e1, s2, e2);
	double dist3 = gpu_point_to_segment_distance(s2, s1, e1);
	double dist4 = gpu_point_to_segment_distance(e2, s1, e1);
	return min(dist1, min(dist2, min(dist3, dist4)));
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

__device__ __forceinline__ double gpu_point_to_segment_within_batch(Point &p, Point *vs, size_t seq_len)
{
	double mindist = DBL_MAX;
	for (int i = 0; i < seq_len - 1; i++)
	{
		double dist = gpu_point_to_segment_distance(p, vs[i], vs[i + 1]);
		if (dist < mindist)
		{
			mindist = dist;
		}
		if (mindist <= WITHIN_DISTANCE)
		{
			return mindist;
		}
	}
	return mindist;
}

__device__ __forceinline__ double gpu_segment_to_segment_within_batch(Point *vs1, Point *vs2, size_t s1, size_t s2)
{
	double mindist = DBL_MAX;
	for (int i = 0; i < s1; i++)
	{
		double dist = gpu_point_to_segment_within_batch(vs1[i], vs2, s2);
		if (dist < mindist)
		{
			mindist = dist;
		}
		if (mindist <= WITHIN_DISTANCE)
		{
			return mindist;
		}
	}
	for (int i = 0; i < s2; i++)
	{
		double dist = gpu_point_to_segment_within_batch(vs2[i], vs1, s1);
		if (dist < mindist)
		{
			mindist = dist;
		}
		if (mindist <= WITHIN_DISTANCE)
		{
			return mindist;
		}
	}
	return mindist;
}