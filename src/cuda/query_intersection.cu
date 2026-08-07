#include "geometry.cuh"
#include "farm.h"
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ void gpu_restrict_area_to_overlap(
    double area_low, double area_high, double pixel_area, double overlap_area,
    double &overlap_low, double &overlap_high)
{
    overlap_low = fmax(0.0, area_low - (pixel_area - overlap_area));
    overlap_high = fmin(overlap_area, area_high);
    overlap_low = fmin(overlap_area, fmax(0.0, overlap_low));
    overlap_high = fmin(overlap_area, fmax(0.0, overlap_high));
    overlap_low = fmin(overlap_low, overlap_high);
}

constexpr uint32_t INTERSECTION_APPROX_THREADS = 128;
constexpr uint32_t INTERSECTION_APPROX_WARPS = INTERSECTION_APPROX_THREADS / 32;

__device__ __forceinline__ double intersection_warp_sum(double value)
{
    #pragma unroll
    for(uint32_t offset = 16; offset > 0; offset >>= 1){
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void kernel_approximate_intersection_area(
    const pair<uint32_t, uint32_t>* __restrict__ pairs,
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info,
    const uint8_t* __restrict__ status,
    size_t size,
	    uint8_t bitwidth,
    double* __restrict__ areas)
{
    const size_t pair_id = blockIdx.x;
    if(pair_id >= size) return;

    __shared__ uint32_t shared_source_idx;
    __shared__ uint32_t shared_target_idx;
    __shared__ uint32_t shared_source_status_start;
    __shared__ uint32_t shared_target_status_start;
    __shared__ double warp_low[INTERSECTION_APPROX_WARPS];
    __shared__ double warp_high[INTERSECTION_APPROX_WARPS];
    if(threadIdx.x == 0){
        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        shared_source_idx = info[pair.first].step_x >= info[pair.second].step_x
            ? pair.first : pair.second;
        shared_target_idx = shared_source_idx == pair.first ? pair.second : pair.first;
        shared_source_status_start = farm_offset[shared_source_idx].status_start;
        shared_target_status_start = farm_offset[shared_target_idx].status_start;
    }
    __syncthreads();

    const uint32_t source_idx = shared_source_idx;
    const uint32_t target_idx = shared_target_idx;
    const RasterInfo source_info = info[source_idx];
    const RasterInfo target_info = info[target_idx];
    const uint32_t source_status_start = shared_source_status_start;
    const uint32_t target_status_start = shared_target_status_start;
    const double source_pixel_area = source_info.step_x * source_info.step_y;
    const double target_pixel_area = target_info.step_x * target_info.step_y;

    const int source_start_x = gpu_get_offset_x(source_info.mbr.low[0],
        target_info.mbr.low[0] + 1e-6, source_info.step_x, source_info.dimx);
    const int source_end_x = gpu_get_offset_x(source_info.mbr.low[0],
        target_info.mbr.high[0] - 1e-6, source_info.step_x, source_info.dimx);
    const int source_start_y = gpu_get_offset_y(source_info.mbr.low[1],
        target_info.mbr.low[1] + 1e-6, source_info.step_y, source_info.dimy);
    const int source_end_y = gpu_get_offset_y(source_info.mbr.low[1],
        target_info.mbr.high[1] - 1e-6, source_info.step_y, source_info.dimy);

    double total_low = 0.0;
    double total_high = 0.0;
    const int source_width = source_end_x >= source_start_x
        ? source_end_x - source_start_x + 1 : 0;
    const int source_height = source_end_y >= source_start_y
        ? source_end_y - source_start_y + 1 : 0;
    const int source_count = source_width * source_height;
    for(int source_linear = threadIdx.x; source_linear < source_count;
        source_linear += blockDim.x){
            const int source_x = source_start_x + source_linear % source_width;
            const int source_y = source_start_y + source_linear / source_width;
            const int source_id = gpu_get_id(source_x, source_y, source_info.dimx);
	            const uint8_t source_fullness = gpu_get_fullness(
					status, source_status_start, source_id, bitwidth);
            if(source_fullness == 0) continue;

            const box source_box = gpu_get_pixel_box(source_x, source_y,
                source_info.mbr.low[0], source_info.mbr.low[1],
                source_info.step_x, source_info.step_y);
            const int target_start_x = gpu_get_offset_x(target_info.mbr.low[0],
                source_box.low[0] + 1e-6, target_info.step_x, target_info.dimx);
            const int target_end_x = gpu_get_offset_x(target_info.mbr.low[0],
                source_box.high[0] - 1e-6, target_info.step_x, target_info.dimx);
            const int target_start_y = gpu_get_offset_y(target_info.mbr.low[1],
                source_box.low[1] + 1e-6, target_info.step_y, target_info.dimy);
            const int target_end_y = gpu_get_offset_y(target_info.mbr.low[1],
                source_box.high[1] - 1e-6, target_info.step_y, target_info.dimy);
            const double source_low = gpu_decode_fullness(source_fullness,
	                source_pixel_area, bitwidth, true);
            const double source_high = gpu_decode_fullness(source_fullness,
	                source_pixel_area, bitwidth, false);

            for(int target_x = target_start_x; target_x <= target_end_x; target_x++){
                for(int target_y = target_start_y; target_y <= target_end_y; target_y++){
                    const int target_id = gpu_get_id(target_x, target_y, target_info.dimx);
	                    const uint8_t target_fullness = gpu_get_fullness(
							status, target_status_start, target_id, bitwidth);
                    if(target_fullness == 0) continue;

                    const box target_box = gpu_get_pixel_box(target_x, target_y,
                        target_info.mbr.low[0], target_info.mbr.low[1],
                        target_info.step_x, target_info.step_y);
                    const double overlap_width = fmin(source_box.high[0], target_box.high[0])
                        - fmax(source_box.low[0], target_box.low[0]);
                    const double overlap_height = fmin(source_box.high[1], target_box.high[1])
                        - fmax(source_box.low[1], target_box.low[1]);
                    if(overlap_width <= 0.0 || overlap_height <= 0.0) continue;

                    const double overlap_area = overlap_width * overlap_height;
                    double source_overlap_low, source_overlap_high;
                    double target_overlap_low, target_overlap_high;
                    gpu_restrict_area_to_overlap(source_low, source_high,
                        source_pixel_area, overlap_area,
                        source_overlap_low, source_overlap_high);
                    gpu_restrict_area_to_overlap(
                        gpu_decode_fullness(target_fullness, target_pixel_area,
	                            bitwidth, true),
                        gpu_decode_fullness(target_fullness, target_pixel_area,
	                            bitwidth, false),
                        target_pixel_area, overlap_area,
                        target_overlap_low, target_overlap_high);

                    double intersection_high = fmin(source_overlap_high, target_overlap_high);
                    intersection_high = fmin(overlap_area, fmax(0.0, intersection_high));
                    double intersection_low = fmax(0.0,
                        source_overlap_low + target_overlap_low - overlap_area);
                    intersection_low = fmin(intersection_high, intersection_low);
                    total_low += intersection_low;
                    total_high += intersection_high;
                }
            }
    }

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    total_low = intersection_warp_sum(total_low);
    total_high = intersection_warp_sum(total_high);
    if(lane == 0){
        warp_low[warp] = total_low;
        warp_high[warp] = total_high;
    }
    __syncthreads();
    if(warp == 0){
        double block_low = lane < INTERSECTION_APPROX_WARPS ? warp_low[lane] : 0.0;
        double block_high = lane < INTERSECTION_APPROX_WARPS ? warp_high[lane] : 0.0;
        block_low = intersection_warp_sum(block_low);
        block_high = intersection_warp_sum(block_high);
        if(lane == 0){
    // areas[] stores the shoelace double area used by the exact path.
            areas[pair_id] = block_low + block_high;
        }
    }
}

__global__ void kernel_filter_intersection(
    const pair<uint32_t, uint32_t>* __restrict__ pairs, 
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info, 
    const uint8_t* __restrict__ status, 
    uint size, 
    PixPair* __restrict__ pixpairs, 
    uint* pp_size, 
	    uint8_t bitwidth,
    uint32_t pixpair_capacity,
    uint32_t* __restrict__ overflow)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;

    const pair<uint32_t, uint32_t> pair = pairs[x];
    const uint32_t idx_a = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
    const uint32_t idx_b = info[pair.first].step_x >= info[pair.second].step_x ? pair.second : pair.first;
    const RasterInfo info_a = info[idx_a];
    const RasterInfo info_b = info[idx_b];

    const int ratio_x = (int)(info_a.step_x / info_b.step_x + 0.5);
    const int ratio_y = (int)(info_a.step_y / info_b.step_y + 0.5);

    const int base_offset_x = __double2int_rn((info_a.mbr.low[0] - info_b.mbr.low[0]) / info_b.step_x);
    const int base_offset_y = __double2int_rn((info_a.mbr.low[1] - info_b.mbr.low[1]) / info_b.step_y);

    const uint32_t status_start_a = farm_offset[idx_a].status_start;
    const uint32_t status_start_b = farm_offset[idx_b].status_start;

    for (int i = 0; i < info_a.dimx; i++)
    {
        int t_i_start = base_offset_x + i * ratio_x;
        int t_i_end   = t_i_start + ratio_x;

        if (t_i_end <= 0 || t_i_start >= info_b.dimx) continue;
        int loop_tx_start = max(0, t_i_start);
        int loop_tx_end   = min(info_b.dimx, t_i_end);

        for (int j = 0; j < info_a.dimy; j++)
        {
            int pa = gpu_get_id(i, j, info_a.dimx);
	            PartitionStatus source_status = gpu_show_status(
					status, status_start_a, pa, bitwidth);
            
            if (source_status != BORDER) continue;

            int t_j_start = base_offset_y + j * ratio_y;
            int t_j_end   = t_j_start + ratio_y;

            if (t_j_end <= 0 || t_j_start >= info_b.dimy) continue;
            int loop_ty_start = max(0, t_j_start);
            int loop_ty_end   = min(info_b.dimy, t_j_end);

            for (int ti = loop_tx_start; ti < loop_tx_end; ti++)
            {
                for (int tj = loop_ty_start; tj < loop_ty_end; tj++)
                {
                    int pb = gpu_get_id(ti, tj, info_b.dimx);
	                    PartitionStatus target_status = gpu_show_status(
							status, status_start_b, pb, bitwidth);

					if (target_status != BORDER) continue;

					uint idx = atomicAdd(pp_size, 1U);
					if(idx < pixpair_capacity){
						pixpairs[idx] = {pa, pb, x};
					}else{
						atomicExch(overflow, 1U);
					}
                }
            }
        }
    }
}

__global__ void kernel_unroll_intersection_tasks(
	const PixPair *pixpairs, uint32_t pixpair_count,
	const pair<uint32_t, uint32_t> *pairs,
	const FarmOffset *farm_offset, const RasterInfo *info,
	const uint32_t *es_offset, const EdgeSeq *edge_sequences,
	Task *tasks, uint32_t task_capacity, uint32_t *task_count,
	uint32_t *overflow)
{
	const uint32_t pixpair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixpair_id >= pixpair_count) return;
	const PixPair pixel_pair = pixpairs[pixpair_id];
	const pair<uint32_t, uint32_t> pair = pairs[pixel_pair.pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x
		? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	const FarmOffset source_offset = farm_offset[source_idx];
	const FarmOffset target_offset = farm_offset[target_idx];
	const uint32_t source_sequence_begin = es_offset[source_offset.offset_start + pixel_pair.pixid_a];
	const uint32_t source_sequence_end = es_offset[source_offset.offset_start + pixel_pair.pixid_a + 1];
	const uint32_t target_sequence_begin = es_offset[target_offset.offset_start + pixel_pair.pixid_b];
	const uint32_t target_sequence_end = es_offset[target_offset.offset_start + pixel_pair.pixid_b + 1];

	for(uint32_t source_sequence = source_sequence_begin;
		source_sequence < source_sequence_end; source_sequence++){
		const EdgeSeq source = edge_sequences[source_offset.edge_sequences_start + source_sequence];
		if(source.length == 0) continue;
		for(uint32_t target_sequence = target_sequence_begin;
			target_sequence < target_sequence_end; target_sequence++){
			const EdgeSeq target = edge_sequences[target_offset.edge_sequences_start + target_sequence];
			if(target.length == 0) continue;
			const uint32_t source_chunks = (source.length + MAX_SIZE - 1) / MAX_SIZE;
			const uint32_t target_chunks = (target.length + MAX_SIZE - 1) / MAX_SIZE;
			const uint32_t sequence_task_count = source_chunks * target_chunks;
			const uint32_t output_begin = atomicAdd(task_count, sequence_task_count);
			if(output_begin > task_capacity || sequence_task_count > task_capacity - output_begin){
				atomicExch(overflow, 1U);
				continue;
			}

			uint32_t local_task = 0;
			for(uint32_t source_start = 0; source_start < source.length; source_start += MAX_SIZE){
				for(uint32_t target_start = 0; target_start < target.length; target_start += MAX_SIZE){
					Task &task = tasks[output_begin + local_task++];
					task.s_start = source_offset.vertices_start + source.start + source_start;
					task.t_start = target_offset.vertices_start + target.start + target_start;
					task.s_length = min((uint32_t)MAX_SIZE, source.length - source_start);
					task.t_length = min((uint32_t)MAX_SIZE, target.length - target_start);
					task.pair_id = pixel_pair.pair_id;
				}
			}
		}
	}
}

namespace {

constexpr float INTERSECTION_PARAM_EPS = 1e-6f;
constexpr float INTERSECTION_PARALLEL_EPS_SQ = 1e-12f;
constexpr uint32_t INTERSECTION_PAYLOAD_MASK = 0x3fffffffU;
constexpr uint32_t INTERSECTION_NO_OVERLAP = INTERSECTION_PAYLOAD_MASK;

struct IntersectionDeviceControl
{
	uint32_t count;
	uint32_t overflow;
	uint32_t auxiliary_count;
};

struct IntersectionStageResult
{
	uint32_t count;
	uint32_t overflow;
};

static_assert(offsetof(IntersectionDeviceControl, auxiliary_count) ==
	sizeof(IntersectionStageResult), "stage control fields must be contiguous");

struct GpuIntersection
{
	uint32_t pair_id;
	uint32_t edge_source_id;
	uint32_t edge_target_id;
	float t;
	float u;
	uint32_t metadata;
};

static_assert(sizeof(GpuIntersection) == 24, "GpuIntersection must remain compact");

__host__ __device__ __forceinline__ uint32_t gpu_intersection_overlap(const GpuIntersection &inter)
{
	const uint32_t value = inter.metadata & INTERSECTION_PAYLOAD_MASK;
	return value == INTERSECTION_NO_OVERLAP ? 0U : value;
}

__device__ __forceinline__ uint32_t gpu_intersection_metadata(uint32_t overlap_group)
{
	return overlap_group == 0 ? INTERSECTION_NO_OVERLAP : overlap_group;
}

__device__ __forceinline__ float intersection_cross(float ax, float ay, float bx, float by)
{
	return fmaf(ax, by, -ay * bx);
}

__device__ __forceinline__ void normalize_ring_parameter(
	uint32_t &edge_id, float &param, uint32_t vertices_start, uint32_t vertices_end)
{
	param = fminf(1.0f, fmaxf(0.0f, param));
	if(param <= INTERSECTION_PARAM_EPS){
		param = 0.0f;
	}else if(param >= 1.0f - INTERSECTION_PARAM_EPS){
		edge_id = edge_id + 1 < vertices_end - 1 ? edge_id + 1 : vertices_start;
		param = 0.0f;
	}
}

__device__ __forceinline__ GpuIntersection make_gpu_intersection(
	uint32_t pair_id,
	uint32_t source_edge, uint32_t target_edge, float t, float u,
	uint32_t source_vertices_start, uint32_t source_vertices_end,
	uint32_t target_vertices_start, uint32_t target_vertices_end)
{
	normalize_ring_parameter(source_edge, t, source_vertices_start, source_vertices_end);
	normalize_ring_parameter(target_edge, u, target_vertices_start, target_vertices_end);
	return {
		pair_id, source_edge, target_edge, t, u,
		gpu_intersection_metadata(0)
	};
}

__device__ __forceinline__ float segment_parameter(
	float px, float py, float qx, float qy, float dx, float dy)
{
	return fabsf(dx) >= fabsf(dy) ? (px - qx) / dx : (py - qy) / dy;
}

struct LocalGpuIntersections
{
	GpuIntersection records[2];
	uint32_t count;
	bool overlap;
};

__device__ __forceinline__ LocalGpuIntersections intersect_gpu_edge_pair(
	const Point &source_p, const Point &source_q,
	const Point &target_p, const Point &target_q,
	uint32_t source_edge, uint32_t target_edge,
	uint32_t source_vertices_start, uint32_t source_vertices_end,
	uint32_t target_vertices_start, uint32_t target_vertices_end,
	uint32_t pair_id)
{
	LocalGpuIntersections result{};
	const float source_dx = source_q.x - source_p.x;
	const float source_dy = source_q.y - source_p.y;
	const float source_len_sq = fmaf(source_dx, source_dx, source_dy * source_dy);
	if(source_len_sq == 0.0f) return result;
	if(fmaxf(source_p.x, source_q.x) + INTERSECTION_PARAM_EPS < fminf(target_p.x, target_q.x) ||
	   fminf(source_p.x, source_q.x) - INTERSECTION_PARAM_EPS > fmaxf(target_p.x, target_q.x) ||
	   fmaxf(source_p.y, source_q.y) + INTERSECTION_PARAM_EPS < fminf(target_p.y, target_q.y) ||
	   fminf(source_p.y, source_q.y) - INTERSECTION_PARAM_EPS > fmaxf(target_p.y, target_q.y)){
		return result;
	}
	const float target_dx = target_q.x - target_p.x;
	const float target_dy = target_q.y - target_p.y;
	const float target_len_sq = fmaf(target_dx, target_dx, target_dy * target_dy);
	if(target_len_sq == 0.0f) return result;
	const float denom = intersection_cross(source_dx, source_dy, target_dx, target_dy);
	const float rel_x = target_p.x - source_p.x;
	const float rel_y = target_p.y - source_p.y;
	if(denom * denom <= INTERSECTION_PARALLEL_EPS_SQ * source_len_sq * target_len_sq){
		const float rel_len_sq = fmaf(rel_x, rel_x, rel_y * rel_y);
		const float collinear = intersection_cross(rel_x, rel_y, source_dx, source_dy);
		if(collinear * collinear > INTERSECTION_PARALLEL_EPS_SQ * source_len_sq *
			fmaxf(rel_len_sq, 1.0f)) return result;
		const float t0 = fmaf(rel_x, source_dx, rel_y * source_dy) / source_len_sq;
		const float t1 = t0 + fmaf(target_dx, source_dx, target_dy * source_dy) / source_len_sq;
		const float overlap_start = fmaxf(0.0f, fminf(t0, t1));
		const float overlap_end = fminf(1.0f, fmaxf(t0, t1));
		if(overlap_end - overlap_start <= INTERSECTION_PARAM_EPS) return result;
		const float start_x = fmaf(overlap_start, source_dx, source_p.x);
		const float start_y = fmaf(overlap_start, source_dy, source_p.y);
		const float end_x = fmaf(overlap_end, source_dx, source_p.x);
		const float end_y = fmaf(overlap_end, source_dy, source_p.y);
		const float u0 = segment_parameter(start_x, start_y,
			target_p.x, target_p.y, target_dx, target_dy);
		const float u1 = segment_parameter(end_x, end_y,
			target_p.x, target_p.y, target_dx, target_dy);
		result.records[0] = make_gpu_intersection(pair_id, source_edge, target_edge,
			overlap_start, u0, source_vertices_start, source_vertices_end,
			target_vertices_start, target_vertices_end);
		result.records[1] = make_gpu_intersection(pair_id, source_edge, target_edge,
			overlap_end, u1, source_vertices_start, source_vertices_end,
			target_vertices_start, target_vertices_end);
		result.count = 2;
		result.overlap = true;
		return result;
	}
	const float inv_denom = 1.0f / denom;
	const float t = intersection_cross(rel_x, rel_y, target_dx, target_dy) * inv_denom;
	const float u = intersection_cross(rel_x, rel_y, source_dx, source_dy) * inv_denom;
	if(t < -INTERSECTION_PARAM_EPS || t > 1.0f + INTERSECTION_PARAM_EPS ||
	   u < -INTERSECTION_PARAM_EPS || u > 1.0f + INTERSECTION_PARAM_EPS) return result;
	result.records[0] = make_gpu_intersection(pair_id, source_edge, target_edge,
		t, u, source_vertices_start, source_vertices_end,
		target_vertices_start, target_vertices_end);
	result.count = 1;
	return result;
}

__device__ __forceinline__ uint32_t warp_exclusive_sum(
	uint32_t value, uint32_t lane, uint32_t &total)
{
	uint32_t inclusive = value;
	#pragma unroll
	for(uint32_t offset = 1; offset < 32; offset <<= 1){
		const uint32_t other = __shfl_up_sync(0xffffffffU, inclusive, offset);
		if(lane >= offset) inclusive += other;
	}
	total = __shfl_sync(0xffffffffU, inclusive, 31);
	return inclusive - value;
}

constexpr uint32_t INTERSECTION_WARPS_PER_BLOCK = 8;

struct SharedIntersectionPoint
{
	float x;
	float y;
};

__global__ void kernel_refinement_intersection(
	const Task *tasks, const pair<uint32_t, uint32_t> *pairs,
	const FarmOffset *farm_offset, const RasterInfo *info, const Point *vertices,
	uint32_t task_count, GpuIntersection *intersections, uint32_t intersection_capacity,
	uint32_t *intersection_count, uint32_t *overlap_counter,
	uint32_t *intersections_per_pair, uint32_t *overflow)
{
	const uint32_t lane = threadIdx.x & 31U;
	const uint32_t warp_id = threadIdx.x >> 5;
	const uint32_t task_id = blockIdx.x * INTERSECTION_WARPS_PER_BLOCK + warp_id;
	if(task_id >= task_count) return;
	Task task{};
	if(lane == 0) task = tasks[task_id];
	task.s_start = __shfl_sync(0xffffffffU, task.s_start, 0);
	task.t_start = __shfl_sync(0xffffffffU, task.t_start, 0);
	task.s_length = __shfl_sync(0xffffffffU, task.s_length, 0);
	task.t_length = __shfl_sync(0xffffffffU, task.t_length, 0);
	task.pair_id = __shfl_sync(0xffffffffU, task.pair_id, 0);
	const pair<uint32_t, uint32_t> pair = pairs[task.pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x
		? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	const uint32_t source_vertices_start = farm_offset[source_idx].vertices_start;
	const uint32_t source_vertices_end = farm_offset[source_idx + 1].vertices_start;
	const uint32_t target_vertices_start = farm_offset[target_idx].vertices_start;
	const uint32_t target_vertices_end = farm_offset[target_idx + 1].vertices_start;
	__shared__ SharedIntersectionPoint source_points[INTERSECTION_WARPS_PER_BLOCK][MAX_SIZE + 1];
	__shared__ SharedIntersectionPoint target_points[INTERSECTION_WARPS_PER_BLOCK][MAX_SIZE + 1];
	if(lane <= task.s_length){
		const Point point = vertices[task.s_start + lane];
		source_points[warp_id][lane] = {point.x, point.y};
	}
	if(lane <= task.t_length){
		const Point point = vertices[task.t_start + lane];
		target_points[warp_id][lane] = {point.x, point.y};
	}
	__syncwarp();
	uint32_t task_record_count = 0;
	const uint32_t edge_pair_count = task.s_length * task.t_length;
	const uint32_t round_count = (edge_pair_count + 31U) / 32U;
	for(uint32_t round = 0; round < round_count; round++){
		const uint32_t flat_id = round * 32U + lane;
		LocalGpuIntersections local{};
		if(flat_id < edge_pair_count){
			const uint32_t source_local = flat_id / task.t_length;
			const uint32_t target_local = flat_id - source_local * task.t_length;
			const SharedIntersectionPoint source_a = source_points[warp_id][source_local];
			const SharedIntersectionPoint source_b = source_points[warp_id][source_local + 1];
			const SharedIntersectionPoint target_a = target_points[warp_id][target_local];
			const SharedIntersectionPoint target_b = target_points[warp_id][target_local + 1];
			local = intersect_gpu_edge_pair(
				Point(source_a.x, source_a.y), Point(source_b.x, source_b.y),
				Point(target_a.x, target_a.y), Point(target_b.x, target_b.y),
				task.s_start + source_local, task.t_start + target_local,
				source_vertices_start, source_vertices_end,
				target_vertices_start, target_vertices_end, task.pair_id);
		}
		uint32_t record_total = 0;
		const uint32_t record_prefix = warp_exclusive_sum(local.count, lane, record_total);
		uint32_t overlap_total = 0;
		const uint32_t overlap_prefix = warp_exclusive_sum(local.overlap ? 1U : 0U,
			lane, overlap_total);
		uint32_t record_begin = 0;
		uint32_t overlap_begin = 0;
		uint32_t valid = 1;
		if(lane == 0){
			if(record_total > 0){
				record_begin = atomicAdd(intersection_count, record_total);
				if(record_begin > intersection_capacity ||
				   record_total > intersection_capacity - record_begin){
					atomicExch(overflow, 1U);
					valid = 0;
				}
			}
			if(overlap_total > 0){
				overlap_begin = atomicAdd(overlap_counter, overlap_total);
				if(overlap_begin + overlap_total >= INTERSECTION_NO_OVERLAP){
					atomicExch(overflow, 1U);
					valid = 0;
				}
			}
			task_record_count += record_total;
		}
		record_begin = __shfl_sync(0xffffffffU, record_begin, 0);
		overlap_begin = __shfl_sync(0xffffffffU, overlap_begin, 0);
		valid = __shfl_sync(0xffffffffU, valid, 0);
		if(valid && local.count > 0){
			if(local.overlap){
				const uint32_t group = overlap_begin + overlap_prefix + 1U;
				local.records[0].metadata = gpu_intersection_metadata(group);
				local.records[1].metadata = gpu_intersection_metadata(group);
			}
			intersections[record_begin + record_prefix] = local.records[0];
			if(local.count == 2){
				intersections[record_begin + record_prefix + 1] = local.records[1];
			}
		}
	}
	if(lane == 0 && task_record_count > 0){
		atomicAdd(intersections_per_pair + task.pair_id, task_record_count);
	}
}

constexpr uint32_t INTERSECTION_PARAM_BITS = 20;
constexpr uint32_t INTERSECTION_PARAM_LEVELS = (1U << INTERSECTION_PARAM_BITS) - 1U;
constexpr uint32_t INTERSECTION_SORT_BITS = INTERSECTION_PARAM_BITS + 32U;

struct IntersectionSortStorage
{
	uint64_t *keys_a;
	uint64_t *keys_b;
	uint32_t *indices_a;
	uint32_t *indices_b;
};

static IntersectionSortStorage make_intersection_sort_storage(void *buffer, uint32_t count)
{
	auto *bytes = reinterpret_cast<uint8_t *>(buffer);
	IntersectionSortStorage storage;
	storage.keys_a = reinterpret_cast<uint64_t *>(bytes);
	storage.keys_b = reinterpret_cast<uint64_t *>(bytes + count * sizeof(uint64_t));
	storage.indices_a = reinterpret_cast<uint32_t *>(
		bytes + count * 2ULL * sizeof(uint64_t));
	storage.indices_b = storage.indices_a + count;
	return storage;
}

__device__ __forceinline__ uint32_t quantize_intersection_parameter(float value)
{
	value = fminf(1.0f, fmaxf(0.0f, value));
	return __float2uint_rn(value * static_cast<float>(INTERSECTION_PARAM_LEVELS));
}

__device__ __forceinline__ uint64_t make_indexed_intersection_key(
	const GpuIntersection &inter, const pair<uint32_t, uint32_t> *pairs,
	const FarmOffset *farm_offset, const RasterInfo *info,
	bool source_order)
{
	const pair<uint32_t, uint32_t> pair = pairs[inter.pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x
		? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	const uint32_t ring_idx = source_order ? source_idx : target_idx;
	const uint32_t edge_id = source_order ? inter.edge_source_id : inter.edge_target_id;
	const float param = source_order ? inter.t : inter.u;
	const uint32_t ring_begin = farm_offset[ring_idx].vertices_start;
	const uint64_t local_edge = edge_id - ring_begin;
	const uint64_t position = (local_edge << INTERSECTION_PARAM_BITS)
		| quantize_intersection_parameter(param);
	return position;
}

__global__ void build_indexed_intersection_keys(
	const GpuIntersection *intersections, const uint32_t *input_indices,
	uint32_t *output_indices, uint64_t *keys, uint32_t count,
	const pair<uint32_t, uint32_t> *pairs, const FarmOffset *farm_offset,
	const RasterInfo *info, bool source_order)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= count) return;
	const uint32_t record_id = input_indices == nullptr ? id : input_indices[id];
	const GpuIntersection inter = intersections[record_id];
	keys[id] = make_indexed_intersection_key(inter, pairs, farm_offset, info,
		source_order);
	output_indices[id] = record_id;
}

__device__ __forceinline__ bool intersection_record_less(
	const GpuIntersection &a, const GpuIntersection &b, bool source_order)
{
	if(a.pair_id != b.pair_id) return a.pair_id < b.pair_id;
	const uint32_t a_primary_edge = source_order ? a.edge_source_id : a.edge_target_id;
	const uint32_t b_primary_edge = source_order ? b.edge_source_id : b.edge_target_id;
	if(a_primary_edge != b_primary_edge) return a_primary_edge < b_primary_edge;
	const float a_primary_param = source_order ? a.t : a.u;
	const float b_primary_param = source_order ? b.t : b.u;
	if(a_primary_param != b_primary_param) return a_primary_param < b_primary_param;
	const uint32_t a_secondary_edge = source_order ? a.edge_target_id : a.edge_source_id;
	const uint32_t b_secondary_edge = source_order ? b.edge_target_id : b.edge_source_id;
	if(a_secondary_edge != b_secondary_edge) return a_secondary_edge < b_secondary_edge;
	const float a_secondary_param = source_order ? a.u : a.t;
	const float b_secondary_param = source_order ? b.u : b.t;
	if(a_secondary_param != b_secondary_param) return a_secondary_param < b_secondary_param;
	return a.metadata < b.metadata;
}

constexpr uint32_t INTERSECTION_LOCAL_SORT_THREADS = 256;
constexpr uint32_t INTERSECTION_LOCAL_SORT_ITEMS = 2;
constexpr uint32_t INTERSECTION_LOCAL_SORT_CAPACITY =
	INTERSECTION_LOCAL_SORT_THREADS * INTERSECTION_LOCAL_SORT_ITEMS;

__global__ void finish_intersection_pair_offsets(
	uint32_t *offsets, uint32_t pair_count, uint32_t record_count)
{
	if(blockIdx.x == 0 && threadIdx.x == 0) offsets[pair_count] = record_count;
}

__global__ void scatter_intersection_indices_by_pair(
	const GpuIntersection *intersections, uint32_t count,
	uint32_t *pair_cursors, uint32_t *grouped_indices)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= count) return;
	const uint32_t output_id = atomicAdd(pair_cursors + intersections[id].pair_id, 1U);
	grouped_indices[output_id] = id;
}

__global__ void build_heavy_intersection_segments(
	const uint32_t *counts, const uint32_t *offsets, uint32_t pair_count,
	uint32_t local_threshold, uint32_t *segment_begin, uint32_t *segment_end,
	uint32_t *segment_count)
{
	const uint32_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(pair_id >= pair_count || counts[pair_id] <= local_threshold) return;
	const uint32_t output_id = atomicAdd(segment_count, 1U);
	segment_begin[output_id] = offsets[pair_id];
	segment_end[output_id] = offsets[pair_id + 1];
}

template<bool SourceOrder>
__global__ void block_sort_intersection_pairs(
	const GpuIntersection *intersections, const uint32_t *input_indices,
	uint32_t *output_indices, const uint32_t *counts, const uint32_t *offsets,
	uint32_t pair_count, uint32_t local_threshold,
	const pair<uint32_t, uint32_t> *pairs, const FarmOffset *farm_offset,
	const RasterInfo *info)
{
	const uint32_t pair_id = blockIdx.x;
	if(pair_id >= pair_count) return;
	const uint32_t count = counts[pair_id];
	if(count == 0 || count > local_threshold || count > INTERSECTION_LOCAL_SORT_CAPACITY) return;
	using BlockSort = cub::BlockRadixSort<uint64_t, INTERSECTION_LOCAL_SORT_THREADS,
		INTERSECTION_LOCAL_SORT_ITEMS, uint32_t>;
	__shared__ typename BlockSort::TempStorage sort_storage;
	__shared__ uint64_t shared_keys[INTERSECTION_LOCAL_SORT_CAPACITY];
	__shared__ uint32_t shared_indices[INTERSECTION_LOCAL_SORT_CAPACITY];
	uint64_t keys[INTERSECTION_LOCAL_SORT_ITEMS];
	uint32_t indices[INTERSECTION_LOCAL_SORT_ITEMS];
	const uint32_t begin = offsets[pair_id];
	#pragma unroll
	for(uint32_t item = 0; item < INTERSECTION_LOCAL_SORT_ITEMS; item++){
		const uint32_t local_id = threadIdx.x * INTERSECTION_LOCAL_SORT_ITEMS + item;
		if(local_id < count){
			indices[item] = input_indices[begin + local_id];
			keys[item] = make_indexed_intersection_key(intersections[indices[item]],
				pairs, farm_offset, info, SourceOrder);
		}else{
			indices[item] = UINT32_MAX;
			keys[item] = UINT64_MAX;
		}
	}
	BlockSort(sort_storage).Sort(keys, indices);
	#pragma unroll
	for(uint32_t item = 0; item < INTERSECTION_LOCAL_SORT_ITEMS; item++){
		const uint32_t local_id = threadIdx.x * INTERSECTION_LOCAL_SORT_ITEMS + item;
		shared_keys[local_id] = keys[item];
		shared_indices[local_id] = indices[item];
	}
	__syncthreads();
	for(uint32_t local_begin = threadIdx.x; local_begin < count;
		local_begin += blockDim.x){
		if(local_begin > 0 && shared_keys[local_begin - 1] == shared_keys[local_begin]) continue;
		uint32_t local_end = local_begin + 1;
		while(local_end < count && shared_keys[local_end] == shared_keys[local_begin]) local_end++;
		for(uint32_t i = local_begin + 1; i < local_end; i++){
			const uint32_t value = shared_indices[i];
			const GpuIntersection current = intersections[value];
			uint32_t j = i;
			while(j > local_begin && intersection_record_less(
				current, intersections[shared_indices[j - 1]], SourceOrder)){
				shared_indices[j] = shared_indices[j - 1];
				j--;
			}
			shared_indices[j] = value;
		}
	}
	__syncthreads();
	for(uint32_t local_id = threadIdx.x; local_id < count; local_id += blockDim.x){
		output_indices[begin + local_id] = shared_indices[local_id];
	}
}

template<bool SourceOrder>
__global__ void order_heavy_intersection_key_runs(
	const uint64_t *keys, uint32_t *indices,
	const uint32_t *segment_begin, const uint32_t *segment_end,
	uint32_t segment_count, const GpuIntersection *intersections)
{
	const uint32_t segment_id = blockIdx.x;
	if(segment_id >= segment_count) return;
	const uint32_t begin = segment_begin[segment_id];
	const uint32_t end = segment_end[segment_id];
	for(uint32_t run_begin = begin + threadIdx.x; run_begin < end;
		run_begin += blockDim.x){
		if(run_begin > begin && keys[run_begin - 1] == keys[run_begin]) continue;
		uint32_t run_end = run_begin + 1;
		while(run_end < end && keys[run_end] == keys[run_begin]) run_end++;
		for(uint32_t i = run_begin + 1; i < run_end; i++){
			const uint32_t value = indices[i];
			const GpuIntersection current = intersections[value];
			uint32_t j = i;
			while(j > run_begin && intersection_record_less(
				current, intersections[indices[j - 1]], SourceOrder)){
				indices[j] = indices[j - 1];
				j--;
			}
			indices[j] = value;
		}
	}
}

size_t query_intersection_scan_workspace(
	uint32_t *counts, uint32_t *offsets, uint32_t pair_count)
{
	void *temporary_storage = nullptr;
	size_t temporary_storage_bytes = 0;
	CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
		temporary_storage, temporary_storage_bytes, counts, offsets, pair_count));
	return temporary_storage_bytes;
}

size_t query_segmented_intersection_sort_workspace(
	IntersectionSortStorage storage, uint32_t count, uint32_t segment_count,
	uint32_t *segment_begin, uint32_t *segment_end)
{
	void *temporary_storage = nullptr;
	size_t temporary_storage_bytes = 0;
	CUDA_SAFE_CALL(cub::DeviceSegmentedRadixSort::SortPairs(
		temporary_storage, temporary_storage_bytes,
		storage.keys_a, storage.keys_b, storage.indices_a, storage.indices_b,
		count, segment_count, segment_begin, segment_end, 0, INTERSECTION_SORT_BITS));
	return temporary_storage_bytes;
}

void segmented_sort_intersections(
	IntersectionSortStorage storage, uint32_t count, uint32_t segment_count,
	uint32_t *segment_begin, uint32_t *segment_end,
	void *temporary_storage, size_t temporary_storage_bytes)
{
	if(count == 0 || segment_count == 0) return;
	CUDA_SAFE_CALL(cub::DeviceSegmentedRadixSort::SortPairs(
		temporary_storage, temporary_storage_bytes,
		storage.keys_a, storage.keys_b, storage.indices_a, storage.indices_b,
		count, segment_count, segment_begin, segment_end, 0, INTERSECTION_SORT_BITS));
}

__device__ __forceinline__ const GpuIntersection &ordered_gpu_intersection(
	const GpuIntersection *intersections, const uint32_t *indices, uint32_t id)
{
	return intersections[indices == nullptr ? id : indices[id]];
}

__global__ void mark_unique_indexed_gpu_intersections(
	const GpuIntersection *intersections, const uint32_t *indices,
	uint32_t count, uint8_t *flags)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= count) return;
	if(id == 0){
		flags[id] = 1;
		return;
	}
	const GpuIntersection current = intersections[indices[id]];
	const GpuIntersection previous = intersections[indices[id - 1]];
	flags[id] = current.pair_id != previous.pair_id ||
		current.edge_source_id != previous.edge_source_id ||
		current.edge_target_id != previous.edge_target_id ||
		fabsf(current.t - previous.t) > INTERSECTION_PARAM_EPS ||
		fabsf(current.u - previous.u) > INTERSECTION_PARAM_EPS;
}

size_t query_indexed_intersection_compact_workspace(
	uint32_t *input, uint8_t *flags, uint32_t *output,
	uint32_t *selected_count, uint32_t count)
{
	void *temporary_storage = nullptr;
	size_t temporary_storage_bytes = 0;
	CUDA_SAFE_CALL(cub::DeviceSelect::Flagged(
		temporary_storage, temporary_storage_bytes,
		input, flags, output, selected_count, count));
	return temporary_storage_bytes;
}

uint32_t compact_unique_indexed_gpu_intersections(
	const GpuIntersection *intersections, uint32_t *input, uint32_t *output,
	uint32_t count, uint8_t *flags, uint32_t *selected_count,
	void *temporary_storage, size_t temporary_storage_bytes)
{
	if(count <= 1){
		if(count == 1 && input != output){
			CUDA_SAFE_CALL(cudaMemcpy(output, input, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
		}
		return count;
	}
	const int grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
	mark_unique_indexed_gpu_intersections<<<grid_size, BLOCK_SIZE>>>(
		intersections, input, count, flags);
	check_execution("mark_unique_indexed_gpu_intersections");
	CUDA_SAFE_CALL(cub::DeviceSelect::Flagged(
		temporary_storage, temporary_storage_bytes,
		input, flags, output, selected_count, count));
	uint32_t host_count = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&host_count, selected_count,
		sizeof(uint32_t), cudaMemcpyDeviceToHost));
	return host_count;
}


__global__ void count_gpu_intersections_per_pair(
	const GpuIntersection *intersections, const uint32_t *indices,
	uint32_t count, uint32_t *counts)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < count){
		atomicAdd(counts + ordered_gpu_intersection(intersections, indices, id).pair_id, 1U);
	}
}

__device__ __forceinline__ Point gpu_interpolate_intersection(
	const Point *vertices, uint32_t edge_id, float param)
{
	const Point a = vertices[edge_id];
	const Point b = vertices[edge_id + 1];
	return Point(fmaf(param, b.x - a.x, a.x), fmaf(param, b.y - a.y, a.y));
}

__device__ __forceinline__ bool gpu_point_on_segment_fp32(
	const Point &point, const Point &a, const Point &b)
{
	const float dx = b.x - a.x;
	const float dy = b.y - a.y;
	const float px = point.x - a.x;
	const float py = point.y - a.y;
	const float length_sq = fmaf(dx, dx, dy * dy);
	if(length_sq == 0.0f){
		return fmaf(px, px, py * py) <= INTERSECTION_PARALLEL_EPS_SQ;
	}
	const float cross = intersection_cross(dx, dy, px, py);
	if(cross * cross > INTERSECTION_PARALLEL_EPS_SQ * length_sq *
		fmaxf(fmaf(px, px, py * py), 1.0f)){
		return false;
	}
	const float dot = fmaf(px, dx, py * dy);
	return dot >= -INTERSECTION_PARAM_EPS && dot <= length_sq + INTERSECTION_PARAM_EPS;
}

__device__ PartitionStatus gpu_point_in_ring_fp32(
	const Point &point, const Point *vertices, uint32_t begin, uint32_t end)
{
	bool inside = false;
	for(uint32_t edge = begin; edge + 1 < end; edge++){
		const Point a = vertices[edge];
		const Point b = vertices[edge + 1];
		if(gpu_point_on_segment_fp32(point, a, b)) return BORDER;
		if((a.y > point.y) != (b.y > point.y)){
			const float intersection_x = fmaf((b.x - a.x) / (b.y - a.y), point.y - a.y, a.x);
			if(intersection_x > point.x) inside = !inside;
		}
	}
	return inside ? IN : OUT;
}

__device__ PartitionStatus classify_shared_gpu_arc(
	const Point &sample, const Point &primary_direction,
	const Point *vertices, uint32_t secondary_begin, uint32_t secondary_end,
	bool is_primary)
{
	for(uint32_t edge = secondary_begin; edge + 1 < secondary_end; edge++){
		const Point a = vertices[edge];
		const Point b = vertices[edge + 1];
		if(!gpu_point_on_segment_fp32(sample, a, b)) continue;
		const float secondary_dx = b.x - a.x;
		const float secondary_dy = b.y - a.y;
		const float cross = intersection_cross(primary_direction.x, primary_direction.y,
			secondary_dx, secondary_dy);
		const float primary_len_sq = fmaf(primary_direction.x, primary_direction.x,
			primary_direction.y * primary_direction.y);
		const float secondary_len_sq = fmaf(secondary_dx, secondary_dx, secondary_dy * secondary_dy);
		if(cross * cross > INTERSECTION_PARALLEL_EPS_SQ * primary_len_sq * secondary_len_sq) continue;
		if(!is_primary) return OUT;
		const float dot = fmaf(primary_direction.x, secondary_dx, primary_direction.y * secondary_dy);
		return dot > 0.0f ? IN : OUT;
	}
	return BORDER;
}

__device__ PartitionStatus classify_gpu_intersection_arc(
	const GpuIntersection &current, const GpuIntersection &next,
	const Point *vertices, uint32_t source_begin, uint32_t source_end,
	uint32_t target_begin, uint32_t target_end, bool is_primary)
{
	const uint32_t overlap_group = gpu_intersection_overlap(current);
	if(overlap_group != 0 && overlap_group == gpu_intersection_overlap(next)){
		if(!is_primary) return OUT;
		const Point source_start = gpu_interpolate_intersection(
			vertices, current.edge_source_id, current.t);
		const Point source_finish = gpu_interpolate_intersection(
			vertices, next.edge_source_id, next.t);
		const Point target_a = vertices[current.edge_target_id];
		const Point target_b = vertices[current.edge_target_id + 1];
		const float dot = fmaf(source_finish.x - source_start.x, target_b.x - target_a.x,
			(source_finish.y - source_start.y) * (target_b.y - target_a.y));
		return dot > 0.0f ? IN : OUT;
	}

	if(overlap_group == 0 &&
	   current.t > INTERSECTION_PARAM_EPS && current.t < 1.0f - INTERSECTION_PARAM_EPS &&
	   current.u > INTERSECTION_PARAM_EPS && current.u < 1.0f - INTERSECTION_PARAM_EPS){
		const Point source_a = vertices[current.edge_source_id];
		const Point source_b = vertices[current.edge_source_id + 1];
		const Point target_a = vertices[current.edge_target_id];
		const Point target_b = vertices[current.edge_target_id + 1];
		const float cross = intersection_cross(target_b.x - target_a.x, target_b.y - target_a.y,
			source_b.x - source_a.x, source_b.y - source_a.y);
		return is_primary ? (cross > 0.0f ? IN : OUT) : (cross < 0.0f ? IN : OUT);
	}

	const uint32_t current_edge = is_primary ? current.edge_source_id : current.edge_target_id;
	const uint32_t next_edge = is_primary ? next.edge_source_id : next.edge_target_id;
	const float current_param = is_primary ? current.t : current.u;
	const float next_param = is_primary ? next.t : next.u;
	const Point current_point = gpu_interpolate_intersection(vertices, current_edge, current_param);
	const Point next_point = gpu_interpolate_intersection(vertices, next_edge, next_param);
	Point sample;
	Point direction;
	if(current_edge == next_edge && current_param < next_param){
		sample = Point((current_point.x + next_point.x) * 0.5f,
			(current_point.y + next_point.y) * 0.5f);
		direction = next_point - current_point;
	}else{
		const Point edge_end = vertices[current_edge + 1];
		sample = Point((current_point.x + edge_end.x) * 0.5f,
			(current_point.y + edge_end.y) * 0.5f);
		direction = edge_end - current_point;
	}

	const uint32_t secondary_begin = is_primary ? target_begin : source_begin;
	const uint32_t secondary_end = is_primary ? target_end : source_end;
	const PartitionStatus status = gpu_point_in_ring_fp32(sample, vertices,
		secondary_begin, secondary_end);
	if(status != BORDER) return status;
	const PartitionStatus shared = classify_shared_gpu_arc(sample, direction, vertices,
		secondary_begin, secondary_end, is_primary);
	return shared == BORDER ? OUT : shared;
}

__device__ __forceinline__ void add_gpu_area_vertex(
	const Point *vertices, uint32_t vertex_id,
	double &area, double &last_x, double &last_y)
{
	const double x = vertices[vertex_id].x;
	const double y = vertices[vertex_id].y;
	area += last_x * y - last_y * x;
	last_x = x;
	last_y = y;
}

__device__ double gpu_arc_double_area(
	const GpuIntersection &current, const GpuIntersection &next,
	const Point *vertices, uint32_t ring_begin, uint32_t ring_end, bool is_primary)
{
	const uint32_t edge1 = is_primary ? current.edge_source_id : current.edge_target_id;
	uint32_t edge2 = is_primary ? next.edge_source_id : next.edge_target_id;
	const float param1 = is_primary ? current.t : current.u;
	float param2 = is_primary ? next.t : next.u;
	const Point point1 = gpu_interpolate_intersection(vertices, edge1, param1);
	const Point point2 = gpu_interpolate_intersection(vertices, edge2, param2);

	if(param2 <= INTERSECTION_PARAM_EPS){
		edge2 = edge2 > ring_begin ? edge2 - 1 : ring_end - 2;
		param2 = 1.0f;
	}

	double area = 0.0;
	double last_x = point1.x;
	double last_y = point1.y;

	if(edge1 < edge2 || (edge1 == edge2 && param1 < param2)){
		for(uint32_t vertex_id = edge1 + 1; vertex_id <= edge2; vertex_id++){
			add_gpu_area_vertex(vertices, vertex_id, area, last_x, last_y);
		}
	}else{
		for(uint32_t vertex_id = edge1 + 1; vertex_id < ring_end - 1; vertex_id++){
			add_gpu_area_vertex(vertices, vertex_id, area, last_x, last_y);
		}
		for(uint32_t vertex_id = ring_begin; vertex_id <= edge2; vertex_id++){
			add_gpu_area_vertex(vertices, vertex_id, area, last_x, last_y);
		}
	}
	area += last_x * point2.y - last_y * point2.x;
	return area;
}

__global__ void classify_and_accumulate_gpu_arcs(
	const GpuIntersection *intersections, const uint32_t *indices, uint32_t count,
	const pair<uint32_t, uint32_t> *pairs, const FarmOffset *farm_offset,
	const RasterInfo *info, const Point *vertices, const uint32_t *counts,
	double *areas, bool is_primary)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= count) return;
	const GpuIntersection current = ordered_gpu_intersection(intersections, indices, id);
	const uint32_t pair_id = current.pair_id;
	const uint32_t pair_count = counts[pair_id];
	const uint32_t next_id = id + 1 >= count ||
		ordered_gpu_intersection(intersections, indices, id + 1).pair_id != pair_id
		? id + 1 - pair_count : id + 1;
	const GpuIntersection next = ordered_gpu_intersection(intersections, indices, next_id);
	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	const uint32_t source_begin = farm_offset[source_idx].vertices_start;
	const uint32_t source_end = farm_offset[source_idx + 1].vertices_start;
	const uint32_t target_begin = farm_offset[target_idx].vertices_start;
	const uint32_t target_end = farm_offset[target_idx + 1].vertices_start;

	if(classify_gpu_intersection_arc(current, next, vertices,
		source_begin, source_end, target_begin, target_end, is_primary) != IN){
		return;
	}
	const uint32_t ring_begin = is_primary ? source_begin : target_begin;
	const uint32_t ring_end = is_primary ? source_end : target_end;
	atomicAdd(areas + pair_id,
		gpu_arc_double_area(current, next, vertices, ring_begin, ring_end, is_primary));
}

__device__ double gpu_positive_double_area(
	const Point *vertices, uint32_t begin, uint32_t end)
{
	double area = 0.0;
	for(uint32_t id = begin; id + 1 < end; id++){
		area += (double)vertices[id].x * vertices[id + 1].y
			- (double)vertices[id].y * vertices[id + 1].x;
	}
	return fabs(area);
}

__device__ bool gpu_mbr_contains(const box &container, const box &subject)
{
	return container.low[0] <= subject.low[0] && container.low[1] <= subject.low[1]
		&& container.high[0] >= subject.high[0] && container.high[1] >= subject.high[1];
}

__global__ void collect_no_crossing_gpu_pairs(
	const pair<uint32_t, uint32_t> *pairs, uint32_t pair_count,
	const RasterInfo *info, const uint32_t *intersection_counts,
	uint32_t *containment_pairs, uint32_t *containment_count)
{
	const uint32_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(pair_id >= pair_count || intersection_counts[pair_id] != 0) return;
	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	if(!gpu_mbr_contains(info[source_idx].mbr, info[target_idx].mbr) &&
	   !gpu_mbr_contains(info[target_idx].mbr, info[source_idx].mbr)) return;
	const uint32_t output_id = atomicAdd(containment_count, 1U);
	containment_pairs[output_id] = pair_id;
}

__global__ void resolve_gpu_containment_pairs(
	const uint32_t *containment_pairs, const uint32_t *containment_count,
	const pair<uint32_t, uint32_t> *pairs, const FarmOffset *farm_offset,
	const RasterInfo *info, const Point *vertices, double *areas)
{
	const uint32_t queue_id = blockIdx.x;
	if(queue_id >= *containment_count) return;
	const uint32_t pair_id = containment_pairs[queue_id];
	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
	const uint32_t source_idx = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
	const uint32_t target_idx = source_idx == pair.first ? pair.second : pair.first;
	const uint32_t source_begin = farm_offset[source_idx].vertices_start;
	const uint32_t source_end = farm_offset[source_idx + 1].vertices_start;
	const uint32_t target_begin = farm_offset[target_idx].vertices_start;
	const uint32_t target_end = farm_offset[target_idx + 1].vertices_start;
	__shared__ uint32_t has_outside;
	__shared__ uint32_t has_inside;
	__shared__ uint32_t resolved;
	__shared__ uint32_t source_contains_target;
	__shared__ uint32_t target_contains_source;
	if(threadIdx.x == 0){
		has_outside = 0;
		has_inside = 0;
		resolved = 0;
		source_contains_target = gpu_mbr_contains(
			info[source_idx].mbr, info[target_idx].mbr);
		target_contains_source = gpu_mbr_contains(
			info[target_idx].mbr, info[source_idx].mbr);
	}
	__syncthreads();

	if(source_contains_target){
		for(uint32_t id = target_begin + threadIdx.x; id + 1 < target_end; id += blockDim.x){
			PartitionStatus status = gpu_point_in_ring_fp32(vertices[id], vertices,
				source_begin, source_end);
			if(status == OUT) atomicExch(&has_outside, 1U);
			if(status == IN) atomicExch(&has_inside, 1U);
			const Point midpoint((vertices[id].x + vertices[id + 1].x) * 0.5f,
				(vertices[id].y + vertices[id + 1].y) * 0.5f);
			status = gpu_point_in_ring_fp32(midpoint, vertices, source_begin, source_end);
			if(status == OUT) atomicExch(&has_outside, 1U);
			if(status == IN) atomicExch(&has_inside, 1U);
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && !has_outside && source_contains_target){
		const double source_area = gpu_positive_double_area(vertices, source_begin, source_end);
		const double target_area = gpu_positive_double_area(vertices, target_begin, target_end);
		if(has_inside || fabs(source_area - target_area) <=
		   1e-6 + 1e-5 * fmax(source_area, target_area)){
			areas[pair_id] = target_area;
			resolved = 1;
		}
	}
	__syncthreads();
	if(resolved) return;

	if(threadIdx.x == 0){
		has_outside = 0;
		has_inside = 0;
	}
	__syncthreads();
	if(target_contains_source){
		for(uint32_t id = source_begin + threadIdx.x; id + 1 < source_end; id += blockDim.x){
			PartitionStatus status = gpu_point_in_ring_fp32(vertices[id], vertices,
				target_begin, target_end);
			if(status == OUT) atomicExch(&has_outside, 1U);
			if(status == IN) atomicExch(&has_inside, 1U);
			const Point midpoint((vertices[id].x + vertices[id + 1].x) * 0.5f,
				(vertices[id].y + vertices[id + 1].y) * 0.5f);
			status = gpu_point_in_ring_fp32(midpoint, vertices, target_begin, target_end);
			if(status == OUT) atomicExch(&has_outside, 1U);
			if(status == IN) atomicExch(&has_inside, 1U);
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && !has_outside && target_contains_source){
		const double source_area = gpu_positive_double_area(vertices, source_begin, source_end);
		const double target_area = gpu_positive_double_area(vertices, target_begin, target_end);
		if(has_inside || fabs(source_area - target_area) <=
		   1e-6 + 1e-5 * fmax(source_area, target_area)){
			areas[pair_id] = source_area;
		}
	}
}

__global__ void finalize_gpu_intersection_areas(double *areas, uint32_t count)
{
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < count) areas[id] = fabs(areas[id]);
}

} // namespace

static void cuda_approximate_intersection(query_context *gctx, size_t batch_size)
{
    if(batch_size == 0) return;

    double *d_areas = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_areas, batch_size * sizeof(double)));

    kernel_approximate_intersection_area<<<batch_size, INTERSECTION_APPROX_THREADS>>>(
        gctx->d_candidate_pairs + gctx->index,
        gctx->d_farm_offset,
        gctx->d_info,
        gctx->d_status,
        batch_size,
	        static_cast<uint8_t>(gctx->bitwidth),
        d_areas);
    check_execution("kernel_approximate_intersection_area");

    CUDA_SAFE_CALL(cudaMemcpy(gctx->areas + gctx->index, d_areas,
        batch_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_areas));
}

void cuda_intersection(query_context *gctx)
{
	const size_t batch_size = gctx->index_end - gctx->index;
	if(gctx->use_approximation){
		cuda_approximate_intersection(gctx, batch_size);
		return;
	}
	if(batch_size == 0) return;

	const int block_size = BLOCK_SIZE;
	const uint32_t pair_count = static_cast<uint32_t>(batch_size);
	auto *batch_pairs = gctx->d_candidate_pairs + gctx->index;
	const uint32_t pixpair_capacity = CUDA_SCRATCH_BUFFER_BYTES / sizeof(PixPair);
	const uint32_t intersection_capacity = CUDA_SCRATCH_BUFFER_BYTES / sizeof(GpuIntersection);

	double *d_areas = nullptr;
	uint32_t *d_intersections_per_pair = nullptr;
	IntersectionDeviceControl *d_control = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_areas, batch_size * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_intersections_per_pair, batch_size * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_control, sizeof(IntersectionDeviceControl)));
	uint32_t *d_stage_count = &d_control->count;
	uint32_t *d_overflow = &d_control->overflow;
	uint32_t *d_overlap_counter = &d_control->auxiliary_count;
	CUDA_SAFE_CALL(cudaMemset(d_areas, 0, batch_size * sizeof(double)));
	CUDA_SAFE_CALL(cudaMemset(d_intersections_per_pair, 0, batch_size * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(d_control, 0, sizeof(IntersectionDeviceControl)));

	int grid_size = (pair_count + block_size - 1) / block_size;
	kernel_filter_intersection<<<grid_size, block_size>>>(
		batch_pairs, gctx->d_farm_offset, gctx->d_info, gctx->d_status,
		pair_count, reinterpret_cast<PixPair *>(gctx->d_BufferInput),
			d_stage_count, static_cast<uint8_t>(gctx->bitwidth),
		pixpair_capacity, d_overflow);
	check_execution("kernel_filter_intersection");

	IntersectionStageResult stage_result{};
	CUDA_SAFE_CALL(cudaMemcpy(&stage_result, d_control,
		sizeof(stage_result), cudaMemcpyDeviceToHost));
	const uint32_t pixpair_count = stage_result.count;
	if(stage_result.overflow || pixpair_count > pixpair_capacity){
		fprintf(stderr, "intersection pixel-pair buffer overflow: %u > %u\n",
			pixpair_count, pixpair_capacity);
		exit(EXIT_FAILURE);
	}

	uint32_t intersection_count = 0;
	if(pixpair_count > 0){
		grid_size = (pixpair_count + block_size - 1) / block_size;
		const uint32_t task_capacity = CUDA_SCRATCH_BUFFER_BYTES / sizeof(Task);
		CUDA_SAFE_CALL(cudaMemset(d_control, 0, sizeof(IntersectionStageResult)));
		kernel_unroll_intersection_tasks<<<grid_size, block_size>>>(
			reinterpret_cast<PixPair *>(gctx->d_BufferInput), pixpair_count,
			batch_pairs, gctx->d_farm_offset, gctx->d_info,
			gctx->d_offset, gctx->d_edge_sequences,
			reinterpret_cast<Task *>(gctx->d_BufferOutput), task_capacity,
			d_stage_count, d_overflow);
		check_execution("kernel_unroll_intersection");

		CUDA_SAFE_CALL(cudaMemcpy(&stage_result, d_control,
			sizeof(stage_result), cudaMemcpyDeviceToHost));
		const uint32_t task_count = stage_result.count;
		if(stage_result.overflow || task_count > task_capacity){
			fprintf(stderr, "intersection task buffer overflow: %u > %u\n",
				task_count, task_capacity);
			exit(EXIT_FAILURE);
		}

		if(task_count > 0){
			CUDA_SAFE_CALL(cudaMemset(d_control, 0, sizeof(IntersectionDeviceControl)));
			grid_size = (task_count + INTERSECTION_WARPS_PER_BLOCK - 1)
				/ INTERSECTION_WARPS_PER_BLOCK;
			kernel_refinement_intersection<<<grid_size, block_size>>>(
				reinterpret_cast<Task *>(gctx->d_BufferOutput), batch_pairs,
				gctx->d_farm_offset, gctx->d_info, gctx->d_vertices,
				task_count, reinterpret_cast<GpuIntersection *>(gctx->d_BufferInput),
				intersection_capacity, d_stage_count,
				d_overlap_counter, d_intersections_per_pair, d_overflow);
			check_execution("kernel_refinement_intersection");
			CUDA_SAFE_CALL(cudaMemcpy(&stage_result, d_control,
				sizeof(stage_result), cudaMemcpyDeviceToHost));
			intersection_count = stage_result.count;
			if(stage_result.overflow || intersection_count > intersection_capacity){
				fprintf(stderr, "intersection record buffer overflow\n");
				exit(EXIT_FAILURE);
			}
		}
	}

	const size_t auxiliary_bytes = std::max(
		static_cast<size_t>(intersection_count) * sizeof(uint8_t),
		static_cast<size_t>(pair_count) * sizeof(uint32_t));
	void *d_auxiliary = nullptr;
	CUDA_SAFE_CALL(cudaMalloc(&d_auxiliary, auxiliary_bytes));
	auto *d_unique_flags = reinterpret_cast<uint8_t *>(d_auxiliary);
	auto *d_containment_pairs = reinterpret_cast<uint32_t *>(d_auxiliary);

	void *d_cub_workspace = nullptr;
	size_t cub_workspace_bytes = 0;
	uint32_t *d_pair_sort_state = nullptr;
	if(intersection_count > 0){
		auto *buffer_a = reinterpret_cast<GpuIntersection *>(gctx->d_BufferInput);
		IntersectionSortStorage sort_storage = make_intersection_sort_storage(
			gctx->d_BufferOutput, intersection_count);
		const uint32_t local_sort_threshold = INTERSECTION_LOCAL_SORT_CAPACITY;
		const size_t pair_state_count = static_cast<size_t>(pair_count) * 4 + 1;
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_pair_sort_state,
			pair_state_count * sizeof(uint32_t)));
		uint32_t *d_pair_offsets = d_pair_sort_state;
		uint32_t *d_pair_cursors = d_pair_offsets + pair_count + 1;
		uint32_t *d_segment_begin = d_pair_cursors + pair_count;
		uint32_t *d_segment_end = d_segment_begin + pair_count;

		const size_t scan_bytes = query_intersection_scan_workspace(
			d_intersections_per_pair, d_pair_offsets, pair_count);
		const size_t segmented_sort_bytes = query_segmented_intersection_sort_workspace(
			sort_storage, intersection_count, pair_count,
			d_segment_begin, d_segment_end);
		const size_t compact_bytes = query_indexed_intersection_compact_workspace(
			sort_storage.indices_b, d_unique_flags, sort_storage.indices_a,
			d_stage_count, intersection_count);
		cub_workspace_bytes = std::max(scan_bytes,
			std::max(segmented_sort_bytes, compact_bytes));
		if(cub_workspace_bytes > 0){
			CUDA_SAFE_CALL(cudaMalloc(&d_cub_workspace, cub_workspace_bytes));
		}

		grid_size = (intersection_count + block_size - 1) / block_size;
		const int pair_grid_size = (pair_count + block_size - 1) / block_size;
		CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
			d_cub_workspace, cub_workspace_bytes,
			d_intersections_per_pair, d_pair_offsets, pair_count));
		finish_intersection_pair_offsets<<<1, 1>>>(
			d_pair_offsets, pair_count, intersection_count);
		check_execution("finish_intersection_pair_offsets");
		CUDA_SAFE_CALL(cudaMemcpy(d_pair_cursors, d_pair_offsets,
			pair_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
		scatter_intersection_indices_by_pair<<<grid_size, block_size>>>(
			buffer_a, intersection_count, d_pair_cursors, sort_storage.indices_a);
		check_execution("scatter_intersection_indices_by_pair");

		build_indexed_intersection_keys<<<grid_size, block_size>>>(
			buffer_a, sort_storage.indices_a, sort_storage.indices_a, sort_storage.keys_a,
			intersection_count, batch_pairs, gctx->d_farm_offset, gctx->d_info,
			true);
		check_execution("build_source_intersection_keys");
		block_sort_intersection_pairs<true><<<pair_count,
			INTERSECTION_LOCAL_SORT_THREADS>>>(
			buffer_a, sort_storage.indices_a, sort_storage.indices_b,
			d_intersections_per_pair, d_pair_offsets, pair_count,
			local_sort_threshold, batch_pairs, gctx->d_farm_offset, gctx->d_info);
		check_execution("block_sort_source_intersection_pairs");
		CUDA_SAFE_CALL(cudaMemset(d_stage_count, 0, sizeof(uint32_t)));
		build_heavy_intersection_segments<<<pair_grid_size, block_size>>>(
			d_intersections_per_pair, d_pair_offsets, pair_count,
			local_sort_threshold, d_segment_begin, d_segment_end,
			d_stage_count);
		check_execution("build_source_heavy_intersection_segments");
		uint32_t heavy_segment_count = 0;
		CUDA_SAFE_CALL(cudaMemcpy(&heavy_segment_count, d_stage_count,
			sizeof(uint32_t), cudaMemcpyDeviceToHost));
		segmented_sort_intersections(sort_storage, intersection_count,
			heavy_segment_count, d_segment_begin, d_segment_end,
			d_cub_workspace, cub_workspace_bytes);
		if(heavy_segment_count > 0){
			order_heavy_intersection_key_runs<true><<<heavy_segment_count,
				block_size>>>(sort_storage.keys_b, sort_storage.indices_b,
				d_segment_begin, d_segment_end, heavy_segment_count, buffer_a);
			check_execution("order_heavy_source_intersection_keys");
		}
		CUDA_SAFE_CALL(cudaMemset(d_stage_count, 0, sizeof(uint32_t)));
		intersection_count = compact_unique_indexed_gpu_intersections(
			buffer_a, sort_storage.indices_b, sort_storage.indices_a,
			intersection_count, d_unique_flags, d_stage_count,
			d_cub_workspace, cub_workspace_bytes);

		if(intersection_count > 0){
			grid_size = (intersection_count + block_size - 1) / block_size;
			CUDA_SAFE_CALL(cudaMemset(d_intersections_per_pair, 0,
				pair_count * sizeof(uint32_t)));
			count_gpu_intersections_per_pair<<<grid_size, block_size>>>(
				buffer_a, sort_storage.indices_a, intersection_count,
				d_intersections_per_pair);
			check_execution("count_gpu_intersections_per_pair");

			CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
				d_cub_workspace, cub_workspace_bytes,
				d_intersections_per_pair, d_pair_offsets, pair_count));
			finish_intersection_pair_offsets<<<1, 1>>>(
				d_pair_offsets, pair_count, intersection_count);
			check_execution("finish_canonical_intersection_pair_offsets");
			classify_and_accumulate_gpu_arcs<<<grid_size, block_size>>>(
				buffer_a, sort_storage.indices_a, intersection_count, batch_pairs,
				gctx->d_farm_offset, gctx->d_info, gctx->d_vertices,
				d_intersections_per_pair, d_areas, true);
			check_execution("classify_source_gpu_arcs");

			build_indexed_intersection_keys<<<grid_size, block_size>>>(
				buffer_a, sort_storage.indices_a, sort_storage.indices_a,
				sort_storage.keys_a, intersection_count, batch_pairs,
				gctx->d_farm_offset, gctx->d_info, false);
			check_execution("build_target_intersection_keys");
			block_sort_intersection_pairs<false><<<pair_count,
				INTERSECTION_LOCAL_SORT_THREADS>>>(
				buffer_a, sort_storage.indices_a, sort_storage.indices_b,
				d_intersections_per_pair, d_pair_offsets, pair_count,
				local_sort_threshold, batch_pairs, gctx->d_farm_offset,
				gctx->d_info);
			check_execution("block_sort_target_intersection_pairs");
			CUDA_SAFE_CALL(cudaMemset(d_stage_count, 0, sizeof(uint32_t)));
			build_heavy_intersection_segments<<<pair_grid_size, block_size>>>(
				d_intersections_per_pair, d_pair_offsets, pair_count,
				local_sort_threshold, d_segment_begin, d_segment_end,
				d_stage_count);
			check_execution("build_target_heavy_intersection_segments");
			heavy_segment_count = 0;
			CUDA_SAFE_CALL(cudaMemcpy(&heavy_segment_count, d_stage_count,
				sizeof(uint32_t), cudaMemcpyDeviceToHost));
			segmented_sort_intersections(sort_storage, intersection_count,
				heavy_segment_count, d_segment_begin, d_segment_end,
				d_cub_workspace, cub_workspace_bytes);
			if(heavy_segment_count > 0){
				order_heavy_intersection_key_runs<false><<<heavy_segment_count,
					block_size>>>(sort_storage.keys_b, sort_storage.indices_b,
					d_segment_begin, d_segment_end, heavy_segment_count, buffer_a);
				check_execution("order_heavy_target_intersection_keys");
			}
			classify_and_accumulate_gpu_arcs<<<grid_size, block_size>>>(
				buffer_a, sort_storage.indices_b, intersection_count, batch_pairs,
				gctx->d_farm_offset, gctx->d_info, gctx->d_vertices,
				d_intersections_per_pair, d_areas, false);
			check_execution("classify_target_gpu_arcs");
		}
	}

	CUDA_SAFE_CALL(cudaMemset(d_stage_count, 0, sizeof(uint32_t)));
	grid_size = (pair_count + block_size - 1) / block_size;
	collect_no_crossing_gpu_pairs<<<grid_size, block_size>>>(
		batch_pairs, pair_count, gctx->d_info, d_intersections_per_pair,
		d_containment_pairs, d_stage_count);
	check_execution("collect_no_crossing_gpu_pairs");
	resolve_gpu_containment_pairs<<<pair_count, block_size>>>(
		d_containment_pairs, d_stage_count, batch_pairs,
		gctx->d_farm_offset, gctx->d_info, gctx->d_vertices, d_areas);
	check_execution("resolve_gpu_containment_pairs");
	finalize_gpu_intersection_areas<<<grid_size, block_size>>>(d_areas, pair_count);
	check_execution("finalize_gpu_intersection_areas");

	CUDA_SAFE_CALL(cudaMemcpy(gctx->areas + gctx->index, d_areas,
		batch_size * sizeof(double), cudaMemcpyDeviceToHost));
	if(d_cub_workspace) CUDA_SAFE_CALL(cudaFree(d_cub_workspace));
	if(d_pair_sort_state) CUDA_SAFE_CALL(cudaFree(d_pair_sort_state));
	CUDA_SAFE_CALL(cudaFree(d_auxiliary));
	CUDA_SAFE_CALL(cudaFree(d_control));
	CUDA_SAFE_CALL(cudaFree(d_intersections_per_pair));
	CUDA_SAFE_CALL(cudaFree(d_areas));
}
