#include "geometry.cuh"
#include "farm.h"
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>
#include <thrust/count.h>

#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>

struct PixelPairWithProb
{
    double probability;
	int source_pixid = 0;
	int target_pixid = 0;
	int pair_id = 0;

    void print(){
        // printf("prob = %lf, pa = %d, pb = %d, pair_id = %d\n", probability, source_pixid, target_pixid, pair_id);
        printf("%lf, %d, %d, %d\n", probability, source_pixid, target_pixid, pair_id);
    }
};
__device__ __forceinline__ bool gpu_closed_ranges_disjoint(
    double low_a, double high_a,
    double low_b, double high_b)
{
    const double eps = 1e-12;
    return high_a < low_b - eps || low_a > high_b + eps;
}

__device__ __forceinline__ int gpu_clamped_int_offset(int offset, int dim)
{
    return min(max(offset, 0), dim);
}

__device__ __forceinline__ int gpu_clamped_floor_offset(
    double origin,
    double value,
    double step,
    int dim)
{
    int offset = (int)floor((value - origin) / step);
    return min(max(offset, 0), dim - 1);
}

__device__ __forceinline__ bool gpu_try_round_int(double value, int* out)
{
    int rounded = __double2int_rn(value);
    double tolerance = 1e-9 * fmax(1.0, fabs(value));
    if (fabs(value - (double)rounded) > tolerance) {
        return false;
    }
    *out = rounded;
    return true;
}

__device__ __forceinline__ bool gpu_box_contains_box(const box& outer, const box& inner)
{
    const double eps = 1e-12;
    return outer.low[0] <= inner.low[0] + eps &&
           outer.low[1] <= inner.low[1] + eps &&
           outer.high[0] >= inner.high[0] - eps &&
           outer.high[1] >= inner.high[1] - eps;
}

__device__ __forceinline__ bool gpu_filter_intersect_status_pair(
    PartitionStatus source_status,
    PartitionStatus target_status,
    bool target_contains_source_pixel,
    int pa,
    int pb,
    int pair_id,
    PixPair* __restrict__ pixpairs,
    uint* pp_size,
    PixPair* __restrict__ mixed_pixpairs,
    uint* mixed_pp_size,
    bool* __restrict__ res)
{
    if (source_status == OUT || target_status == OUT) return false;

    if (source_status == IN && target_status == IN) {
        res[pair_id] = true;
        return true;
    }

    if (source_status == IN) {
        res[pair_id] = true;
        return true;
    }

    if (target_status == IN && target_contains_source_pixel) {
        res[pair_id] = true;
        return true;
    }

    if (source_status == BORDER && target_status == IN) {
        uint idx = atomicAdd(mixed_pp_size, 1U);
        mixed_pixpairs[idx] = {pa, pb, pair_id};
        return false;
    }

    if (source_status == BORDER && target_status == BORDER) {
        uint idx = atomicAdd(pp_size, 1U);
        pixpairs[idx] = {pa, pb, pair_id};
    }

    return false;
}

__global__ void kernel_filter_intersect(
    const pair<uint32_t, uint32_t>* __restrict__ pairs, 
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info, 
    const uint8_t* __restrict__ status, 
    uint size, 
    PixPair* __restrict__ pixpairs, 
    uint* pp_size, 
    uint8_t bitwidth,
    bool* __restrict__ res,
    PixPair* __restrict__ mixed_pixpairs,
    uint* mixed_pp_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;

    const pair<uint32_t, uint32_t> pair = pairs[x];
    const uint32_t idx_a = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
    const uint32_t idx_b = info[pair.first].step_x >= info[pair.second].step_x ? pair.second : pair.first;
    const RasterInfo info_a = info[idx_a];
    const RasterInfo info_b = info[idx_b];

    if (gpu_closed_ranges_disjoint(info_a.mbr.low[0], info_a.mbr.high[0], info_b.mbr.low[0], info_b.mbr.high[0]) ||
        gpu_closed_ranges_disjoint(info_a.mbr.low[1], info_a.mbr.high[1], info_b.mbr.low[1], info_b.mbr.high[1])) {
        return;
    }

    int ratio_x = 0;
    int ratio_y = 0;
    int base_offset_x = 0;
    int base_offset_y = 0;
    bool use_fast_path =
        info_a.step_x >= info_b.step_x &&
        info_a.step_y >= info_b.step_y &&
        gpu_try_round_int(info_a.step_x / info_b.step_x, &ratio_x) &&
        gpu_try_round_int(info_a.step_y / info_b.step_y, &ratio_y) &&
        ratio_x > 0 &&
        ratio_y > 0 &&
        gpu_try_round_int((info_a.mbr.low[0] - info_b.mbr.low[0]) / info_b.step_x, &base_offset_x) &&
        gpu_try_round_int((info_a.mbr.low[1] - info_b.mbr.low[1]) / info_b.step_y, &base_offset_y);

    const uint32_t status_start_a = farm_offset[idx_a].status_start;
    const uint32_t status_start_b = farm_offset[idx_b].status_start;

    if (use_fast_path) {
        for (int i = 0; i < info_a.dimx; i++)
        {
            int t_i_start = base_offset_x + i * ratio_x;
            int t_i_end = t_i_start + ratio_x;

            if (t_i_end < 0 || t_i_start > info_b.dimx) continue;
            int loop_tx_start = gpu_clamped_int_offset(t_i_start, info_b.dimx);
            int loop_tx_end = gpu_clamped_int_offset(t_i_end, info_b.dimx);

            for (int j = 0; j < info_a.dimy; j++)
            {
                int pa = gpu_get_id(i, j, info_a.dimx); 
	                PartitionStatus source_status = gpu_show_status(
						status, status_start_a, pa, bitwidth);
                if (source_status == OUT) continue;

                int t_j_start = base_offset_y + j * ratio_y;
                int t_j_end = t_j_start + ratio_y;

                if (t_j_end < 0 || t_j_start > info_b.dimy) continue;
                int loop_ty_start = gpu_clamped_int_offset(t_j_start, info_b.dimy);
                int loop_ty_end = gpu_clamped_int_offset(t_j_end, info_b.dimy);

                for (int ti = loop_tx_start; ti < loop_tx_end; ti++)
                {
                    for (int tj = loop_ty_start; tj < loop_ty_end; tj++)
                    {
                        int pb = gpu_get_id(ti, tj, info_b.dimx);
	                        PartitionStatus target_status = gpu_show_status(
								status, status_start_b, pb, bitwidth);
                        bool b_contains_a = ratio_x == 1 && ratio_y == 1;
                        if (gpu_filter_intersect_status_pair(source_status, target_status, b_contains_a, pa, pb, x, pixpairs, pp_size, mixed_pixpairs, mixed_pp_size, res)) {
                            return;
                        }
                    }
                }
            }
        }
        return;
    }

    int i_min = gpu_clamped_floor_offset(info_a.mbr.low[0], info_b.mbr.low[0], info_a.step_x, info_a.dimx);
    int i_max = gpu_clamped_floor_offset(info_a.mbr.low[0], info_b.mbr.high[0], info_a.step_x, info_a.dimx);
    int j_min = gpu_clamped_floor_offset(info_a.mbr.low[1], info_b.mbr.low[1], info_a.step_y, info_a.dimy);
    int j_max = gpu_clamped_floor_offset(info_a.mbr.low[1], info_b.mbr.high[1], info_a.step_y, info_a.dimy);

    for (int i = i_min; i <= i_max; i++)
    {
        const double pix_low_x = info_a.mbr.low[0] + i * info_a.step_x;
        const double pix_high_x = pix_low_x + info_a.step_x;
        if (gpu_closed_ranges_disjoint(pix_low_x, pix_high_x, info_b.mbr.low[0], info_b.mbr.high[0])) continue;

        int loop_tx_start = gpu_clamped_floor_offset(info_b.mbr.low[0], pix_low_x, info_b.step_x, info_b.dimx);
        int loop_tx_end = gpu_clamped_floor_offset(info_b.mbr.low[0], pix_high_x, info_b.step_x, info_b.dimx);

        for (int j = j_min; j <= j_max; j++)
        {
            int pa = gpu_get_id(i, j, info_a.dimx);
	            PartitionStatus source_status = gpu_show_status(
					status, status_start_a, pa, bitwidth);

            if (source_status == OUT) continue;

            const double pix_low_y = info_a.mbr.low[1] + j * info_a.step_y;
            const double pix_high_y = pix_low_y + info_a.step_y;
            if (gpu_closed_ranges_disjoint(pix_low_y, pix_high_y, info_b.mbr.low[1], info_b.mbr.high[1])) continue;
            const box box_a = gpu_get_pixel_box(i, j, info_a.mbr.low[0], info_a.mbr.low[1], info_a.step_x, info_a.step_y);

            int loop_ty_start = gpu_clamped_floor_offset(info_b.mbr.low[1], pix_low_y, info_b.step_y, info_b.dimy);
            int loop_ty_end = gpu_clamped_floor_offset(info_b.mbr.low[1], pix_high_y, info_b.step_y, info_b.dimy);

            for (int ti = loop_tx_start; ti <= loop_tx_end; ti++)
            {
                for (int tj = loop_ty_start; tj <= loop_ty_end; tj++)
                {
                    int pb = gpu_get_id(ti, tj, info_b.dimx);
	                    PartitionStatus target_status = gpu_show_status(
							status, status_start_b, pb, bitwidth);
                    const box box_b = gpu_get_pixel_box(ti, tj, info_b.mbr.low[0], info_b.mbr.low[1], info_b.step_x, info_b.step_y);
                    const bool b_contains_a = gpu_box_contains_box(box_b, box_a);

                    if (gpu_filter_intersect_status_pair(source_status, target_status, b_contains_a, pa, pb, x, pixpairs, pp_size, mixed_pixpairs, mixed_pp_size, res)) {
                        return;
                    }
                }
            }
        }
    }
}

__global__ void kernel_calculate_probability(PixPair *pixpairs,
	pair<uint32_t, uint32_t> *pairs, FarmOffset *farm_offset,
	RasterInfo *info, uint8_t *status, uint *size,
	PixelPairWithProb *p_pixpairs, uint *p_pixpairs_size,
	uint8_t bitwidth, bool *res){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < *size){
        int pa = pixpairs[x].pixid_a;
        int pb = pixpairs[x].pixid_b;
        int pair_id = pixpairs[x].pair_id;

        if(res[pair_id]) return;

        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        const uint32_t idx_a = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
        const uint32_t idx_b = info[pair.first].step_x >= info[pair.second].step_x ? pair.second : pair.first;
        const FarmOffset offset_a = farm_offset[idx_a];
        const FarmOffset offset_b = farm_offset[idx_b];

	        uint8_t pa_fullness = gpu_get_fullness(
				status, offset_a.status_start, pa, bitwidth);
			uint8_t pb_fullness = gpu_get_fullness(
				status, offset_b.status_start, pb, bitwidth);
        double pa_pixelArea = info[idx_a].step_x * info[idx_a].step_y;
        double pb_pixelArea = info[idx_b].step_x * info[idx_b].step_y;
	        double pa_low = gpu_decode_fullness(pa_fullness, pa_pixelArea, bitwidth, true);
	        double pa_high = gpu_decode_fullness(pa_fullness, pa_pixelArea, bitwidth, false);
        double pa_apx = (pa_low + pa_high) / 2;
	        double pb_low = gpu_decode_fullness(pb_fullness, pb_pixelArea, bitwidth, true);
	        double pb_high = gpu_decode_fullness(pb_fullness, pb_pixelArea, bitwidth, false);
        double pb_apx = (pb_low + pb_high) / 2;
        double probability = (pa_apx + pb_apx) / max(pa_pixelArea, pb_pixelArea);
        probability = min(1.0, max(0.0, probability));

        if(pa_low + pb_low >= max(pa_pixelArea, pb_pixelArea)){
            res[pair_id] = true;
            return;
        }

        int idx = atomicAdd(p_pixpairs_size, 1U);
        p_pixpairs[idx] = {probability, pa, pb, pair_id};
    }
}

__global__ void kernel_accumulate_approximate_intersect(
    PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs,
    FarmOffset *farm_offset, RasterInfo *info, uint8_t *status,
	    uint *size, uint8_t bitwidth, bool *res,
    double *scores, uint *candidate_counts)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= *size) return;

    const int pa = pixpairs[x].pixid_a;
    const int pb = pixpairs[x].pixid_b;
    const int pair_id = pixpairs[x].pair_id;
    if(res[pair_id]) return;

    const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t idx_a = info[pair.first].step_x >= info[pair.second].step_x
        ? pair.first : pair.second;
    const uint32_t idx_b = idx_a == pair.first ? pair.second : pair.first;
    const FarmOffset offset_a = farm_offset[idx_a];
    const FarmOffset offset_b = farm_offset[idx_b];
	    const uint8_t pa_fullness = gpu_get_fullness(
			status, offset_a.status_start, pa, bitwidth);
	    const uint8_t pb_fullness = gpu_get_fullness(
			status, offset_b.status_start, pb, bitwidth);
    const double pa_pixel_area = info[idx_a].step_x * info[idx_a].step_y;
    const double pb_pixel_area = info[idx_b].step_x * info[idx_b].step_y;
    const double pixel_area = max(pa_pixel_area, pb_pixel_area);
	    const double pa_low = gpu_decode_fullness(pa_fullness, pa_pixel_area, bitwidth, true);
	    const double pa_high = gpu_decode_fullness(pa_fullness, pa_pixel_area, bitwidth, false);
	    const double pb_low = gpu_decode_fullness(pb_fullness, pb_pixel_area, bitwidth, true);
	    const double pb_high = gpu_decode_fullness(pb_fullness, pb_pixel_area, bitwidth, false);

    if(pa_low + pb_low >= pixel_area){
        res[pair_id] = true;
        return;
    }

    const double pa_approx = (pa_low + pa_high) * 0.5;
    const double pb_approx = (pb_low + pb_high) * 0.5;
    const double probability = min(1.0, max(0.0,
        (pa_approx + pb_approx) / pixel_area));
    if(probability >= 1.0){
        res[pair_id] = true;
        return;
    }

    atomicAdd(candidate_counts + pair_id, 1U);
    if(probability > 0.0){
        atomicAdd(scores + pair_id, -log1p(-probability));
    }
}

__global__ void kernel_finalize_approximate_intersect(
    const double *scores, const uint *candidate_counts, uint size,
    bool *res, double required_score)
{
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= size || res[pair_id] || candidate_counts[pair_id] == 0) return;

    if(scores[pair_id] >= required_score){
        res[pair_id] = true;
    }
}

struct PairComparator {
    __host__ __device__
    bool operator()(const PixelPairWithProb &a, const PixelPairWithProb &b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id;
        }
        return a.probability > b.probability;
    }
};

void OrganizePixelPairs(
    PixelPairWithProb* d_input_buffer,
    int num_pairs,
    PixelPairWithProb** d_out_sorted,
    int** d_out_offsets,
    int** d_cur_ptr,
    int* out_num_groups) 
{
    if (num_pairs == 0) return;

    thrust::device_ptr<PixelPairWithProb> dev_ptr(d_input_buffer);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + num_pairs, PairComparator());

    cudaMalloc((void**)d_out_sorted, num_pairs * sizeof(PixelPairWithProb));
    cudaMemcpy(*d_out_sorted, d_input_buffer, num_pairs * sizeof(PixelPairWithProb), cudaMemcpyDeviceToDevice);

    thrust::device_vector<int> temp_keys(num_pairs);
    thrust::transform(thrust::device, 
                      dev_ptr, 
                      dev_ptr + num_pairs, 
                      temp_keys.begin(), 
                      [] __device__ (const PixelPairWithProb& p) { return p.pair_id; });

    thrust::device_vector<int> temp_counts(num_pairs);
    
    auto new_end = thrust::reduce_by_key(
        thrust::device,
        temp_keys.begin(), temp_keys.end(),
        thrust::constant_iterator<int>(1),
        thrust::make_discard_iterator(),
        temp_counts.begin()
    );

    int num_groups = new_end.second - temp_counts.begin();
    *out_num_groups = num_groups;

    cudaMalloc((void**)d_out_offsets, (num_groups + 1) * sizeof(int));
    
    thrust::device_ptr<int> dev_offsets(*d_out_offsets);
    thrust::exclusive_scan(
        thrust::device,
        temp_counts.begin(),
        temp_counts.begin() + num_groups,
        dev_offsets
    );

    cudaMemcpy(*d_out_offsets + num_groups, &num_pairs, sizeof(int), cudaMemcpyHostToDevice);
    
    if (d_cur_ptr != nullptr) {
        cudaMalloc((void **)d_cur_ptr, (num_groups + 1) * sizeof(int));
        cudaMemcpy(*d_cur_ptr, *d_out_offsets, (num_groups + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
}

__global__ void kernel_merge_intersect(PixelPairWithProb *pixpairs, int *cur, int *offset, int pairsize, PixPair *buffer, uint *buffer_size, bool *res, double threshold){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairsize && cur[tid] < offset[tid + 1])
    {
        int start = cur[tid];
        int end = offset[tid + 1];
        int pairId = pixpairs[start].pair_id;

        if(res[pairId]){
            cur[tid] = end;
            return;
        }
        
        double obj_prob = 1;
        for(int i = start; i < end; i ++){
            double pix_prob = pixpairs[i].probability;
            assert(pix_prob >= 0.0);
            obj_prob = obj_prob * (1 - pix_prob);
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = {pixpairs[i].source_pixid, pixpairs[i].target_pixid, pairId};
            if(1 - obj_prob >= threshold) { 
                cur[tid] = i + 1;
                return;
            } 
        }
        cur[tid] = end;
    }
}

__device__ __forceinline__ bool gpu_segment_sequences_intersect(
    const Point* __restrict__ p1,
    const Point* __restrict__ p2,
    uint s1,
    uint s2)
{
    if (s1 <= MAX_SIZE && s2 <= MAX_SIZE) {
        Point local_p1[MAX_SIZE + 1], local_p2[MAX_SIZE + 1];

        #pragma unroll 4
        for (uint i = 0; i <= s1; i++) {
            local_p1[i] = p1[i];
        }

        #pragma unroll 4
        for (uint i = 0; i <= s2; i++) {
            local_p2[i] = p2[i];
        }

        for (uint i = 0; i < s1; i++) {
            Point a = local_p1[i];
            Point b = local_p1[i + 1];
            for (uint j = 0; j < s2; j++) {
                if (gpu_segment_intersect(a, b, local_p2[j], local_p2[j + 1])) {
                    return true;
                }
            }
        }
        return false;
    }

    for (uint i = 0; i < s1; i++) {
        for (uint j = 0; j < s2; j++) {
            if (gpu_segment_intersect(p1[i], p1[i + 1], p2[j], p2[j + 1])) {
                return true;
            }
        }
    }
    return false;
}

__device__ bool gpu_contain_point(
    uint32_t poly_idx,
    Point p,
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info,
    const uint8_t* __restrict__ status,
    const uint32_t* __restrict__ es_offset,
    const EdgeSeq* __restrict__ edge_sequences,
    const Point* __restrict__ vertices,
    const uint32_t* __restrict__ gridline_offset,
    const double* __restrict__ gridline_nodes,
	    uint8_t bitwidth);

__global__ void kernel_mixed_border_in_intersect(
    const PixPair* __restrict__ pixpairs,
    const pair<uint32_t, uint32_t>* __restrict__ pairs,
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info,
    const uint8_t* __restrict__ status,
    const uint32_t* __restrict__ es_offset,
    const EdgeSeq* __restrict__ edge_sequences,
    const Point* __restrict__ vertices,
    const uint32_t* __restrict__ gridline_offset,
    const double* __restrict__ gridline_nodes,
    const uint* __restrict__ size,
	    uint8_t bitwidth,
    bool* __restrict__ res)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= *size) return;

    const int pair_id = pixpairs[x].pair_id;
    if (res[pair_id]) return;

    const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t idx_a = info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
    const uint32_t idx_b = info[pair.first].step_x >= info[pair.second].step_x ? pair.second : pair.first;
    const FarmOffset offset_a = farm_offset[idx_a];
    const RasterInfo info_b = info[idx_b];
    const int pa = pixpairs[x].pixid_a;
    const int pb = pixpairs[x].pixid_b;

    const int bx = gpu_get_x(pb, info_b.dimx);
    const int by = gpu_get_y(pb, info_b.dimx, info_b.dimy);
    const box target_box = gpu_get_pixel_box(bx, by, info_b.mbr.low[0], info_b.mbr.low[1], info_b.step_x, info_b.step_y);

    Point target_border[5];
    target_border[0] = Point(target_box.low[0], target_box.low[1]);
    target_border[1] = Point(target_box.low[0], target_box.high[1]);
    target_border[2] = Point(target_box.high[0], target_box.high[1]);
    target_border[3] = Point(target_box.high[0], target_box.low[1]);
    target_border[4] = Point(target_box.low[0], target_box.low[1]);

    const uint32_t sequence_start = (es_offset + offset_a.offset_start)[pa];
    const uint32_t sequence_end = (es_offset + offset_a.offset_start)[pa + 1];
    for (uint32_t i = sequence_start; i < sequence_end; i++) {
        const EdgeSeq edges = (edge_sequences + offset_a.edge_sequences_start)[i];
        if (gpu_segment_sequences_intersect(
                vertices + offset_a.vertices_start + edges.start,
                target_border,
                edges.length,
                4)) {
            res[pair_id] = true;
            return;
        }
    }

    const Point center((target_box.low[0] + target_box.high[0]) * 0.5,
                       (target_box.low[1] + target_box.high[1]) * 0.5);
    if (gpu_contain_point(idx_a, center, farm_offset, info, status, es_offset,
                          edge_sequences, vertices, gridline_offset, gridline_nodes,
	                          bitwidth)) {
        res[pair_id] = true;
    }
}

__global__ void kernel_unroll_intersect(
    PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs,
    FarmOffset *farm_offset, RasterInfo *info,
    uint32_t *es_offset, EdgeSeq *edge_sequences, uint *size,
    Task *tasks, uint *task_size, uint unroll_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *size) return;

    int pa = pixpairs[idx].pixid_a;
    int pb = pixpairs[idx].pixid_b;
    int pair_id = pixpairs[idx].pair_id;

    const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t idx_a =
        info[pair.first].step_x >= info[pair.second].step_x ? pair.first : pair.second;
    const uint32_t idx_b =
        info[pair.first].step_x >= info[pair.second].step_x ? pair.second : pair.first;
    const FarmOffset offset_a = farm_offset[idx_a];
    const FarmOffset offset_b = farm_offset[idx_b];

    uint offset_start_a = offset_a.offset_start;
    uint offset_start_b = offset_b.offset_start;
    uint edge_sequences_start_a = offset_a.edge_sequences_start;
    uint edge_sequences_start_b = offset_b.edge_sequences_start;
    uint s_vertices_start = offset_a.vertices_start;
    uint t_vertices_start = offset_b.vertices_start;

    int s_num_sequence =
        (es_offset + offset_start_a)[pa + 1] - (es_offset + offset_start_a)[pa];
    int t_num_sequence =
        (es_offset + offset_start_b)[pb + 1] - (es_offset + offset_start_b)[pb];

    for (uint i = 0; i < s_num_sequence; ++i)
    {
        EdgeSeq r =
            (edge_sequences + edge_sequences_start_a)[(es_offset + offset_start_a)[pa] + i];
        if (r.length == 0) continue;
        uint source_chunks = (r.length - 1) / unroll_size + 1;
        for (uint j = 0; j < t_num_sequence; ++j)
        {
            EdgeSeq r2 =
                (edge_sequences + edge_sequences_start_b)[(es_offset + offset_start_b)[pb] + j];
            if (r2.length == 0) continue;
            uint target_chunks = (r2.length - 1) / unroll_size + 1;
            uint task_count = source_chunks * target_chunks;
            uint task_base = atomicAdd(task_size, task_count);

            uint source_chunk = 0;
            for (uint s = 0; s < r.length; s += unroll_size, source_chunk++)
            {
                uint source_length = min(unroll_size, r.length - s);
                uint target_chunk = 0;
                for (uint t = 0; t < r2.length; t += unroll_size, target_chunk++)
                {
                    uint target_length = min(unroll_size, r2.length - t);
                    uint task_id =
                        task_base + source_chunk * target_chunks + target_chunk;
                    tasks[task_id].s_start = s_vertices_start + r.start + s;
                    tasks[task_id].t_start = t_vertices_start + r2.start + t;
                    tasks[task_id].s_length = source_length;
                    tasks[task_id].t_length = target_length;
                    tasks[task_id].pair_id = pair_id;
                }
            }
        }
    }
}

__global__ void kernel_refinement_intersect(Task *tasks, Point *d_vertices, uint *size, bool *res)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;
	
	uint s1 = tasks[x].s_start;
	uint s2 = tasks[x].t_start;
	uint len1 = tasks[x].s_length;
	uint len2 = tasks[x].t_length;
	int pair_id = tasks[x].pair_id;

    if(gpu_segment_sequences_intersect((d_vertices + s1), (d_vertices + s2), len1, len2)){
	    res[pair_id] = true;
    }

    return;
}

__device__ __forceinline__ bool gpu_box_contains_point(const box& bx, const Point& p)
{
    return p.x >= bx.low[0] && p.x <= bx.high[0] &&
           p.y >= bx.low[1] && p.y <= bx.high[1];
}

__device__ bool gpu_contain_point(
    uint32_t poly_idx,
    Point p,
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info,
    const uint8_t* __restrict__ status,
    const uint32_t* __restrict__ es_offset,
    const EdgeSeq* __restrict__ edge_sequences,
    const Point* __restrict__ vertices,
    const uint32_t* __restrict__ gridline_offset,
    const double* __restrict__ gridline_nodes,
	    uint8_t bitwidth)
{
    const RasterInfo raster = info[poly_idx];
    if (!gpu_box_contains_point(raster.mbr, p)) {
        return false;
    }

    const FarmOffset offset = farm_offset[poly_idx];
    const int xoff = gpu_get_offset_x(raster.mbr.low[0], p.x, raster.step_x, raster.dimx);
    const int yoff = gpu_get_offset_y(raster.mbr.low[1], p.y, raster.step_y, raster.dimy);
    const int pix_id = gpu_get_id(xoff, yoff, raster.dimx);

	    const PartitionStatus st = gpu_show_status(
			status, offset.status_start, pix_id, bitwidth);
    if (st == IN) {
        return true;
    }
    if (st == OUT) {
        return false;
    }

    bool ret = false;
    const box pix_box = gpu_get_pixel_box(
        xoff, yoff,
        raster.mbr.low[0], raster.mbr.low[1],
        raster.step_x, raster.step_y
    );

    const uint32_t sequence_start = (es_offset + offset.offset_start)[pix_id];
    const uint32_t sequence_end = (es_offset + offset.offset_start)[pix_id + 1];
    for (uint32_t i = sequence_start; i < sequence_end; i++) {
        const EdgeSeq edges = (edge_sequences + offset.edge_sequences_start)[i];
        for (uint32_t k = 0; k < edges.length; k++) {
            const Point v1 = (vertices + offset.vertices_start)[edges.start + k];
            const Point v2 = (vertices + offset.vertices_start)[edges.start + k + 1];

            if ((v1.y >= p.y) != (v2.y >= p.y)) {
                const double int_x = (v2.x - v1.x) * (p.y - v1.y) / (v2.y - v1.y) + v1.x;
                if (p.x <= int_x && int_x <= pix_box.high[0]) {
                    ret = !ret;
                }
            }
        }
    }

    const uint32_t gridline_start = offset.gridline_offset_start;
    const uint32_t node_start = (gridline_offset + gridline_start)[xoff + 1];
    const uint32_t node_end = (gridline_offset + gridline_start)[xoff + 2];
    const int nc = binary_search_count((gridline_nodes + offset.gridline_nodes_start), node_start, node_end, p.y);

    return ret ^ (nc & 1);
}

__global__ void kernel_containment_intersect(
    const pair<uint32_t, uint32_t>* __restrict__ pairs,
    const FarmOffset* __restrict__ farm_offset,
    const RasterInfo* __restrict__ info,
    const uint8_t* __restrict__ status,
    const uint32_t* __restrict__ es_offset,
    const EdgeSeq* __restrict__ edge_sequences,
    const Point* __restrict__ vertices,
    const uint32_t* __restrict__ gridline_offset,
    const double* __restrict__ gridline_nodes,
    uint size,
	    uint8_t bitwidth,
    bool* __restrict__ res)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size || res[x]) return;

    const pair<uint32_t, uint32_t> pair = pairs[x];
    const uint32_t source_idx = pair.first;
    const uint32_t target_idx = pair.second;

    const Point target_point = (vertices + farm_offset[target_idx].vertices_start)[0];
    if (gpu_contain_point(source_idx, target_point, farm_offset, info, status, es_offset,
                          edge_sequences, vertices, gridline_offset, gridline_nodes,
	                          bitwidth)) {
        res[x] = true;
        return;
    }

    const Point source_point = (vertices + farm_offset[source_idx].vertices_start)[0];
    if (gpu_contain_point(target_idx, source_point, farm_offset, info, status, es_offset,
                          edge_sequences, vertices, gridline_offset, gridline_nodes,
	                          bitwidth)) {
        res[x] = true;
    }
}


__global__ void statistic_result(bool *res, uint size, uint *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size && res[x] == true) atomicAdd(result, 1);
}

void cuda_intersect(query_context *gctx)
{
	size_t batch_size = gctx->index_end - gctx->index;
	uint h_bufferinput_size, h_bufferoutput_size;
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_result, 0, sizeof(uint)));

    bool *d_res = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_res, batch_size * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemset(d_res, 0, batch_size * sizeof(bool)));

	/*1. Raster Model Filtering*/
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_filter_intersect<<<grid_size, block_size>>>(
        gctx->d_candidate_pairs + gctx->index,
        gctx->d_farm_offset,
        gctx->d_info,
        gctx->d_status,
        batch_size,
        (PixPair *)gctx->d_BufferInput,
        gctx->d_bufferinput_size,
	        static_cast<uint8_t>(gctx->bitwidth),
        d_res,
        (PixPair *)gctx->d_BufferOutput,
        gctx->d_bufferoutput_size);
    check_execution("kernel_filter_intersect");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

	if (h_bufferoutput_size > 0) {
        grid_size = (h_bufferoutput_size + block_size - 1) / block_size;
        kernel_mixed_border_in_intersect<<<grid_size, block_size>>>(
            (PixPair *)gctx->d_BufferOutput,
            gctx->d_candidate_pairs + gctx->index,
            gctx->d_farm_offset,
            gctx->d_info,
            gctx->d_status,
            gctx->d_offset,
            gctx->d_edge_sequences,
            gctx->d_vertices,
            gctx->d_gridline_offset,
            gctx->d_gridline_nodes,
            gctx->d_bufferoutput_size,
	            static_cast<uint8_t>(gctx->bitwidth),
            d_res);
        check_execution("kernel_mixed_border_in_intersect");
    }
    CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

    const bool need_refine = !gctx->use_approximation;
    if(gctx->use_approximation){
        if(h_bufferinput_size > 0){
            double *d_scores = nullptr;
            uint *d_candidate_counts = nullptr;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, batch_size * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_candidate_counts, batch_size * sizeof(uint)));
            CUDA_SAFE_CALL(cudaMemset(d_scores, 0, batch_size * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemset(d_candidate_counts, 0, batch_size * sizeof(uint)));

            grid_size = (h_bufferinput_size + block_size - 1) / block_size;
            kernel_accumulate_approximate_intersect<<<grid_size, block_size>>>(
                (PixPair *)gctx->d_BufferInput,
                gctx->d_candidate_pairs + gctx->index,
                gctx->d_farm_offset,
                gctx->d_info,
                gctx->d_status,
                gctx->d_bufferinput_size,
	                static_cast<uint8_t>(gctx->bitwidth),
                d_res,
                d_scores,
                d_candidate_counts);
            check_execution("kernel_accumulate_approximate_intersect");

            const double confidence = static_cast<double>(gctx->approx_confidence);
            const double required_score = confidence >= 1.0
                ? HUGE_VAL : -std::log1p(-confidence);
            grid_size = (batch_size + block_size - 1) / block_size;
            kernel_finalize_approximate_intersect<<<grid_size, block_size>>>(
                d_scores, d_candidate_counts, batch_size, d_res, required_score);
            check_execution("kernel_finalize_approximate_intersect");

            CUDA_SAFE_CALL(cudaFree(d_scores));
            CUDA_SAFE_CALL(cudaFree(d_candidate_counts));
        }
    }else{
        if(h_bufferinput_size > 0){
            grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_calculate_probability<<<grid_size, block_size>>>(
                (PixPair *)gctx->d_BufferInput,
                gctx->d_candidate_pairs + gctx->index,
                gctx->d_farm_offset,
                gctx->d_info,
                gctx->d_status,
                gctx->d_bufferinput_size,
                (PixelPairWithProb*)gctx->d_BufferOutput,
                gctx->d_bufferoutput_size,
	                static_cast<uint8_t>(gctx->bitwidth),
                d_res);
            check_execution("kernel_calculate_probability");
        }

        CUDA_SWAP_BUFFER();
		if(h_bufferinput_size > 0){
            PixelPairWithProb* d_sorted_data = nullptr;
            int* d_offsets = nullptr;
            int* d_cur_ptr = nullptr;
            int num_groups = 0;

            OrganizePixelPairs(
                (PixelPairWithProb*)gctx->d_BufferInput,
                h_bufferinput_size,
                &d_sorted_data,
                &d_offsets,
                &d_cur_ptr,
                &num_groups
            );

			while(true){
				grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernel_merge_intersect<<<grid_size, block_size>>>(d_sorted_data, d_cur_ptr, d_offsets, num_groups, (PixPair*)gctx->d_BufferOutput, gctx->d_bufferoutput_size, d_res, gctx->merge_threshold);
                check_execution("kernel_merge_intersect");

                CUDA_SWAP_BUFFER();
                if(h_bufferinput_size == 0) break;

				// 2. Unroll Refinement

                grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernel_unroll_intersect<<<grid_size, block_size>>>(
                    (PixPair *)gctx->d_BufferInput,
                    gctx->d_candidate_pairs + gctx->index,
                    gctx->d_farm_offset,
                    gctx->d_info,
                    gctx->d_offset,
                    gctx->d_edge_sequences,
                    gctx->d_bufferinput_size,
                    (Task *)gctx->d_BufferOutput,
                    gctx->d_bufferoutput_size,
                    static_cast<uint>(gctx->unroll_size));
                check_execution("kernel_unroll_intersect");

                CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
                /*3. Refinement step*/

                if(h_bufferoutput_size > 0){
                    grid_size = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    kernel_refinement_intersect<<<grid_size, block_size>>>((Task *)gctx->d_BufferOutput, gctx->d_vertices, gctx->d_bufferoutput_size, d_res);
                    check_execution("kernel_refinement_intersect");
                }

                CUDA_SWAP_BUFFER();
            }

            CUDA_SAFE_CALL(cudaFree(d_sorted_data));
            CUDA_SAFE_CALL(cudaFree(d_offsets));
            CUDA_SAFE_CALL(cudaFree(d_cur_ptr));
        }
    }

    if(need_refine){
        grid_size = (batch_size + block_size - 1) / block_size;
        kernel_containment_intersect<<<grid_size, block_size>>>(
            gctx->d_candidate_pairs + gctx->index,
            gctx->d_farm_offset,
            gctx->d_info,
            gctx->d_status,
            gctx->d_offset,
            gctx->d_edge_sequences,
            gctx->d_vertices,
            gctx->d_gridline_offset,
            gctx->d_gridline_nodes,
            batch_size,
	            static_cast<uint8_t>(gctx->bitwidth),
            d_res
        );
        check_execution("kernel_containment_intersect");
    }

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    statistic_result<<<grid_size, block_size>>>(d_res, batch_size, gctx->d_result);
    check_execution("statistic_result"); 

    uint h_result;
	CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
	gctx->found += h_result;
    CUDA_SAFE_CALL(cudaFree(d_res));
    return;
}
