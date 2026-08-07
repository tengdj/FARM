#include "geometry.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <cmath>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

struct BoxDistRange
{
    int sourcePixelId;
    int targetPixelId;
    int pairId;
    float minDist;
    float maxDist;
};

static_assert(sizeof(BoxDistRange) == 20, "final raster candidate must stay compact");

enum class ExpansionMode : uint8_t
{
    Fixed = 0,
    SourceOnly = 1,
    TargetOnly = 2,
    Both = 3
};

struct ActiveRasterCandidate
{
    int sourcePixelId;
    int targetPixelId;
    int pairId;
    uint8_t s_cur_level;
    uint8_t t_cur_level;
    ExpansionMode mode;
    uint8_t reserved;
};

static_assert(sizeof(ActiveRasterCandidate) == 16, "active raster candidate must stay compact");

__device__ __forceinline__ ExpansionMode get_expansion_mode(
    const RasterInfo *__restrict__ layer_info,
    const FarmOffset &source,
    const FarmOffset &target,
    uint8_t s_cur_level,
    uint8_t t_cur_level,
    uint s_level,
    uint t_level)
{
    const float s_step = static_cast<float>(
        (layer_info + source.layer_start)[s_cur_level].step_x);
    const float t_step = static_cast<float>(
        (layer_info + target.layer_start)[t_cur_level].step_x);
    const bool expand_source = s_cur_level < s_level &&
        (s_step >= t_step || t_cur_level >= t_level);
    const uint8_t next_source_level = s_cur_level + static_cast<uint8_t>(expand_source);
    const bool expand_target = t_cur_level < t_level &&
        (s_step <= t_step || next_source_level >= s_level);

    return static_cast<ExpansionMode>(
        static_cast<uint8_t>(expand_source) |
        (static_cast<uint8_t>(expand_target) << 1));
}

__device__ __forceinline__ BoxDistRange *get_final_candidate_slot(
    char *__restrict__ final_buffer, uint final_index)
{
    return reinterpret_cast<BoxDistRange *>(final_buffer + CUDA_SCRATCH_BUFFER_BYTES -
        (static_cast<size_t>(final_index) + 1) * sizeof(BoxDistRange));
}

__device__ __forceinline__ void append_active_candidate(
    ActiveRasterCandidate *__restrict__ active_buffer,
    uint *__restrict__ active_size,
    uint *__restrict__ final_size,
    bool active_shares_final,
    uint *__restrict__ overflow_flag,
    const ActiveRasterCandidate &value)
{
    const uint index = atomicAdd(active_size, 1U);
    const size_t active_bytes = (static_cast<size_t>(index) + 1) * sizeof(ActiveRasterCandidate);
    const size_t final_bytes = active_shares_final
        ? static_cast<size_t>(*final_size) * sizeof(BoxDistRange)
        : 0;
    if(active_bytes > CUDA_SCRATCH_BUFFER_BYTES ||
       active_bytes + final_bytes > CUDA_SCRATCH_BUFFER_BYTES){
        atomicExch(overflow_flag, 1U);
        return;
    }
    active_buffer[index] = value;
}

__device__ __forceinline__ void append_final_candidate(
    char *__restrict__ final_buffer,
    uint *__restrict__ final_size,
    uint *__restrict__ active_size,
    bool active_output_shares_final,
    uint input_shared_active_size,
    uint *__restrict__ overflow_flag,
    const BoxDistRange &value)
{
    const uint index = atomicAdd(final_size, 1U);
    const size_t final_bytes = (static_cast<size_t>(index) + 1) * sizeof(BoxDistRange);
    const size_t active_bytes = active_output_shares_final
        ? static_cast<size_t>(*active_size) * sizeof(ActiveRasterCandidate)
        : static_cast<size_t>(input_shared_active_size) * sizeof(ActiveRasterCandidate);
    if(final_bytes > CUDA_SCRATCH_BUFFER_BYTES ||
       active_bytes + final_bytes > CUDA_SCRATCH_BUFFER_BYTES){
        atomicExch(overflow_flag, 1U);
        return;
    }
    *get_final_candidate_slot(final_buffer, index) = value;
}

struct PixelDist
{
    int sourcePixelId;
    int targetPixelId;
    uint8_t pf;
    int pairId;
    float apxDist;
    float minDist;
    float maxDist;
};

struct WithinApproximation
{
    float normalized_mean;
    float approximate_distance;
};

__device__ __forceinline__ WithinApproximation calculate_within_approximation(
    uint8_t pf, uint8_t bitwidth, float min_distance, float max_distance)
{
    const float category_count = static_cast<float>(status_category_count(bitwidth));
    const float normalized_mean =
        (1.0f - pf / category_count) * 0.55f;
    const float approximate_distance = min_distance
        + normalized_mean * (max_distance - min_distance);
    return {normalized_mean, approximate_distance};
}

__device__ __forceinline__ uint8_t calculate_combined_fullness(
    uint8_t source_fullness, double source_pixel_area,
    uint8_t target_fullness, double target_pixel_area,
    uint8_t bitwidth)
{
    const double source_low = gpu_decode_fullness(
        source_fullness, source_pixel_area, bitwidth, true);
    const double source_high = gpu_decode_fullness(
        source_fullness, source_pixel_area, bitwidth, false);
    const double target_low = gpu_decode_fullness(
        target_fullness, target_pixel_area, bitwidth, true);
    const double target_high = gpu_decode_fullness(
        target_fullness, target_pixel_area, bitwidth, false);
    const double source_approx = (source_low + source_high) * 0.5;
    const double target_approx = (target_low + target_high) * 0.5;
    return gpu_encode_fullness(source_approx, source_pixel_area,
        target_approx, target_pixel_area, bitwidth);
}

struct PixelDistKeyDecomposer
{
    __host__ __device__
    auto operator()(PixelDist &value) const
    {
        return cuda::std::tie(value.pairId, value.apxDist, value.minDist);
    }
};

static PixelDist *radix_sort_pixel_dist(PixelDist *input, uint size, uint pair_count,
    int begin_bit = 0)
{
    PixelDist *sorted = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&sorted, size * sizeof(PixelDist)));

    uint max_pair_id = pair_count > 0 ? pair_count - 1 : 0;
    int pair_id_bits = 0;
    while(max_pair_id > 0){
        pair_id_bits ++;
        max_pair_id >>= 1;
    }
    const int end_bit = 64 + pair_id_bits;

    if(begin_bit >= end_bit){
        CUDA_SAFE_CALL(cudaMemcpy(sorted, input, size * sizeof(PixelDist),
            cudaMemcpyDeviceToDevice));
        return sorted;
    }

    cub::DoubleBuffer<PixelDist> keys(input, sorted);
    size_t temp_storage_bytes = 0;
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
        keys, size, PixelDistKeyDecomposer(), begin_bit, end_bit));

    void *temp_storage = nullptr;
    if(temp_storage_bytes > 0){
        CUDA_SAFE_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
    }
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes,
        keys, size, PixelDistKeyDecomposer(), begin_bit, end_bit));

    if(keys.Current() != sorted){
        CUDA_SAFE_CALL(cudaMemcpy(sorted, keys.Current(), size * sizeof(PixelDist),
            cudaMemcpyDeviceToDevice));
    }
    if(temp_storage){
        CUDA_SAFE_CALL(cudaFree(temp_storage));
    }
    return sorted;
}

__global__ void kernel_init_distance(
    const pair<uint32_t, uint32_t> *__restrict__ pairs,
    const FarmOffset *__restrict__ farm_offset,
    const RasterInfo *__restrict__ layer_info,
    const uint32_t *__restrict__ layer_offset,
    const uint8_t *__restrict__ status,
    uint size,
    float *__restrict__ max_box_dist,
    ActiveRasterCandidate *__restrict__ buffer,
    uint *__restrict__ buffer_size,
    uint *__restrict__ overflow_flag,
    uint8_t bitwidth)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        const FarmOffset source = farm_offset[pair.first];
        const FarmOffset target = farm_offset[pair.second];
        const uint s_level = farm_offset[pair.first + 1].layer_start - source.layer_start - 1;
        const uint t_level = farm_offset[pair.second + 1].layer_start - target.layer_start - 1;
        const ExpansionMode mode = get_expansion_mode(
            layer_info, source, target, 0, 0, s_level, t_level);

        int s_dimx = (layer_info + source.layer_start)[0].dimx, s_dimy = (layer_info + source.layer_start)[0].dimy;
        int t_dimx = (layer_info + target.layer_start)[0].dimx, t_dimy = (layer_info + target.layer_start)[0].dimy;
        uint32_t source_offset = (layer_offset + source.layer_start)[0];
        uint32_t target_offset = (layer_offset + target.layer_start)[0];

        for(int i = 0; i < s_dimx * s_dimy; i ++){
            for(int j = 0; j < t_dimx * t_dimy; j ++){
                if(gpu_show_status(status, source.status_start, i, bitwidth, source_offset) == BORDER &&
                    gpu_show_status(status, target.status_start, j, bitwidth, target_offset) == BORDER){
                    uint idx = atomicAdd(buffer_size, 1);
                    if((static_cast<size_t>(idx) + 1) * sizeof(ActiveRasterCandidate) >
                       CUDA_SCRATCH_BUFFER_BYTES){
                        atomicExch(overflow_flag, 1U);
                        continue;
                    }
                    buffer[idx] = {i, j, pair_id, 0, 0, mode, 0};
                }
            }
        }
       
        // buffer[pair_id] = {0, 0, pair_id, 0.0, FLT_MAX, 0, 0};
        max_box_dist[pair_id] = FLT_MAX;
    }
}

// calculate lower bound and upper bound between box from (top down)
__global__ void iterative_filtering_step(const ActiveRasterCandidate *__restrict__ candidate,
    const pair<uint32_t, uint32_t> *__restrict__ pairs,
    const FarmOffset *__restrict__ farm_offset,
    const RasterInfo *__restrict__ layer_info,
    const uint32_t *__restrict__ layer_offset,
    const uint8_t *__restrict__ status,
    float *__restrict__ max_box_dist,
    uint size,
    ActiveRasterCandidate *__restrict__ active_buffer,
    uint *__restrict__ active_size,
    char *__restrict__ final_buffer,
    uint *__restrict__ final_size,
    bool active_shares_final,
    uint input_shared_active_size,
    uint *__restrict__ overflow_flag,
    const float *__restrict__ degree_per_kilometer_latitude,
    const float *__restrict__ degree_per_kilometer_longitude_arr,
    uint8_t bitwidth,
    float within_distance)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (candidate_id < size)
    {
        const ActiveRasterCandidate current = candidate[candidate_id];
        int source_pixel_id = current.sourcePixelId;
        int target_pixel_id = current.targetPixelId;
        int pair_id = current.pairId;
        if(max_box_dist[pair_id] < 0.0f) return;

        uint8_t s_cur_level = candidate[candidate_id].s_cur_level;
        uint8_t t_cur_level = candidate[candidate_id].t_cur_level;

        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        const FarmOffset source = farm_offset[pair.first];
        const FarmOffset target = farm_offset[pair.second];
        uint s_level = farm_offset[pair.first + 1].layer_start - source.layer_start - 1;
        uint t_level = farm_offset[pair.second + 1].layer_start - target.layer_start - 1;

        int source_start_x, source_start_y, source_end_x, source_end_y, target_start_x, target_start_y, target_end_x, target_end_y;
        uint32_t source_offset, target_offset;
        float s_origin_x, s_origin_y, t_origin_x, t_origin_y;
        float s_step_x, s_step_y, t_step_x, t_step_y;
        int s_dimx, s_dimy, t_dimx, t_dimy;

        const bool expand_source = current.mode == ExpansionMode::SourceOnly ||
            current.mode == ExpansionMode::Both;
        const bool expand_target = current.mode == ExpansionMode::TargetOnly ||
            current.mode == ExpansionMode::Both;

        if(expand_source){
            int source_parent_dimx = (layer_info + source.layer_start)[s_cur_level].dimx;
            int source_parent_y = source_pixel_id / source_parent_dimx;
            int source_parent_x = source_pixel_id - source_parent_y * source_parent_dimx;
            s_cur_level ++;

            source_offset = (layer_offset + source.layer_start)[s_cur_level];
            const RasterInfo &source_info = (layer_info + source.layer_start)[s_cur_level];
            s_origin_x = static_cast<float>(source_info.mbr.low[0]);
            s_origin_y = static_cast<float>(source_info.mbr.low[1]);
            s_step_x = static_cast<float>(source_info.step_x);
            s_step_y = static_cast<float>(source_info.step_y);
            s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

            source_start_x = source_parent_x * 2;
            source_start_y = source_parent_y * 2;
            source_end_x = min(source_start_x + 1, s_dimx - 1);
            source_end_y = min(source_start_y + 1, s_dimy - 1);
        }else{
            source_offset = (layer_offset + source.layer_start)[s_cur_level];
            const RasterInfo &source_info = (layer_info + source.layer_start)[s_cur_level];
            s_origin_x = static_cast<float>(source_info.mbr.low[0]);
            s_origin_y = static_cast<float>(source_info.mbr.low[1]);
            s_step_x = static_cast<float>(source_info.step_x);
            s_step_y = static_cast<float>(source_info.step_y);
            s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

            source_start_y = source_pixel_id / s_dimx;
            source_start_x = source_pixel_id - source_start_y * s_dimx;
            source_end_x = source_start_x;
            source_end_y = source_start_y;
        }

        if(expand_target){
            int target_parent_dimx = (layer_info + target.layer_start)[t_cur_level].dimx;
            int target_parent_y = target_pixel_id / target_parent_dimx;
            int target_parent_x = target_pixel_id - target_parent_y * target_parent_dimx;
            t_cur_level ++;

            target_offset = (layer_offset + target.layer_start)[t_cur_level];
            const RasterInfo &target_info = (layer_info + target.layer_start)[t_cur_level];
            t_origin_x = static_cast<float>(target_info.mbr.low[0]);
            t_origin_y = static_cast<float>(target_info.mbr.low[1]);
            t_step_x = static_cast<float>(target_info.step_x);
            t_step_y = static_cast<float>(target_info.step_y);
            t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

            target_start_x = target_parent_x * 2;
            target_start_y = target_parent_y * 2;
            target_end_x = min(target_start_x + 1, t_dimx - 1);
            target_end_y = min(target_start_y + 1, t_dimy - 1);
        }else{
            target_offset = (layer_offset + target.layer_start)[t_cur_level];
            const RasterInfo &target_info = (layer_info + target.layer_start)[t_cur_level];
            t_origin_x = static_cast<float>(target_info.mbr.low[0]);
            t_origin_y = static_cast<float>(target_info.mbr.low[1]);
            t_step_x = static_cast<float>(target_info.step_x);
            t_step_y = static_cast<float>(target_info.step_y);
            t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

            target_start_y = target_pixel_id / t_dimx;
            target_start_x = target_pixel_id - target_start_y * t_dimx;
            target_end_x = target_start_x;
            target_end_y = target_start_y;
        }

        const float inv_latitude = 1.0f / *degree_per_kilometer_latitude;
        const float within_distance_sq = within_distance * within_distance;
        const ExpansionMode next_mode = get_expansion_mode(
            layer_info, source, target, s_cur_level, t_cur_level, s_level, t_level);
        const bool final_level = next_mode == ExpansionMode::Fixed;
        float min_surviving_max_distance = FLT_MAX;
        for (int x1 = source_start_x; x1 <= source_end_x; x1++)
        {
            for (int y1 = source_start_y; y1 <= source_end_y; y1++)
            {
                int id1 = gpu_get_id(x1, y1, s_dimx);
                if(gpu_show_status(status, source.status_start, id1, bitwidth, source_offset) != BORDER) continue;

                const FloatBox box1 = make_float_pixel_box(x1, y1, s_origin_x, s_origin_y, s_step_x, s_step_y);
                const float longitude_factor = gpu_degree_per_kilometer_longitude(
                    box1.low_y, degree_per_kilometer_longitude_arr);
                const float inv_longitude = 1.0f / longitude_factor;
                for (int x2 = target_start_x; x2 <= target_end_x; x2++)
                {
                    for (int y2 = target_start_y; y2 <= target_end_y; y2++)
                    {
                        int id2 = gpu_get_id(x2, y2, t_dimx);
                        if (gpu_show_status(status, target.status_start, id2, bitwidth, target_offset) == BORDER)
                        {  
                            const FloatBox box2 = make_float_pixel_box(x2, y2,
                                t_origin_x, t_origin_y, t_step_x, t_step_y);
                            const float min_distance_sq = float_box_min_distance_sq(
                                box1, box2, inv_latitude, inv_longitude);
                            if(min_distance_sq > within_distance_sq) continue;

                            const float max_distance_sq = float_box_max_distance_sq(
                                box1, box2, inv_latitude, inv_longitude);
                            if(max_distance_sq <= within_distance_sq){
                                atomicMinFloat(max_box_dist + pair_id, -1.0f);
                                return;
                            }

                            const float max_distance = sqrtf(max_distance_sq);
                            if(final_level){
                                append_final_candidate(final_buffer, final_size, active_size,
                                    active_shares_final, input_shared_active_size, overflow_flag,
                                    {id1, id2, pair_id, sqrtf(min_distance_sq), max_distance});
                            }else{
                                append_active_candidate(active_buffer, active_size, final_size,
                                    active_shares_final, overflow_flag,
                                    {id1, id2, pair_id, s_cur_level, t_cur_level, next_mode, 0});
                            }
                            min_surviving_max_distance = fminf(min_surviving_max_distance, max_distance);
                        }
                    }
                }
            }
        }
        if(min_surviving_max_distance < FLT_MAX){
            atomicMinFloat(max_box_dist + pair_id, min_surviving_max_distance);
        }
    }
}

__global__ void calculate_apxDist(const BoxDistRange *__restrict__ bufferinput,
    const pair<uint32_t, uint32_t> *__restrict__ pairs,
    const FarmOffset *__restrict__ farm_offset,
    const RasterInfo *__restrict__ info,
    const uint8_t *__restrict__ status,
    uint size,
    PixelDist *__restrict__ bufferoutput,
    uint8_t bitwidth){
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < size)
    {
        int pa = bufferinput[bufferId].sourcePixelId;
        int pb = bufferinput[bufferId].targetPixelId;
        int pair_id = bufferinput[bufferId].pairId;

        pair<uint32_t, uint32_t> pair = pairs[pair_id];
        uint32_t src_idx = pair.first;
        uint32_t tar_idx = pair.second;
        FarmOffset source = farm_offset[src_idx];
        FarmOffset target = farm_offset[tar_idx];

        uint8_t pa_fullness = gpu_get_fullness(
            status, source.status_start, pa, bitwidth);
        uint8_t pb_fullness = gpu_get_fullness(
            status, target.status_start, pb, bitwidth);
        double pa_pixelArea = info[src_idx].step_x * info[src_idx].step_y;
        double pb_pixelArea = info[tar_idx].step_x * info[tar_idx].step_y;
        uint8_t pf = calculate_combined_fullness(
            pa_fullness, pa_pixelArea, pb_fullness, pb_pixelArea, bitwidth);
        const WithinApproximation approximation = calculate_within_approximation(
            pf, bitwidth, bufferinput[bufferId].minDist, bufferinput[bufferId].maxDist);

        bufferoutput[bufferId] = {pa, pb, pf, pair_id, approximation.approximate_distance,
            bufferinput[bufferId].minDist, bufferinput[bufferId].maxDist};
    }
}

__device__ __forceinline__ float calculate_within_probability(
    float apx_distance, float min_distance, float max_distance,
    float distance_threshold)
{
    if(distance_threshold < min_distance) return 0.0f;
    if(distance_threshold >= max_distance) return 1.0f;

    float span = max_distance - min_distance;
    if(span <= 0.0f) return distance_threshold >= min_distance ? 1.0f : 0.0f;

    float ratio = (distance_threshold - min_distance) / span;
    float mean = (apx_distance - min_distance) / span;
    ratio = fminf(1.0f, fmaxf(0.0f, ratio));
    mean = fminf(1.0f, fmaxf(0.0f, mean));

    float stddev = 0.3f * mean;
    if(stddev <= 0.0f) return ratio >= mean ? 1.0f : 0.0f;

    float probability = 0.5f * (1.0f + erff((ratio - mean) / (stddev * sqrtf(2.0f))));
    return fminf(1.0f, fmaxf(0.0f, probability));
}

__global__ void kernel_accumulate_approximate_within(
    const BoxDistRange *__restrict__ candidates,
    const pair<uint32_t, uint32_t> *__restrict__ pairs,
    const FarmOffset *__restrict__ farm_offset,
    const RasterInfo *__restrict__ info,
    const uint8_t *__restrict__ status,
    uint size,
    uint8_t bitwidth, float within_distance, bool *res,
    float *scores, uint *candidate_counts)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size) return;

    const BoxDistRange candidate = candidates[x];
    const int pair_id = candidate.pairId;
    if(res[pair_id]) return;

    const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const FarmOffset source = farm_offset[pair.first];
    const FarmOffset target = farm_offset[pair.second];
    const uint8_t source_fullness = gpu_get_fullness(
        status, source.status_start, candidate.sourcePixelId, bitwidth);
    const uint8_t target_fullness = gpu_get_fullness(
        status, target.status_start, candidate.targetPixelId, bitwidth);
    const double source_pixel_area = info[pair.first].step_x * info[pair.first].step_y;
    const double target_pixel_area = info[pair.second].step_x * info[pair.second].step_y;
    const uint8_t pf = calculate_combined_fullness(
        source_fullness, source_pixel_area,
        target_fullness, target_pixel_area, bitwidth);
    const WithinApproximation approximation = calculate_within_approximation(
        pf, bitwidth, candidate.minDist, candidate.maxDist);
    const float probability = calculate_within_probability(
        approximation.approximate_distance, candidate.minDist, candidate.maxDist, within_distance);

    if(probability >= 1.0f){
        res[pair_id] = true;
        return;
    }

    atomicAdd(candidate_counts + pair_id, 1U);
    if(probability > 0.0f){
        atomicAdd(scores + pair_id, -log1pf(-probability));
    }
}

__global__ void kernel_finalize_approximate_within(
    const float *scores, const uint *candidate_counts, uint size,
    bool *res, float required_score)
{
    const uint pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(pair_id >= size || res[pair_id] || candidate_counts[pair_id] == 0) return;

    if(scores[pair_id] >= required_score){
        res[pair_id] = true;
    }
}

__global__ void statistic_result_a(bool *res, float *max_box_dist, uint size, uint *result, float within_distance){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size && (res[x] || max_box_dist[x] <= within_distance))
    {
        atomicAdd(result, 1);
    }
}

__global__ void preprocess_suffixmin(PixelDist *pixpairs, int *pixelpairidx, uint pairsize, float *suffix_min)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairsize && pixelpairidx[tid + 1] - pixelpairidx[tid] > 0)
    {
        int start = pixelpairidx[tid];
        int end = pixelpairidx[tid + 1];
        for(int i = end - 1; i >= start; i --){
            if(i == end - 1){
                suffix_min[i] = pixpairs[i].minDist;
            }else{
                suffix_min[i] = fminf(pixpairs[i].minDist, suffix_min[i + 1]);
            }
        }
    }
}

__global__ void kernel_merge(PixelDist *pixpairs, int *pixelpairidx,
    int *pixelpairsize, float *suffix_min, uint pairsize, PixPair* buffer,
    uint *buffer_size, float *max_box_dist, float threshold)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairsize && pixelpairsize[tid + 1] - pixelpairidx[tid] > 0)
    {
        int start = pixelpairidx[tid];
        int end = pixelpairsize[tid + 1];
        int pairId = pixpairs[start].pairId;

        int num_pix_pairs = pixelpairsize[tid + 1] - pixelpairsize[tid];

        if(max_box_dist[pairId] < 0 || max_box_dist[pairId] <= suffix_min[start]){
            pixelpairidx[tid] = end;
            return;
        }

        int i = start;
        float delta = 0.01;
        uint step = max(1, static_cast<int>(delta * num_pix_pairs));

        while(i < end){
            int end_idx = min(i + step, end);
            for(int j = i; j < end_idx; j ++){
                int idx = atomicAdd(buffer_size, 1);
                buffer[idx] = {pixpairs[j].sourcePixelId, pixpairs[j].targetPixelId, pixpairs[j].pairId};
            }

            if(end_idx == end){
                pixelpairidx[tid] = end;
                return;
            }

            float prob = 1.0f;
            float d_r = (end_idx < end) ? suffix_min[end_idx] : numeric_limits<float>::infinity();

            for(int j = i; j < end_idx; j ++){
                const float probability = calculate_within_probability(
                    pixpairs[j].apxDist, pixpairs[j].minDist, pixpairs[j].maxDist, d_r);
                if(probability >= 1.0f){
                    pixelpairidx[tid] = end_idx;
                    return;
                }
                prob *= 1.0f - probability;
            }

            if(1 - prob >= threshold) {
                pixelpairidx[tid] = end_idx;
                return;
            }

            i = end_idx;
            step = min(step, end - i);
        }
    }
}

__global__ void kernel_unroll_within_polygon(PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs, FarmOffset *farm_offset, uint32_t *es_offset, EdgeSeq *edge_sequences, uint* size, Task *tasks, uint *task_size, uint unroll_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int p = pixpairs[bufferId].pixid_a;
        int p2 = pixpairs[bufferId].pixid_b;
        int pairId = pixpairs[bufferId].pair_id;

        pair<uint32_t, uint32_t> &pair = pairs[pairId];
        FarmOffset &source = farm_offset[pair.first];
        FarmOffset &target = farm_offset[pair.second];

        int s_num_sequence = (es_offset + source.offset_start)[p + 1] - (es_offset + source.offset_start)[p];
        int t_num_sequence = (es_offset + target.offset_start)[p2 + 1] - (es_offset + target.offset_start)[p2];

        for (int i = 0; i < s_num_sequence; ++ i)
        {
            EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[p] + i];
            if(r.length == 0) continue;
            uint source_segments = r.length;
            uint source_chunks = (source_segments + unroll_size - 1) / unroll_size;
            for (int j = 0; j < t_num_sequence; ++j)
            {
                EdgeSeq r2 = (edge_sequences + target.edge_sequences_start)[(es_offset + target.offset_start)[p2] + j];
                if(r2.length == 0) continue;
                uint target_segments = r2.length;
                uint target_chunks = (target_segments + unroll_size - 1) / unroll_size;
                uint task_base = atomicAdd(task_size, source_chunks * target_chunks);
                uint source_chunk = 0;
                for (uint s = 0; s < source_segments; s += unroll_size, source_chunk ++)
                {
                    uint source_segment_count = min(unroll_size, source_segments - s);
                    uint target_chunk = 0;
                    for (uint t = 0; t < target_segments; t += unroll_size, target_chunk ++)
                    {
                        uint target_segment_count = min(unroll_size, target_segments - t);
                        uint task_id = task_base + source_chunk * target_chunks + target_chunk;
                        tasks[task_id] = {source.vertices_start + r.start + s,
                            target.vertices_start + r2.start + t,
                            source_segment_count, target_segment_count, pairId};
                    }
                }
           }
        }
    }
}


__global__ void kernel_refine_within_polygon(Task *tasks, Point *vertices, uint *size, float *max_box_dist, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr, float within_distance)
{
    const int taskId = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskId < *size)
    {
        uint s1 = tasks[taskId].s_start;
        uint s2 = tasks[taskId].t_start;
        uint len1 = tasks[taskId].s_length;
        uint len2 = tasks[taskId].t_length;
        int pair_id = tasks[taskId].pair_id;
        if(max_box_dist[pair_id] < 0.0f){
            return;
        }

        float dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2,
            len1 + 1, len2 + 1, degree_per_kilometer_latitude,
            degree_per_kilometer_longitude_arr, within_distance);
        if(dist <= within_distance){
            atomicMinFloat(max_box_dist + pair_id, -1.0f); 
            return;
        }

        atomicMinFloat(max_box_dist + pair_id, dist);
    }
}

__global__ void statistic_result_polygon(float *max_box_dist, uint size, uint *result, float within_distance){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size && max_box_dist[x] <= within_distance)
    {
        atomicAdd(result, 1);
    }
}

void cuda_within_polygon(query_context *gctx)
{
    size_t batch_size = gctx->index_end - gctx->index;
	uint h_bufferinput_size = 0, h_bufferoutput_size = 0;
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_result, 0, sizeof(uint)));

    float *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, batch_size * sizeof(float)));
    uint *d_final_size = nullptr;
    uint *d_overflow_flag = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_final_size, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_overflow_flag, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_final_size, 0, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_overflow_flag, 0, sizeof(uint)));

    const int block_size = BLOCK_SIZE;
    const float within_distance = static_cast<float>(gctx->within_distance);
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_init_distance<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index,
        gctx->d_farm_offset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status,
        batch_size, d_max_box_dist, (ActiveRasterCandidate *)gctx->d_BufferInput,
        gctx->d_bufferinput_size, d_overflow_flag,
        static_cast<uint8_t>(gctx->bitwidth));
    check_execution("kernel_init");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    uint h_overflow_flag = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_overflow_flag, d_overflow_flag, sizeof(uint), cudaMemcpyDeviceToHost));
    if(h_overflow_flag){
        fprintf(stderr, "GPU within raster active queue exceeded scratch buffer capacity\n");
        exit(EXIT_FAILURE);
    }

    ActiveRasterCandidate *active_input =
        reinterpret_cast<ActiveRasterCandidate *>(gctx->d_BufferInput);
    ActiveRasterCandidate *active_output =
        reinterpret_cast<ActiveRasterCandidate *>(gctx->d_BufferOutput);
    uint *active_input_size = gctx->d_bufferinput_size;
    uint *active_output_size = gctx->d_bufferoutput_size;
    char *final_buffer = reinterpret_cast<char *>(gctx->d_BufferOutput);
    uint h_final_size = 0;

    while(h_bufferinput_size > 0){
        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const bool active_output_shares_final =
            reinterpret_cast<char *>(active_output) == final_buffer;
        const bool active_input_shares_final =
            reinterpret_cast<char *>(active_input) == final_buffer;

        iterative_filtering_step<<<grid_size, block_size>>>(active_input,
            gctx->d_candidate_pairs + gctx->index, gctx->d_farm_offset, gctx->d_layer_info,
            gctx->d_layer_offset, gctx->d_status, d_max_box_dist, h_bufferinput_size,
            active_output, active_output_size, final_buffer, d_final_size,
            active_output_shares_final,
            active_input_shares_final ? h_bufferinput_size : 0,
            d_overflow_flag,
            gctx->d_degree_degree_per_kilometer_latitude,
            gctx->d_degree_per_kilometer_longitude_arr,
            static_cast<uint8_t>(gctx->bitwidth),
            within_distance);
        check_execution("iterative_filtering_step");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, active_output_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_final_size, d_final_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_overflow_flag, d_overflow_flag, sizeof(uint), cudaMemcpyDeviceToHost));
        if(h_overflow_flag){
            fprintf(stderr, "GPU within raster active/final queues exceeded scratch buffer capacity\n");
            exit(EXIT_FAILURE);
        }

        std::swap(active_input, active_output);
        std::swap(active_input_size, active_output_size);
        h_bufferinput_size = h_bufferoutput_size;
        CUDA_SAFE_CALL(cudaMemset(active_output_size, 0, sizeof(uint)));
    }

    BoxDistRange *final_candidates = reinterpret_cast<BoxDistRange *>(
        final_buffer + CUDA_SCRATCH_BUFFER_BYTES -
        static_cast<size_t>(h_final_size) * sizeof(BoxDistRange));
    h_bufferinput_size = h_final_size;
    CUDA_SAFE_CALL(cudaFree(d_final_size));
    CUDA_SAFE_CALL(cudaFree(d_overflow_flag));

    if(gctx->use_approximation){
        bool *d_res = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_res, batch_size * sizeof(bool)));
        CUDA_SAFE_CALL(cudaMemset(d_res, 0, batch_size * sizeof(bool)));

        if(h_bufferinput_size > 0){
            float *d_scores = nullptr;
            uint *d_candidate_counts = nullptr;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, batch_size * sizeof(float)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_candidate_counts, batch_size * sizeof(uint)));
            CUDA_SAFE_CALL(cudaMemset(d_scores, 0, batch_size * sizeof(float)));
            CUDA_SAFE_CALL(cudaMemset(d_candidate_counts, 0, batch_size * sizeof(uint)));

            grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_accumulate_approximate_within<<<grid_size, block_size>>>(
                final_candidates,
                gctx->d_candidate_pairs + gctx->index,
                gctx->d_farm_offset,
                gctx->d_info,
                gctx->d_status,
                h_bufferinput_size,
                static_cast<uint8_t>(gctx->bitwidth),
                within_distance,
                d_res,
                d_scores,
                d_candidate_counts);
            check_execution("kernel_accumulate_approximate_within");

            const float confidence = gctx->approx_confidence;
            const float required_score = confidence >= 1.0f
                ? std::numeric_limits<float>::infinity()
                : -static_cast<float>(std::log1p(-static_cast<double>(confidence)));
            grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_finalize_approximate_within<<<grid_size, block_size>>>(
                d_scores, d_candidate_counts, batch_size, d_res, required_score);
            check_execution("kernel_finalize_approximate_within");

            CUDA_SAFE_CALL(cudaFree(d_scores));
            CUDA_SAFE_CALL(cudaFree(d_candidate_counts));
        }

        grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        statistic_result_a<<<grid_size, block_size>>>(d_res, d_max_box_dist, batch_size, gctx->d_result, within_distance);
        check_execution("statistic_result_a");

        uint h_result;
        CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
        gctx->found += h_result;
        
        CUDA_SAFE_CALL(cudaFree(d_res));
   
    }else{
      if(h_bufferinput_size > 0){
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        calculate_apxDist<<<grid_size, block_size>>>(
            final_candidates,
            gctx->d_candidate_pairs + gctx->index,
            gctx->d_farm_offset,
            gctx->d_info,
            gctx->d_status,
            h_bufferinput_size,
            (PixelDist *)gctx->d_BufferInput,
            static_cast<uint8_t>(gctx->bitwidth));
        check_execution("calculate_apxDist");

        PixelDist* d_pixpairs = radix_sort_pixel_dist(
            (PixelDist*)gctx->d_BufferInput, h_bufferinput_size,
            static_cast<uint>(batch_size));
        thrust::device_ptr<PixelDist> begin = thrust::device_pointer_cast(d_pixpairs);
        thrust::device_ptr<PixelDist> end = begin + h_bufferinput_size;

        thrust::device_vector<int> d_indices(h_bufferinput_size);
        thrust::sequence(d_indices.begin(), d_indices.end());

        thrust::device_vector<int> pair_ids(h_bufferinput_size);
        thrust::transform(begin, end, pair_ids.begin(), 
            [] __device__(const PixelDist &r){
                return r.pairId;});

        thrust::device_vector<int> d_flags(h_bufferinput_size);
        thrust::adjacent_difference(thrust::device, pair_ids.begin(), pair_ids.end(), d_flags.begin());

        thrust::transform(d_flags.begin(), d_flags.end(), d_flags.begin(),
            [] __device__(int x){ return x != 0 ? 1 : 0; });

        d_flags[0] = 1;	

        uint num_groups = thrust::count(d_flags.begin(), d_flags.end(), 1);

        thrust::device_vector<int> d_starts(num_groups + 1, h_bufferinput_size);

        thrust::copy_if(thrust::device,
            d_indices.begin(), d_indices.end(),
            d_flags.begin(), d_starts.begin(),
            thrust::identity<int>());

        int* d_start_ptr = thrust::raw_pointer_cast(d_starts.data());

        // free up
        thrust::device_vector<int>().swap(d_indices);
        thrust::device_vector<int>().swap(pair_ids);
        thrust::device_vector<int>().swap(d_flags);

        int *d_end_ptr = nullptr; 
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_end_ptr, (num_groups + 1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(d_end_ptr, d_start_ptr, (num_groups + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

        float *d_suffix_min = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_suffix_min, h_bufferinput_size * sizeof(float)));

        grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;
        preprocess_suffixmin<<<grid_size, block_size>>>(d_pixpairs, d_start_ptr, num_groups, d_suffix_min);
        check_execution("preprocess_suffixmin"); 

        while(true){
            grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_merge<<<grid_size, block_size>>>(d_pixpairs, d_start_ptr,
                d_end_ptr, d_suffix_min, num_groups,
                (PixPair *)gctx->d_BufferOutput, gctx->d_bufferoutput_size,
                d_max_box_dist, gctx->merge_threshold);
            check_execution("kernel_merge"); 

            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            if(h_bufferoutput_size == 0) break;

            CUDA_SWAP_BUFFER();

            grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_unroll_within_polygon<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_farm_offset, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, static_cast<uint>(gctx->unroll_size));
            check_execution("kernel_unroll_within_polygon");
    
            CUDA_SWAP_BUFFER();

            grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_refine_within_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, d_max_box_dist, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr, within_distance);
            check_execution("kernel_refine_within_polygon");
        }

        CUDA_SAFE_CALL(cudaFree(d_end_ptr));
        CUDA_SAFE_CALL(cudaFree(d_suffix_min));
        CUDA_SAFE_CALL(cudaFree(d_pixpairs));
      }

        grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        statistic_result_polygon<<<grid_size, block_size>>>(
            d_max_box_dist, batch_size, gctx->d_result, within_distance);
        check_execution("statistic_result");

        uint h_result;
        CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
        gctx->found += h_result;

    }
    CUDA_SAFE_CALL(cudaFree(d_max_box_dist));
    return;
}
