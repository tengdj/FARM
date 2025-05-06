#include "geometry.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define WITHIN_DISTANCE 10

unordered_map<int, int> ht;

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

struct BoxDistRange
{
    int sourcePixelId;
    int targetPixelId;
    int pairId;
    float minDist;
    float maxDist;

    void print() const {
        printf("%d %d %d %f %f\n", sourcePixelId, targetPixelId, pairId, minDist, maxDist);
    }
};

__device__ int d_level;

__global__ void kernel_init_distance(pair<uint32_t, uint32_t> *pairs, uint source_size, BoxDistRange *buffer, float *max_box_dist, uint size, uint8_t *flags)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        // target id follows source id
        pairs[pair_id].second += source_size;
        buffer[pair_id] = {0, 0, pair_id, 0.0, FLT_MAX};
        // max_box_dist[pair_id] = flags[pair_id] == 2 ? -1.0f : FLT_MAX;
        max_box_dist[pair_id] = FLT_MAX;
    }
}

__global__ void cal_box_distance_polygon(BoxDistRange *candidate, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, float *max_box_dist, uint *size, BoxDistRange *buffer, uint *buffer_size, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(candidate_id == 0) printf("d_level %d\n", d_level);
    if (candidate_id < *size)
    {
        int source_pixel_id = candidate[candidate_id].sourcePixelId;
        int target_pixel_id = candidate[candidate_id].targetPixelId;
        int pair_id = candidate[candidate_id].pairId;

        pair<uint32_t, uint32_t> &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.first];
        IdealOffset &target = idealoffset[pair.second];
        uint s_level = idealoffset[pair.first + 1].layer_start - source.layer_start - 1;
        uint t_level = idealoffset[pair.second + 1].layer_start - target.layer_start - 1;

        int source_start_x, source_start_y, source_end_x, source_end_y, target_start_x, target_start_y, target_end_x, target_end_y;
        uint32_t source_offset, target_offset;
        box s_mbr, t_mbr;
        double s_step_x, s_step_y, t_step_x, t_step_y;
        int s_dimx, s_dimy, t_dimx, t_dimy;
        box source_pixel_box, target_pixel_box;
        if (d_level > s_level)
        {
            source_offset = (layer_offset + source.layer_start)[s_level];
            s_mbr = (layer_info + source.layer_start)[s_level].mbr;
            s_step_x = (layer_info + source.layer_start)[s_level].step_x, s_step_y = (layer_info + source.layer_start)[s_level].step_y;
            s_dimx = (layer_info + source.layer_start)[s_level].dimx, s_dimy = (layer_info + source.layer_start)[s_level].dimy;

            source_start_x = gpu_get_x(source_pixel_id, s_dimx);
            source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
            source_end_x = gpu_get_x(source_pixel_id, s_dimx);
            source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        }
        else
        {
            source_offset = (layer_offset + source.layer_start)[d_level];
            s_mbr = (layer_info + source.layer_start)[d_level].mbr;
            s_step_x = (layer_info + source.layer_start)[d_level].step_x, s_step_y = (layer_info + source.layer_start)[d_level].step_y;
            s_dimx = (layer_info + source.layer_start)[d_level].dimx, s_dimy = (layer_info + source.layer_start)[d_level].dimy;

            source_pixel_box = gpu_get_pixel_box(
                gpu_get_x(source_pixel_id, (layer_info + source.layer_start)[d_level - 1].dimx),
                gpu_get_y(source_pixel_id, (layer_info + source.layer_start)[d_level - 1].dimx, (layer_info + source.layer_start)[d_level - 1].dimy),
                (layer_info + source.layer_start)[d_level - 1].mbr.low[0], (layer_info + source.layer_start)[d_level - 1].mbr.low[1],
                (layer_info + source.layer_start)[d_level - 1].step_x, (layer_info + source.layer_start)[d_level - 1].step_y);
            source_pixel_box.low[0] += 0.000001;
            source_pixel_box.low[1] += 0.000001;
            source_pixel_box.high[0] -= 0.000001;
            source_pixel_box.high[1] -= 0.000001;

            source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
            source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
            source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
            source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
        }

        if (d_level > t_level)
        {
            target_offset = (layer_offset + target.layer_start)[t_level];
            t_mbr = (layer_info + target.layer_start)[t_level].mbr;
            t_step_x = (layer_info + target.layer_start)[t_level].step_x, t_step_y = (layer_info + target.layer_start)[t_level].step_y;
            t_dimx = (layer_info + target.layer_start)[t_level].dimx, t_dimy = (layer_info + target.layer_start)[t_level].dimy;

            target_start_x = gpu_get_x(target_pixel_id, t_dimx);
            target_start_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
            target_end_x = gpu_get_x(target_pixel_id, t_dimx);
            target_end_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
        }
        else
        {
            target_offset = (layer_offset + target.layer_start)[d_level];
            t_mbr = (layer_info + target.layer_start)[d_level].mbr;
            t_step_x = (layer_info + target.layer_start)[d_level].step_x, t_step_y = (layer_info + target.layer_start)[d_level].step_y;
            t_dimx = (layer_info + target.layer_start)[d_level].dimx, t_dimy = (layer_info + target.layer_start)[d_level].dimy;

            target_pixel_box = gpu_get_pixel_box(
                gpu_get_x(target_pixel_id, (layer_info + target.layer_start)[d_level - 1].dimx),
                gpu_get_y(target_pixel_id, (layer_info + target.layer_start)[d_level - 1].dimx, (layer_info + target.layer_start)[d_level - 1].dimy),
                (layer_info + target.layer_start)[d_level - 1].mbr.low[0], (layer_info + target.layer_start)[d_level - 1].mbr.low[1],
                (layer_info + target.layer_start)[d_level - 1].step_x, (layer_info + target.layer_start)[d_level - 1].step_y);
            target_pixel_box.low[0] += 0.000001;
            target_pixel_box.low[1] += 0.000001;
            target_pixel_box.high[0] -= 0.000001;
            target_pixel_box.high[1] -= 0.000001;

            target_start_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.low[0], t_step_x, t_dimx);
            target_start_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.low[1], t_step_y, t_dimy);
            target_end_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.high[0], t_step_x, t_dimx);
            target_end_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.high[1], t_step_y, t_dimy);
        }
        
        for (int x1 = source_start_x; x1 <= source_end_x; x1++)
        {
            for (int y1 = source_start_y; y1 <= source_end_y; y1++)
            {
                int id1 = gpu_get_id(x1, y1, s_dimx);
                for (int x2 = target_start_x; x2 <= target_end_x; x2++)
                {
                    for (int y2 = target_start_y; y2 <= target_end_y; y2++)
                    {
                        int id2 = gpu_get_id(x2, y2, t_dimx);
                        if (gpu_show_status(status, source.status_start, id1, source_offset) == BORDER && gpu_show_status(status, target.status_start, id2, target_offset) == BORDER)
                        {
                            auto box1 = gpu_get_pixel_box(x1, y1, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                            auto box2 = gpu_get_pixel_box(x2, y2, t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                            float min_distance = gpu_distance(box1, box2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                            float max_distance = gpu_max_distance(box1, box2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                            if(max_distance <= WITHIN_DISTANCE){
                                atomicMinFloat(max_box_dist + pair_id, -1.0f);
                                return;
                            }
                            if(min_distance > WITHIN_DISTANCE) continue;
            
                            uint idx = atomicAdd(buffer_size, 1);
                            buffer[idx] = {id1, id2, pair_id, min_distance, max_distance};
                            atomicMinFloat(max_box_dist + pair_id, max_distance);
                        }
                    }
                }
            }
        }
    }
}

__global__ void kernel_filter_within_polygon(BoxDistRange *bufferinput, float *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        float left = bufferinput[bufferId].minDist;
        int pairId = bufferinput[bufferId].pairId;
        if (left < max_box_dist[pairId])
        {
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = bufferinput[bufferId];
        }
    }
}

__global__ void statistic_size(BoxDistRange *pixpairs, uint size, uint *pixelpairidx){
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < size)
    {
        int pairId = pixpairs[bufferId].pairId;
        atomicAdd(pixelpairidx + pairId, 1);
    }
}

__global__ void mark_nonzeros(uint *input, uint *flags, uint size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        flags[tid] = (input[tid] != 0) ? 1 : 0;
    }
}

__global__ void compact_array(uint *input, uint *prefix_sum, uint *output, uint size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && input[tid]) {
        int pos = prefix_sum[tid]; // The prefix and the target location are given
        output[pos] = input[tid];  // Write non-zero elements to a compact array
    }
}

struct CompareKeyValuePairs {
    __host__ __device__
    bool operator()(const BoxDistRange& a, const BoxDistRange& b) const {
        if (a.pairId != b.pairId) {
            return a.pairId < b.pairId; 
        }
        return a.minDist < b.minDist; 
    }
};

__global__ void  kernel_merge(uint *pixelpairidx, uint *pixelpairsize, BoxDistRange *pixpairs, int pairsize, BoxDistRange* buffer, uint *buffer_size, float *max_box_dist)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairsize && pixelpairsize[tid + 1] - pixelpairidx[tid] > 0)
    {
        int start = pixelpairidx[tid];
        int end = pixelpairsize[tid + 1];
        int pairId = pixpairs[start].pairId;
        if(max_box_dist[pairId] < 0) return;

        float left = 100000.0, right = 100000.0;
        for(int i = start; i < end; i ++){
            float mind = pixpairs[i].minDist, maxd = pixpairs[i].maxDist;
            left = min(mind, left);
            right = min(maxd, right);
            float ratio = (right - mind) / (right - left);
            if(ratio < 0.95) {
                pixelpairidx[tid] = i;
                return;
            } 
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = pixpairs[i];
        }

        pixelpairidx[tid] = end;
    }
}

__global__ void kernel_unroll_within_polygon(BoxDistRange *pixpairs, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, uint32_t *es_offset, EdgeSeq *edge_sequences, uint* size, Task *tasks, uint *task_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int p = pixpairs[bufferId].sourcePixelId;
        int p2 = pixpairs[bufferId].targetPixelId;
        int pairId = pixpairs[bufferId].pairId;

        pair<uint32_t, uint32_t> &pair = pairs[pairId];
        IdealOffset &source = idealoffset[pair.first];
        IdealOffset &target = idealoffset[pair.second];

        int s_num_sequence = (es_offset + source.offset_start)[p + 1] - (es_offset + source.offset_start)[p];
        int t_num_sequence = (es_offset + target.offset_start)[p2 + 1] - (es_offset + target.offset_start)[p2];

        for (int i = 0; i < s_num_sequence; ++ i)
        {
            EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[p] + i];
            for (int j = 0; j < t_num_sequence; ++j)
            {
                EdgeSeq r2 = (edge_sequences + target.edge_sequences_start)[(es_offset + target.offset_start)[p2] + j];
                int max_size = 16;
                for (uint s = 0; s < r.length; s += max_size)
                {
                    uint end_s = min(s + max_size, r.length);
                    for (uint t = 0; t < r2.length; t += max_size)
                    {
                        uint end_t = min(t + max_size, r2.length);
                        uint idx = atomicAdd(task_size, 1U);
                        tasks[idx] = {source.vertices_start + r.start + s, target.vertices_start + r2.start + t, end_s - s, end_t - t, pairId};
                    }
                }
            }
        }
    }
}

__global__ void kernel_refine_within_polygon(Task *tasks, Point *vertices, uint *size, float *max_box_dist, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    const int taskId = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskId < *size)
    {
        uint s1 = tasks[taskId].s_start;
        uint s2 = tasks[taskId].t_start;
        uint len1 = tasks[taskId].s_length;
        uint len2 = tasks[taskId].t_length;
        int pair_id = tasks[taskId].pair_id;

        double dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2, len1, len2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
        if(dist <= WITHIN_DISTANCE){
            atomicMinFloat(max_box_dist + pair_id, -1.0f); 
            return;
        }

        atomicMinFloat(max_box_dist + pair_id, dist);
    }
}

__global__ void statistic_result_polygon(float *max_box_dist, uint size, uint *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size && max_box_dist[x] < 0.0f)
    {
        atomicAdd(result, 1);
    }
}

void cuda_within_polygon(query_context *gctx)
{
    size_t batch_size = gctx->index_end - gctx->index;
    uint h_bufferinput_size, h_bufferoutput_size;
    CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

    float *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, batch_size * sizeof(float)));

    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_init_distance<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->source_ideals.size(), (BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, batch_size, gctx->d_flags);
    cudaDeviceSynchronize();
    check_execution("kernel_init");

    h_bufferinput_size = batch_size;
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, sizeof(uint), cudaMemcpyHostToDevice));

    int h_level = 0;
    while(true){
        h_level ++;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_level, &h_level, sizeof(int)));

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cal_box_distance_polygon<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr);
        cudaDeviceSynchronize();
        check_execution("cal_box_distance_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);

        if(h_bufferinput_size == h_bufferoutput_size) break;

        CUDA_SWAP_BUFFER();

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_filter_within_polygon<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter_within_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("filter h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("filter h_bufferoutput_size = %u\n", h_bufferoutput_size);

        CUDA_SWAP_BUFFER();
    }

    CUDA_SWAP_BUFFER();

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_bufferinput_size = %u\n", h_bufferinput_size);

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint *d_input = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_input, batch_size * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_input, 0, batch_size * sizeof(uint)));
    uint *d_flags = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_flags, batch_size * sizeof(uint)));
    uint *d_prefix_sum = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_prefix_sum, batch_size * sizeof(uint)));
    int *h_count = new int();

    statistic_size<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, h_bufferinput_size, d_input);
    cudaDeviceSynchronize();
    check_execution("statistic_size");

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mark_nonzeros<<<grid_size, block_size>>>(d_input, d_flags, batch_size);
    cudaDeviceSynchronize();
    check_execution("mark_nonzeros");

    thrust::device_ptr<uint> d_flags_ptr(d_flags);
    thrust::device_ptr<uint> d_prefix_sum_ptr(d_prefix_sum);
    thrust::exclusive_scan(d_flags_ptr, d_flags_ptr + batch_size, d_prefix_sum_ptr);

    int count;
    cudaMemcpy(&count, d_prefix_sum + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count, d_flags + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    count += *h_count; // Is the last element non-zero?

    uint *d_pixelpairidx = nullptr; 
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairidx, (count+1) * sizeof(uint)));
    uint *d_pixelpairsize = nullptr; 
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairsize, (count+1) * sizeof(uint)));

    compact_array<<<grid_size, block_size>>>(d_input, d_prefix_sum, d_pixelpairidx, batch_size);
    cudaDeviceSynchronize();
    check_execution("compact_array");

    delete h_count;
    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_flags));
    CUDA_SAFE_CALL(cudaFree(d_prefix_sum));

    thrust::device_ptr<uint> scan_begin = thrust::device_pointer_cast(d_pixelpairidx);
    thrust::device_ptr<uint> scan_end = thrust::device_pointer_cast(d_pixelpairidx + count + 1);

    thrust::exclusive_scan(scan_begin, scan_end, scan_begin);

    CUDA_SAFE_CALL(cudaMemcpy(d_pixelpairsize, d_pixelpairidx, (count+1) * sizeof(uint), cudaMemcpyDeviceToDevice));

    thrust::device_ptr<BoxDistRange> begin = thrust::device_pointer_cast((BoxDistRange*)gctx->d_BufferInput);
    thrust::device_ptr<BoxDistRange> end = thrust::device_pointer_cast((BoxDistRange*)gctx->d_BufferInput + h_bufferinput_size);
    thrust::sort(thrust::device, begin, end, CompareKeyValuePairs());

    BoxDistRange* d_pixpairs = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixpairs, h_bufferinput_size * sizeof(BoxDistRange)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pixpairs, gctx->d_BufferInput, h_bufferinput_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));

    while(true){
        printf("count = %d\n", count);

        grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_merge<<<grid_size, block_size>>>(d_pixelpairidx, d_pixelpairsize, d_pixpairs, count, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, d_max_box_dist);
        cudaDeviceSynchronize();
        check_execution("kernel_merge");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("h_bufferoutput_size = %d\n", h_bufferoutput_size);

        if(h_bufferoutput_size == 0) break;

        CUDA_SWAP_BUFFER();

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_unroll_within_polygon<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_unroll_within_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("h_bufferoutput_size = %d\n", h_bufferoutput_size);

        CUDA_SWAP_BUFFER();

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_refine_within_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, d_max_box_dist, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr);
        cudaDeviceSynchronize();
        check_execution("kernel_refine_within_polygon");
    }

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    statistic_result_polygon<<<grid_size, block_size>>>(d_max_box_dist, batch_size, gctx->d_result);
    cudaDeviceSynchronize();
    check_execution("statistic_result");

    uint h_result;
    CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
    gctx->found += h_result;

    CUDA_SAFE_CALL(cudaFree(d_pixelpairidx));
    CUDA_SAFE_CALL(cudaFree(d_pixelpairsize));
    CUDA_SAFE_CALL(cudaFree(d_pixpairs));
    CUDA_SAFE_CALL(cudaFree(d_max_box_dist));

    return;
}