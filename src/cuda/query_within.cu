#include "geometry.cuh"

#define WITHIN_DISTANCE 10

struct Task
{
    uint s_start = 0;
    uint s_length = 0;
    int pair_id = 0;
};

struct BoxDistRange
{
    int sourcePixelId;
    int pairId;
    float minDist;
    float maxDist;

    void print() const {
        printf("%d %d %f %f\n", sourcePixelId, pairId, minDist, maxDist);
    }
};

__device__ int d_level;

__global__ void kernel_init(BoxDistRange *buffer, float *max_box_dist, uint size, uint8_t *flags)
{
	const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair_id < size)
	{
        buffer[pair_id] = {0, pair_id, 0.0, FLT_MAX};
        max_box_dist[pair_id] = flags[pair_id] ? FLT_MAX : -1.0f;
        // if(flags[pair_id]) max_box_dist[pair_id] = FLT_MAX;
        // else max_box_dist[pair_id] = -2.0;	
	}
}

// __global__ void 
// cal_box_distance(IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, double *min_box_dist, double *max_box_dist, uint size, BoxDistRange *buffer, uint *buffer_size, double *degree_per_kilometer_latitude, double *degree_per_kilometer_longitude_arr)
// {
//     const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (pair_id < size)
//     {
//         IdealPair &pair = pairs[pair_id];
//         IdealOffset &source = idealoffset[pair.source];
//         Point &target = points[pair.target];     

//         box &s_mbr = info[pair.source].mbr;
//         const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y;
//         const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;
//         assert(s_dimx > 0 && s_dimy > 0);

//         for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i ++)
//         {
//             uint uidx;
// 			// printf("STATUS: %d\n", gpu_show_status(status, source.status_start, i, source_offset));
// 			if (gpu_show_status(status, source.status_start, i) == BORDER)
// 			{
// 				auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
// 				double min_distance = gpu_distance(source_box, target, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
// 				double max_distance = gpu_max_distance(target, source_box, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
// 				uint idx = atomicAdd(buffer_size, 1);
//                 assert(idx < BUFFER_MAX_SIZE);
// 				buffer[idx] = {i, min_distance, max_distance, pair_id};
// 				atomicMinDouble(min_box_dist + pair_id, min_distance);
// 				atomicMinDouble(max_box_dist + pair_id, max_distance);
//                 uidx = idx;
// 			}
//         }
//     }
// }


__global__ void cal_box_distance(BoxDistRange *candidate, pair<uint32_t, uint32_t> *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, float *max_box_dist, uint *size, BoxDistRange *buffer, uint *buffer_size, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(candidate_id == 0) std::printf("d_level %d\n", d_level);
    if (candidate_id >= *size) return;
    
    int source_pixel_id = candidate[candidate_id].sourcePixelId;
    int pair_id = candidate[candidate_id].pairId;

    pair<uint32_t, uint32_t> &pair = pairs[pair_id];
    IdealOffset &source = idealoffset[pair.first];
    Point &target = points[pair.second];
    uint s_level = idealoffset[pair.first + 1].layer_start - source.layer_start - 1;

    int source_start_x, source_start_y, source_end_x, source_end_y;
    uint32_t source_offset;
    box s_mbr;
    double s_step_x, s_step_y;
    int s_dimx, s_dimy;

    if(d_level > s_level){
        int idx = atomicAdd(buffer_size, 1);
        buffer[idx] = candidate[candidate_id];
        return;
    }

    source_offset = (layer_offset + source.layer_start)[d_level];
    s_mbr = (layer_info + source.layer_start)[d_level].mbr;
    s_step_x = (layer_info + source.layer_start)[d_level].step_x, s_step_y = (layer_info + source.layer_start)[d_level].step_y;
    s_dimx = (layer_info + source.layer_start)[d_level].dimx, s_dimy = (layer_info + source.layer_start)[d_level].dimy;

    box source_pixel_box = gpu_get_pixel_box(
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

    for(int x = source_start_x; x <= source_end_x; x++){
        for(int y = source_start_y; y <= source_end_y; y++){
            int id = gpu_get_id(x, y, s_dimx);
            
            auto bx = gpu_get_pixel_box(x, y, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
            double min_distance = gpu_distance(bx, target, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
            double max_distance = gpu_max_distance(target, bx, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
            
            if (gpu_show_status(status, source.status_start, id, source_offset) == BORDER) {
                auto bx = gpu_get_pixel_box(x, y, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                float min_distance = gpu_distance(bx, target, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                float max_distance = gpu_max_distance(target, bx, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                if(max_distance <= WITHIN_DISTANCE){
                    atomicMinFloat(max_box_dist + pair_id, -1.0f);
                    return;
                }
                if(min_distance > WITHIN_DISTANCE) continue;
                
                uint idx = atomicAdd(buffer_size, 1);
                buffer[idx] = {id, pair_id, min_distance, max_distance};
                atomicMinFloat(max_box_dist + pair_id, max_distance);
            }
        }
    }
}

__global__ void kernel_filter_within(BoxDistRange *bufferinput, float *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId >= *size) return;

    double left = bufferinput[bufferId].minDist;
    int pairId = bufferinput[bufferId].pairId;
    
    if (left < max_box_dist[pairId]) {
        int idx = atomicAdd(bufferoutput_size, 1);
        bufferoutput[idx] = bufferinput[bufferId];
    }
}

__global__ void kernel_unroll(BoxDistRange *pixpairs, pair<uint32_t, uint32_t> *pairs, Point *points, IdealOffset *idealoffset, uint32_t *es_offset, EdgeSeq *edge_sequences, uint *size, Task *tasks, uint *task_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId >= *size) return;
    
    int pairId = pixpairs[bufferId].pairId;
    int p = pixpairs[bufferId].sourcePixelId;

    pair<uint32_t, uint32_t> &pair = pairs[pairId];
    IdealOffset &source = idealoffset[pair.first];

    int s_num_sequence = (es_offset + source.offset_start)[p + 1] - (es_offset + source.offset_start)[p];

    for (int i = 0; i < s_num_sequence; ++i)
    {
        EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[p] + i];
        int max_size = 16;
        for (uint s = 0; s < r.length; s += max_size)
        {
            uint end_s = min(s + max_size, r.length);
            uint idx = atomicAdd(task_size, 1U);

            tasks[idx].s_start = source.vertices_start + r.start + s;
            tasks[idx].s_length = end_s - s;
            tasks[idx].pair_id = pairId;
        }
    }
}

__global__ void kernel_refine(Task *tasks, pair<uint32_t, uint32_t> *pairs, Point *points, Point *vertices, uint *size, float *max_box_dist,  float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr)
{
    const int taskId = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskId >= *size) return;

    uint s = tasks[taskId].s_start;
    uint len = tasks[taskId].s_length;
    int pair_id = tasks[taskId].pair_id;

    pair<uint32_t, uint32_t> &pair = pairs[pair_id];
    Point &target = points[pair.second];

    double dist = gpu_point_to_segment_within_batch(target, vertices + s, len, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
    if(dist <= WITHIN_DISTANCE){
        atomicMinFloat(max_box_dist + pair_id, -1.0f); 
    }else{
        atomicMinFloat(max_box_dist + pair_id, dist);
    }
}

__global__ void statistic_result(float *max_box_dist, uint size, uint *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size && max_box_dist[x] < 0.0f)
    {
        atomicAdd(result, 1);
    }
}

void cuda_within(query_context *gctx)
{
    size_t batch_size = gctx->index_end - gctx->index;
	uint h_bufferinput_size, h_bufferoutput_size; 
 
    float *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, batch_size * sizeof(float)));

    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_init<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, batch_size, gctx->d_flags);
    cudaDeviceSynchronize();
    check_execution("kernel init");

    h_bufferinput_size = batch_size;
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, sizeof(uint), cudaMemcpyHostToDevice));

    int h_level = 0;
    while(true){
        h_level ++;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_level, &h_level, sizeof(int)));
        
        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_points, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr);
        cudaDeviceSynchronize();
        check_execution("cal_box_distance");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);

        if(h_bufferinput_size == h_bufferoutput_size) break;

        CUDA_SWAP_BUFFER();

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernel_filter_within<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter_within");

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

    kernel_unroll<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_points, gctx->d_idealoffset, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_unroll");

    CUDA_SWAP_BUFFER();

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_refine<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_points, gctx->d_vertices, gctx->d_bufferinput_size, d_max_box_dist, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr);
    cudaDeviceSynchronize();
    check_execution("kernel_refine");

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    statistic_result<<<grid_size, block_size>>>(d_max_box_dist, batch_size, gctx->d_result);
    cudaDeviceSynchronize();
    check_execution("statistic_result");

    uint h_result;
    CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
    gctx->found += h_result;

    CUDA_SAFE_CALL(cudaFree(d_max_box_dist));
	return;
}