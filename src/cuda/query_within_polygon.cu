#include "geometry.cuh"

#define WITHIN_DISTANCE 10

struct Batch
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
    double minDist;
    double maxDist; // maxDist is not nessnary
    int pairId;
    int level = 0;
};

__global__ void kernel_init(PolygonPair *d_pairs, RasterInfo *d_info, uint size, double *distance, double *min_box_dist, double *max_box_dist)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        PolygonPair &pair = d_pairs[pair_id];
        IdealOffset &source = pair.source;
        IdealOffset &target = pair.target;
        box &s_mbr = d_info[source.info_start].mbr;
        box &t_mbr = d_info[target.info_start].mbr;

        distance[pair_id] = gpu_max_distance(s_mbr, t_mbr);
        min_box_dist[pair_id] = DBL_MAX;
        max_box_dist[pair_id] = DBL_MAX;
    }
}

__global__ void first_cal_box_distance(PolygonPair *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint size, BoxDistRange *buffer, uint *buffer_size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        PolygonPair &pair = pairs[pair_id];
        IdealOffset &source = pair.source;
        IdealOffset &target = pair.target;     

        uint16_t source_offset = (layer_offset + source.layer_offset_start)[*global_level];
        uint16_t target_offset = (layer_offset + target.layer_offset_start)[*global_level];

        box &s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr, &t_mbr = (layer_info + target.layer_info_start)[*global_level].mbr;
        const double &s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, &s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
        const int &s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, &s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;
        const double &t_step_x = (layer_info + target.layer_info_start)[*global_level].step_x, &t_step_y = (layer_info + target.layer_info_start)[*global_level].step_y;
        const int &t_dimx = (layer_info + target.layer_info_start)[*global_level].dimx, &t_dimy = (layer_info + target.layer_info_start)[*global_level].dimy;

        for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i++)
        {
            for (int j = 0; j < (t_dimx + 1) * (t_dimy + 1); j++)
            {
                if (gpu_show_status(status, source.status_start, source_offset, i) == BORDER && gpu_show_status(status, target.status_start, target_offset, j) == BORDER)
                {
                    auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                    auto target_box = gpu_get_pixel_box(gpu_get_x(j, t_dimx), gpu_get_y(j, t_dimx, t_dimy), t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                    double min_distance = gpu_distance(source_box, target_box);
                    double max_distance = gpu_max_distance(source_box, target_box);
                    int idx = atomicAdd(buffer_size, 1);
                    buffer[idx] = {i, j, min_distance, max_distance, pair_id};
                    atomicMinDouble(min_box_dist + pair_id, min_distance);
                    atomicMinDouble(max_box_dist + pair_id, max_distance);
                }
            }
        }
    }
}

__global__ void cal_box_distance(BoxDistRange *candidate, PolygonPair *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint *size, BoxDistRange *buffer, uint *buffer_size)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate_id < *size)
    {
        int source_pixel_id = candidate[candidate_id].sourcePixelId;
        int target_pixel_id = candidate[candidate_id].targetPixelId;
        int pair_id = candidate[candidate_id].pairId;

        PolygonPair &pair = pairs[pair_id];
        IdealOffset &source = pair.source;
        IdealOffset &target = pair.target;
        uint s_level = pair.s_level;
        uint t_level = pair.t_level;

        if(*global_level > s_level && *global_level > t_level){
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = candidate[candidate_id];
            return;
        }

        int source_start_x, source_start_y, source_end_x, source_end_y, target_start_x, target_start_y, target_end_x, target_end_y;
        uint16_t source_offset, target_offset;
        box s_mbr, t_mbr;
        double s_step_x, s_step_y, t_step_x, t_step_y;
        int s_dimx, s_dimy, t_dimx, t_dimy;
        box source_pixel_box, target_pixel_box;

        if(*global_level > s_level) {
            source_offset = (layer_offset + source.layer_offset_start)[s_level];
            s_mbr = (layer_info + source.layer_info_start)[s_level].mbr;
            s_step_x = (layer_info + source.layer_info_start)[s_level].step_x, s_step_y = (layer_info + source.layer_info_start)[s_level].step_y;
            s_dimx = (layer_info + source.layer_info_start)[s_level].dimx, s_dimy = (layer_info + source.layer_info_start)[s_level].dimy;

            source_start_x = gpu_get_x(source_pixel_id, s_dimx);
            source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
            source_end_x = gpu_get_x(source_pixel_id, s_dimx);
            source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        }else{      
            source_offset = (layer_offset + source.layer_offset_start)[*global_level];
            s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr;
            s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
            s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;
            source_pixel_box = gpu_get_pixel_box(
                gpu_get_x(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx), 
                gpu_get_y(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx, (layer_info + source.layer_info_start)[*global_level-1].dimy),
                (layer_info + source.layer_info_start)[*global_level-1].mbr.low[0], (layer_info + source.layer_info_start)[*global_level-1].mbr.low[1],
                (layer_info + source.layer_info_start)[*global_level-1].step_x, (layer_info + source.layer_info_start)[*global_level-1].step_y);
            source_pixel_box.low[0] += 0.0001;
            source_pixel_box.low[1] += 0.0001;
            source_pixel_box.high[0] -= 0.0001;
            source_pixel_box.high[1] -= 0.0001;

            source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
            source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
            source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
            source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
        }

        if(*global_level > t_level){
            target_offset = (layer_offset + target.layer_offset_start)[t_level]; 
            t_mbr = (layer_info + target.layer_info_start)[t_level].mbr;  
            t_step_x = (layer_info + target.layer_info_start)[t_level].step_x, t_step_y = (layer_info + target.layer_info_start)[t_level].step_y;
            t_dimx = (layer_info + target.layer_info_start)[t_level].dimx, t_dimy = (layer_info + target.layer_info_start)[t_level].dimy;

            target_start_x = gpu_get_x(target_pixel_id, t_dimx);
            target_start_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
            target_end_x = gpu_get_x(target_pixel_id, t_dimx);
            target_end_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
        }else{
            target_offset = (layer_offset + target.layer_offset_start)[*global_level]; 
            t_mbr = (layer_info + target.layer_info_start)[*global_level].mbr;  
            t_step_x = (layer_info + target.layer_info_start)[*global_level].step_x, t_step_y = (layer_info + target.layer_info_start)[*global_level].step_y;
            t_dimx = (layer_info + target.layer_info_start)[*global_level].dimx, t_dimy = (layer_info + target.layer_info_start)[*global_level].dimy;
            target_pixel_box = gpu_get_pixel_box(
                gpu_get_x(target_pixel_id, (layer_info + target.layer_info_start)[*global_level-1].dimx), 
                gpu_get_y(target_pixel_id, (layer_info + target.layer_info_start)[*global_level-1].dimx, (layer_info + target.layer_info_start)[*global_level-1].dimy),
                (layer_info + target.layer_info_start)[*global_level-1].mbr.low[0], (layer_info + target.layer_info_start)[*global_level-1].mbr.low[1],
                (layer_info + target.layer_info_start)[*global_level-1].step_x, (layer_info + target.layer_info_start)[*global_level-1].step_y); 
            target_pixel_box.low[0] += 0.0001;
            target_pixel_box.low[1] += 0.0001;
            target_pixel_box.high[0] -= 0.0001;
            target_pixel_box.high[1] -= 0.0001;

            target_start_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.low[0], t_step_x, t_dimx);
            target_start_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.low[1], t_step_y, t_dimy);
            target_end_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.high[0], t_step_x, t_dimx);
            target_end_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.high[1], t_step_y, t_dimy);
        }

        for(int x1 = source_start_x; x1 <= source_end_x; x1 ++){
            for(int y1 = source_start_y; y1 <= source_end_y; y1 ++){
                int id1 = gpu_get_id(x1, y1, s_dimx);
                for(int x2 = target_start_x; x2 <= target_end_x; x2 ++){
                    for(int y2 = target_start_y; y2 <= target_end_y; y2 ++){
                        int id2 = gpu_get_id(x2, y2, t_dimx);
                        if (gpu_show_status(status, source.status_start, source_offset, id1) == BORDER && gpu_show_status(status, target.status_start, target_offset, id2) == BORDER){
                            // printf("block_id = %u thread_id = %u pairid = %d level = %d id1 = %d id2 = %d\n", blockIdx.x, threadIdx.x, pair_id, *global_level, id1, id2);
                            auto box1 = gpu_get_pixel_box(x1, y1, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                            auto box2 = gpu_get_pixel_box(x2, y2, t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                            double min_distance = gpu_distance(box1, box2);
                            double max_distance = gpu_max_distance(box1, box2);
                            int idx = atomicAdd(buffer_size, 1);
                            buffer[idx] = {id1, id2, min_distance, max_distance, pair_id};
                            atomicMinDouble(min_box_dist + pair_id, min_distance);
                            atomicMinDouble(max_box_dist + pair_id, max_distance);
                        }
                    }
                }
            }
        }
    }
}

__global__ void kernel_filter(BoxDistRange *bufferinput, double *min_box_dist, double *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        double left = bufferinput[bufferId].minDist;
        int pairId = bufferinput[bufferId].pairId;

        if (left < max_box_dist[pairId])
        {
            // printf("id1 = %d id2 = %d\n", bufferinput[bufferId].sourcePixelId, bufferinput[bufferId].targetPixelId);
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = bufferinput[bufferId];
        }
    }
}

__global__ void kernel_unroll(BoxDistRange *pixpairs, PolygonPair *pairs, uint16_t *offset, EdgeSeq *edge_sequences, uint *size, Batch *batches, uint *batch_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int pairId = pixpairs[bufferId].pairId;
        int p = pixpairs[bufferId].sourcePixelId;
        int p2 = pixpairs[bufferId].targetPixelId;

        IdealOffset &source = pairs[pairId].source;
        IdealOffset &target = pairs[pairId].target;

        int s_num_sequence = (offset + source.offset_start)[p + 1] - (offset + source.offset_start)[p];
        int t_num_sequence = (offset + target.offset_start)[p2 + 1] - (offset + target.offset_start)[p2];

        for (int i = 0; i < s_num_sequence; ++i)
        {
            EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(offset + source.offset_start)[p] + i];
            for (int j = 0; j < t_num_sequence; ++j)
            {
                EdgeSeq r2 = (edge_sequences + target.edge_sequences_start)[(offset + target.offset_start)[p2] + j];
                if (r.length < 2 || r2.length < 2)
                    continue;
                int max_size = 32;
                for (uint s = 0; s < r.length; s += max_size)
                {
                    uint end_s = min(s + max_size, r.length);
                    for (uint t = 0; t < r2.length; t += max_size)
                    {
                        uint end_t = min(t + max_size, r2.length);
                        uint idx = atomicAdd(batch_size, 1U);
                        batches[idx].s_start = source.vertices_start + r.start + s;
                        batches[idx].t_start = target.vertices_start + r2.start + t;
                        batches[idx].s_length = end_s - s;
                        batches[idx].t_length = end_t - t;
                        batches[idx].pair_id = pairId;
                    }
                }
            }
        }
    }
}

__global__ void kernel_refine(Batch *batches, Point *vertices, uint *size, double *distance)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        uint s1 = batches[bufferId].s_start;
        uint s2 = batches[bufferId].t_start;
        uint len1 = batches[bufferId].s_length;
        uint len2 = batches[bufferId].t_length;
        int pair_id = batches[bufferId].pair_id;

        double dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2, len1, len2);

        atomicMinDouble(distance + pair_id, dist);
    }
}

uint cuda_within_polygon(query_context *gctx)
{
    CudaTimer timer, duration;

    duration.startTimer();

    uint size = gctx->polygon_pairs.size();

    printf("SIZE = %u\n", size);

    PolygonPair *h_pairs = new PolygonPair[size];
    PolygonPair *d_pairs = nullptr;

    for (int i = 0; i < size; i++)
    {
        Ideal *source = gctx->polygon_pairs[i].first;
        Ideal *target = gctx->polygon_pairs[i].second;
        h_pairs[i] = {*source->idealoffset, *target->idealoffset, source->get_num_layers(), target->get_num_layers()};
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, size * sizeof(PolygonPair)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(PolygonPair), cudaMemcpyHostToDevice));

    double *h_distance = new double[size * sizeof(double)];
    double *d_distance = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_distance, size * sizeof(double)));

    double *d_min_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_min_box_dist, size * sizeof(double)));

    double *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, size * sizeof(double)));

    bool *h_resultmap = new bool[size * sizeof(bool)];
    bool *d_resultmap = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, size * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(bool)));
    memset(h_resultmap, 0, size * sizeof(bool));

    char *d_BufferInput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferInput, 4UL * 1024 * 1024 * 1024));
    uint *d_bufferinput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferinput_size, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_bufferinput_size, 0, sizeof(uint)));
    uint h_bufferinput_size;

    char *d_BufferOutput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferOutput, 4UL * 1024 * 1024 * 1024));
    uint *d_bufferoutput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferoutput_size, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));
    uint h_bufferoutput_size;

    int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, 1, 1);
    dim3 grid_size(grid_size_x, 1, 1);

    timer.startTimer();

    kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_info, size, d_distance, d_min_box_dist, d_max_box_dist);
    cudaDeviceSynchronize();
    check_execution("kernel init");

    timer.stopTimer();
    printf("kernel init: %f ms\n", timer.getElapsedTime());

    uint h_level = 0;
    uint *d_level = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_level, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_level, 0, sizeof(uint)));

    grid_size_x = (size + 512 - 1) / 512;
    block_size.x = 512;
    grid_size.x = grid_size_x;

    timer.startTimer();
    first_cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_min_box_dist, d_max_box_dist, d_level, size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("first_cal_box_distance");
    timer.stopTimer();
    printf("kernel first calculate box distance: %f ms\n", timer.getElapsedTime());

    /* To delete  */
    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_bufferinput_size = %u\n", h_bufferinput_size);
    printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
    // double h_min_box_dist, h_max_box_dist;
    // CUDA_SAFE_CALL(cudaMemcpy(&h_min_box_dist, d_min_box_dist, sizeof(double), cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(&h_max_box_dist, d_max_box_dist, sizeof(double), cudaMemcpyDeviceToHost));
    /*   To delete  */

    while(true){
        h_level ++;
        CUDA_SAFE_CALL(cudaMemcpy(d_level, &h_level, sizeof(uint), cudaMemcpyHostToDevice));
        if(h_level > gctx->num_layers) break;

        swap(d_BufferInput, d_BufferOutput);
        swap(d_bufferinput_size, d_bufferoutput_size);
        swap(h_bufferinput_size, h_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        grid_size_x = (h_bufferinput_size + 512 - 1) / 512;
        block_size.x = 512;
        grid_size.x = grid_size_x;

        timer.startTimer();
        cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_min_box_dist, d_max_box_dist, d_level, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("cal_box_distance");
        timer.stopTimer();
        printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

        /* To delete  */
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);
        /* To delete  */

        if(h_bufferinput_size == h_bufferoutput_size) break;

        swap(d_BufferInput, d_BufferOutput);
        swap(d_bufferinput_size, d_bufferoutput_size);
        swap(h_bufferinput_size, h_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        timer.startTimer();
        kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_min_box_dist, d_max_box_dist, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter");
        timer.stopTimer();
        printf("kernel filter: %f ms\n", timer.getElapsedTime());

        /* To delete  */
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("filter h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("filter h_bufferoutput_size = %u\n", h_bufferoutput_size);
        /*   To delete  */
    }

    swap(d_BufferInput, d_BufferOutput);
    swap(d_bufferinput_size, d_bufferoutput_size);
    swap(h_bufferinput_size, h_bufferoutput_size);
    CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

    grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_size.x = BLOCK_SIZE;
    grid_size.x = grid_size_x;

    timer.startTimer();

    kernel_unroll<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_pairs, gctx->d_offset, gctx->d_edge_sequences, d_bufferinput_size, (Batch *)d_BufferOutput, d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_unroll");

    timer.stopTimer();
    printf("kernel unroll: %f ms\n", timer.getElapsedTime());

    /* To delete  */
    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

    printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
    /*   To delete  */

    swap(d_BufferInput, d_BufferOutput);
    swap(d_bufferinput_size, d_bufferoutput_size);
    swap(h_bufferinput_size, h_bufferoutput_size);
    CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

    grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_size.x = BLOCK_SIZE;
    grid_size.x = grid_size_x;

    timer.startTimer();

    kernel_refine<<<grid_size, block_size>>>((Batch *)d_BufferInput, gctx->d_vertices, d_bufferinput_size, d_distance);
    cudaDeviceSynchronize();
    check_execution("kernel_refine");

    timer.stopTimer();
    printf("kernel refine: %f ms\n", timer.getElapsedTime());

    duration.stopTimer();
    printf("kernel total time = %lf ms\n", duration.getElapsedTime());

    CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
    int found = 0;
    for (int i = 0; i < size; i++)
    {
        if (h_distance[i] <= WITHIN_DISTANCE)
            found++;
        printf("%lf\n", h_distance[i]);
    }

    return found;
}