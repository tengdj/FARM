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
};

__global__ void kernel_init(pair<IdealOffset, IdealOffset> *d_pairs, Idealinfo *d_info, uint size, double *distance, double *min_box_dist, double *max_box_dist)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        pair<IdealOffset, IdealOffset> &pair = d_pairs[pair_id];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;
        box &s_mbr = d_info[source.info_start].mbr;
        box &t_mbr = d_info[target.info_start].mbr;

        distance[pair_id] = gpu_max_distance(s_mbr, t_mbr);
        min_box_dist[pair_id] = DBL_MAX;
        max_box_dist[pair_id] = DBL_MAX;
    }
}

__global__ void cal_box_distance(pair<IdealOffset, IdealOffset> *pairs, Idealinfo *info, uint8_t *status, double *min_box_dist, double *max_box_dist, uint size, BoxDistRange *buffer, uint *buffer_size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        pair<IdealOffset, IdealOffset> &pair = pairs[pair_id];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;

        box &s_mbr = info[source.info_start].mbr, &t_mbr = info[target.info_start].mbr;
        const double &s_step_x = info[source.info_start].step_x, &s_step_y = info[source.info_start].step_y;
        const int &s_dimx = info[source.info_start].dimx, &s_dimy = info[source.info_start].dimy;
        const double &t_step_x = info[target.info_start].step_x, &t_step_y = info[target.info_start].step_y;
        const int &t_dimx = info[target.info_start].dimx, &t_dimy = info[target.info_start].dimy;

        for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i++)
        {
            for (int j = 0; j < (t_dimx + 1) * (t_dimy + 1); j++)
            {
                if (gpu_show_status(status, source.status_start, i) == BORDER && gpu_show_status(status, target.status_start, j) == BORDER)
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

__global__ void kernel_filter(BoxDistRange *bufferinput, double *min_box_dist, double *max_box_dist, uint *size, PixPair *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        double left = bufferinput[bufferId].minDist;
        int pairId = bufferinput[bufferId].pairId;
        // printf("left = %lf\n", left);

        if (left < max_box_dist[pairId])
        {
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = {bufferinput[bufferId].sourcePixelId, bufferinput[bufferId].targetPixelId, pairId};
        }
    }
}

__global__ void kernel_unroll(PixPair *pixpairs, pair<IdealOffset, IdealOffset> *pairs, uint16_t *offset, EdgeSeq *edge_sequences, uint *size, Batch *batches, uint *batch_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int pairId = pixpairs[bufferId].pair_id;
        int p = pixpairs[bufferId].source_pixid;
        int p2 = pixpairs[bufferId].target_pixid;

        IdealOffset &source = pairs[pairId].first;
        IdealOffset &target = pairs[pairId].second;

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

    pair<IdealOffset, IdealOffset> *h_pairs = new pair<IdealOffset, IdealOffset>[size];
    pair<IdealOffset, IdealOffset> *d_pairs = nullptr;

    for (int i = 0; i < size; i++)
    {
        Ideal *source = gctx->polygon_pairs[i].first;
        Ideal *target = gctx->polygon_pairs[i].second;
        h_pairs[i] = {*source->idealoffset, *target->idealoffset};
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, size * sizeof(pair<IdealOffset, IdealOffset>)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(pair<IdealOffset, IdealOffset>), cudaMemcpyHostToDevice));

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

    grid_size_x = (size + 512 - 1) / 512;
    block_size.x = 512;
    grid_size.x = grid_size_x;

    timer.startTimer();

    cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, d_min_box_dist, d_max_box_dist, size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("cal_box_distance");

    timer.stopTimer();
    printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

    /* To delete  */
    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

    printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
    double h_min_box_dist, h_max_box_dist;
    CUDA_SAFE_CALL(cudaMemcpy(&h_min_box_dist, d_min_box_dist, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_max_box_dist, d_max_box_dist, sizeof(double), cudaMemcpyDeviceToHost));
    /*   To delete  */

    /*can be packaged*/
    swap(d_BufferInput, d_BufferOutput);
    swap(d_bufferinput_size, d_bufferoutput_size);
    swap(h_bufferinput_size, h_bufferoutput_size);
    CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));
    /*can be packaged*/

    grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_size.x = BLOCK_SIZE;
    grid_size.x = grid_size_x;

    timer.startTimer();

    kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_min_box_dist, d_max_box_dist, d_bufferinput_size, (PixPair *)d_BufferOutput, d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_filter");

    timer.stopTimer();
    printf("kernel filter: %f ms\n", timer.getElapsedTime());

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

    kernel_unroll<<<grid_size, block_size>>>((PixPair *)d_BufferInput, d_pairs, gctx->d_offset, gctx->d_edge_sequences, d_bufferinput_size, (Batch *)d_BufferOutput, d_bufferoutput_size);
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