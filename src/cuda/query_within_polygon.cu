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

__global__ void kernel_init(pair<IdealOffset, IdealOffset> *d_pairs, Idealinfo *d_info, uint size, double *distance)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        pair<IdealOffset, IdealOffset> &pair = d_pairs[x];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;
        box &s_mbr = d_info[source.info_start].mbr;
        box &t_mbr = d_info[target.info_start].mbr;

        distance[x] = gpu_max_distance(s_mbr, t_mbr);
    }
}

// Input：PolygonPairs中的两个polygon，轮次为step，
// Output：当前step，所有pixel和polygon的组合
__global__ void kernel_1(pair<IdealOffset, IdealOffset> *d_pairs, Idealinfo *d_info, uint8_t *d_status, PixMapping *d_pixpolypairs, uint *buffer_size, uint size, int *step, bool *resultmap)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        // if(resultmap[pair_id] != 0) return;
        pair<IdealOffset, IdealOffset> &pair = d_pairs[pair_id];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;

        box &s_mbr = d_info[source.info_start].mbr, &t_mbr = d_info[target.info_start].mbr;
        const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
        const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

        /*这一部分是多余的*/

        int lowx = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx);
        int lowy = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy);
        int highx = gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx);
        int highy = gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy);

        /*应该放在循环外面，不需要每次都计算*/

        if (*step == 0)
        {
            for (int x = lowx; x <= highx; x++)
            {
                for (int y = lowy; y <= highy; y++)
                {
                    int id = gpu_get_id(x, y, s_dimx);
                    assert((id / (s_dimx + 1)) <= s_dimy);
                    int idx = atomicAdd(buffer_size, 1);
                    d_pixpolypairs[idx] = {pair_id, id};
                }
            }
        }
        else
        {
            int ymin = max(0, lowy - *step);
            int ymax = min(s_dimy, highy + *step);

            // left scan
            if (lowx - *step >= 0)
            {
                for (int y = ymin; y <= ymax; y++)
                {
                    int id = gpu_get_id(lowx - *step, y, s_dimx);
                    assert((id / (s_dimx + 1)) <= s_dimy);
                    if (gpu_show_status(d_status, source.status_start, id) == BORDER)
                    {
                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpolypairs[idx] = {pair_id, id};
                    }
                }
            }
            // right scan
            if (highx + *step <= s_dimx)
            {
                for (int y = ymin; y <= ymax; y++)
                {
                    int id = gpu_get_id(highx + *step, y, s_dimx);
                    assert((id / (s_dimx + 1)) <= s_dimy);
                    if (gpu_show_status(d_status, source.status_start, id) == BORDER)
                    {
                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpolypairs[idx] = {pair_id, id};
                    }
                }
            }
            // skip the first if there is left scan
            int xmin = max(0, lowx - *step + (lowx - *step >= 0));
            // skip the last if there is right scan
            int xmax = min(s_dimx, highx + *step - (highx + *step <= s_dimx));

            // bottom scan
            if (lowy - *step >= 0)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    int id = gpu_get_id(x, lowy - *step, s_dimx);
                    assert((id / (s_dimx + 1)) <= s_dimy);
                    if (gpu_show_status(d_status, source.status_start, id) == BORDER)
                    {
                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpolypairs[idx] = {pair_id, id};
                    }
                }
            }
            // top scan
            if (highy + *step <= s_dimy)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    int id = gpu_get_id(x, highy + *step, s_dimx);
                    assert((id / (s_dimx + 1)) <= s_dimy);
                    if (gpu_show_status(d_status, source.status_start, id) == BORDER)
                    {
                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpolypairs[idx] = {pair_id, id};
                    }
                }
            }
        }
    }
}

// Input：当前step，所有pixel和polygon的组合
// Output：当前step，当前inner_step，所有pixel和pixel的组合

__global__ void kernel_2_1(PixMapping *d_pixpolypairs, pair<IdealOffset, IdealOffset> *d_pairs, Idealinfo *d_info, uint8_t *d_status, int *step, uint *size, PixPair *d_pixpairs, uint *buffer_size, bool *resultmap)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < *size)
    {
        int pair_id = d_pixpolypairs[x].pair_id;
        int cur = d_pixpolypairs[x].pix_id;
        // if(resultmap[pair_id]) return;

        pair<IdealOffset, IdealOffset> &pair = d_pairs[pair_id];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;

        box &s_mbr = d_info[source.info_start].mbr, &t_mbr = d_info[target.info_start].mbr;
        const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
        const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;
        const double &t_step_x = d_info[target.info_start].step_x, &t_step_y = d_info[target.info_start].step_y;
        const int &t_dimx = d_info[target.info_start].dimx, &t_dimy = d_info[target.info_start].dimy;

        if((cur / (s_dimx+1)) > s_dimy){
            printf("cur = %d, dimx = %d, dimy = %d\n", cur, s_dimx, s_dimy);
           assert((cur / (s_dimx+1)) <= s_dimy);
        }
        auto pix_box = gpu_get_pixel_box(gpu_get_x(cur, s_dimx), gpu_get_y(cur, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
        

        int lowx = gpu_get_offset_x(t_mbr.low[0], pix_box.low[0], t_step_x, t_dimx);
        int lowy = gpu_get_offset_y(t_mbr.low[1], pix_box.low[1], t_step_y, t_dimy);
        int highx = gpu_get_offset_x(t_mbr.low[0], pix_box.high[0], t_step_x, t_dimx);
        int highy = gpu_get_offset_y(t_mbr.low[1], pix_box.high[1], t_step_y, t_dimy);

        if (*step == 0)
        {
            for (int x = lowx; x <= highx; x++)
            {
                for (int y = lowy; y <= highy; y++)
                {
                    int id = gpu_get_id(x, y, t_dimx);
                    int idx = atomicAdd(buffer_size, 1);
                    d_pixpairs[idx] = {cur, id, pair_id};
                }
            }
        }
        else
        {

            int ymin = max(0, lowy - *step);
            int ymax = min(t_dimy, highy + *step);

            // left scan
            if (lowx - *step >= 0)
            {
                for (int y = ymin; y <= ymax; y++)
                {
                    int id = gpu_get_id(lowx - *step, y, t_dimx);
                    if (gpu_show_status(d_status, target.status_start, id) == BORDER)
                    {
                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpairs[idx] = {cur, id, pair_id};
                    }
                }
            }
            // right scan
            if (highx + *step <= t_dimx)
            {
                for (int y = ymin; y <= ymax; y++)
                {
                    int id = gpu_get_id(highx + *step, y, t_dimx);
                    if (gpu_show_status(d_status, target.status_start, id) == BORDER)
                    {

                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpairs[idx] = {cur, id, pair_id};
                    }
                }
            }
            // skip the first if there is left scan
            int xmin = max(0, lowx - *step + (lowx - *step >= 0));
            // skip the last if there is right scan
            int xmax = min(t_dimx, highx + *step - (highx + *step <= t_dimx));

            // bottom scan
            if (lowy - *step >= 0)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    int id = gpu_get_id(x, lowy - *step, t_dimx);
                    if (gpu_show_status(d_status, target.status_start, id) == BORDER)
                    {

                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpairs[idx] = {cur, id, pair_id};
                    }
                }
            }
            // top scan
            if (highy + *step <= t_dimy)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    int id = gpu_get_id(x, highy + *step, t_dimx);
                    if (gpu_show_status(d_status, target.status_start, id) == BORDER)
                    {

                        int idx = atomicAdd(buffer_size, 1);
                        d_pixpairs[idx] = {cur, id, pair_id};
                    }
                }
            }
        }
    }
}

__global__ void kernel_2_2(PixPair *d_pixpairs, pair<IdealOffset, IdealOffset> *d_pairs, uint16_t *d_offset, EdgeSeq *d_edge_sequences, uint *size, Batch *batches, uint *batch_size, bool *resultmap)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < *size)
    {
        int pair_id = d_pixpairs[x].pair_id;
        int p = d_pixpairs[x].source_pixid;
        int p2 = d_pixpairs[x].target_pixid;
        if (resultmap[pair_id])
            return;

        pair<IdealOffset, IdealOffset> &pair = d_pairs[pair_id];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;

        int s_num_sequence = (d_offset + source.offset_start)[p + 1] - (d_offset + source.offset_start)[p];
        int t_num_sequence = (d_offset + target.offset_start)[p2 + 1] - (d_offset + target.offset_start)[p2];

        for (int i = 0; i < s_num_sequence; ++i)
        {
            EdgeSeq r = (d_edge_sequences + source.edge_sequences_start)[(d_offset + source.offset_start)[p] + i];
            for (int j = 0; j < t_num_sequence; ++j)
            {
                EdgeSeq r2 = (d_edge_sequences + target.edge_sequences_start)[(d_offset + target.offset_start)[p2] + j];
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
                        batches[idx].pair_id = pair_id;
                    }
                }
            }
        }
    }
}

__global__ void kernel_2_3(Batch *batches, Point *d_vertices, uint *size, double *distance, bool *resultmap)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < *size)
    {
        uint s1 = batches[x].s_start;
        uint s2 = batches[x].t_start;
        uint len1 = batches[x].s_length;
        uint len2 = batches[x].t_length;
        int pair_id = batches[x].pair_id;
        if (resultmap[pair_id] != 0)
            return;

        double dist = gpu_segment_to_segment_within_batch(d_vertices + s1, d_vertices + s1, len1, len2);

        atomicMinDouble(distance + pair_id, dist);

        if (distance[pair_id] <= WITHIN_DISTANCE)
        {
            resultmap[pair_id] = true;
            return;
        }
    }
}

__global__ void kernel_3(pair<IdealOffset, IdealOffset> *d_pairs, Idealinfo *d_info, int *step, uint size, double *distance, bool *resultmap)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        if (resultmap[x])
            return;

        pair<IdealOffset, IdealOffset> &pair = d_pairs[x];
        IdealOffset &source = pair.first;
        IdealOffset &target = pair.second;

        box &s_mbr = d_info[source.info_start].mbr, &t_mbr = d_info[target.info_start].mbr;
        const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
        const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

        // min_possible = mbrdist + step * step_size
        double min_possible = gpu_distance(s_mbr, t_mbr) + (*step) * gpu_get_step(s_mbr, s_dimx, s_dimy);
        if (distance[x] <= min_possible)
        {
            resultmap[x] = true;
            return;
        }
    }
}

uint cuda_within_polygon(query_context *gctx)
{
    CudaTimer timer;

    float sum_filter = 0.0;
    float sum_unroll = 0.0;
    float sum_refinement = 0.0;
    float sum_check = 0.0;

    uint size = gctx->polygon_pairs.size();

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

    bool *d_resultmap = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, size * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(bool)));

    int h_step = 0;
    int *d_step = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_step, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_step, 0, sizeof(int)));

    int h_step_inner = 0;
    int *d_step_inner = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_step_inner, sizeof(int)));

    PixMapping *d_pair_pixpoly = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pair_pixpoly, 16 * 1024 * 1024 * sizeof(PixMapping)));
    uint *d_pixpoly_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixpoly_size, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_pixpoly_size, 0, sizeof(uint)));
    uint h_pixpoly_size;

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

    kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_info, size, d_distance);
    cudaDeviceSynchronize();
    check_execution("kernel init");

    timer.stopTimer();
    printf("kernel initialization time: %f ms\n", timer.getElapsedTime());

    printf("SIZE = %u\n", size);

    while (true)
    {
        // h_step = 2;
        CUDA_SAFE_CALL(cudaMemcpy(d_step, &h_step, sizeof(int), cudaMemcpyHostToDevice));

        // printf("STEP: %d\n", h_step);
        CUDA_SAFE_CALL(cudaMemset(d_pixpoly_size, 0, sizeof(uint)));

        // timer.startTimer();

        kernel_1<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, d_pair_pixpoly, d_pixpoly_size, size, d_step, d_resultmap);
        cudaDeviceSynchronize();
        check_execution("Kernel filter");

        // timer.stopTimer();
        // printf("kernel_1 time: %f ms\n", timer.getElapsedTime());
        // sum_filter += timer.getElapsedTime();

        CUDA_SAFE_CALL(cudaMemcpy(&h_pixpoly_size, d_pixpoly_size, sizeof(uint), cudaMemcpyDeviceToHost));

        if (h_pixpoly_size == 0)
            break;
        // printf("h_pixpoly_size: %u\n", h_pixpoly_size);
        h_step_inner = 0;
        CUDA_SAFE_CALL(cudaMemcpy(d_step_inner, &h_step_inner, sizeof(int), cudaMemcpyHostToDevice));
        while (true)
        {
            CUDA_SAFE_CALL(cudaMemset(d_bufferinput_size, 0, sizeof(uint)));
            CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

            grid_size.x = (h_pixpoly_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            kernel_2_1<<<grid_size, block_size>>>(d_pair_pixpoly, d_pairs, gctx->d_info, gctx->d_status, d_step_inner, d_pixpoly_size, (PixPair *)d_BufferInput, d_bufferinput_size, d_resultmap);
            cudaDeviceSynchronize();
            check_execution("Kernel_2_1");

            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            printf("step = %d, step_inner = %d, h_bufferinput_size: %u\n", h_step, h_step_inner, h_bufferinput_size);
            if (h_bufferinput_size == 0)
                break;

            //         grid_size.x = (h_bufferinput_size + BLOCK_SIZE - 1) /
            //         BLOCK_SIZE;

            //         kernel_2_2<<<grid_size,
            //         block_size>>>((PixPair*)d_BufferInput, d_pairs,
            //         gctx->d_offset, gctx->d_edge_sequences,
            //         d_bufferinput_size, (Batch *)d_BufferOutput,
            //         d_bufferoutput_size, d_resultmap);
            //         cudaDeviceSynchronize();
            // 	    check_execution("Kernel_2_2");

            //         swap(d_BufferInput, d_BufferOutput);
            //         swap(d_bufferinput_size, d_bufferoutput_size);
            //         CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0,
            //         sizeof(uint)));

            //         grid_size.x = (h_bufferinput_size + BLOCK_SIZE - 1) /
            //         BLOCK_SIZE;

            //         kernel_2_3<<<grid_size, block_size>>>((Batch
            //         *)d_BufferInput, gctx->d_vertices, d_bufferinput_size,
            //         d_distance, d_resultmap); cudaDeviceSynchronize();
            // 	    check_execution("Kernel_2_3");
            if (h_step_inner == 2)
                break;
            h_step_inner++;
            CUDA_SAFE_CALL(cudaMemcpy(d_step_inner, &h_step_inner, sizeof(int), cudaMemcpyHostToDevice));
        }

        //     grid_size.x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        //     kernel_3<<<grid_size, block_size>>>(d_pairs, gctx->d_info, d_step, size, d_distance, d_resultmap);
        //     cudaDeviceSynchronize();
        //     check_execution("Kernel_3");

        if (h_step == 2)
            break;
        h_step++;
        CUDA_SAFE_CALL(cudaMemcpy(d_step, &h_step, sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
    int found = 0;
    for (int i = 0; i < size; i++)
    {
        if (h_distance[i] <= WITHIN_DISTANCE)
            found++;
    }

    return found;
}