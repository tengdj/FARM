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
    int level;
    double minDist;
    double maxDist;
};

__device__ int d_level;

__global__ void kernel_init(pair<uint32_t, uint32_t> *pairs, uint source_size, BoxDistRange *buffer, double *max_box_dist, uint size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        // target id follows source id
        pairs[pair_id].second += source_size;
        buffer[pair_id] = {0, 0, pair_id, 0, 0.0, DBL_MAX};
        max_box_dist[pair_id] = DBL_MAX;
    }
}

__global__ void cal_box_distance(BoxDistRange *candidate, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, double *max_box_dist, uint *size, BoxDistRange *buffer, uint *buffer_size, uint *result)
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
        uint s_level = idealoffset[pair.first + 1].layer_start - source.layer_start;
        uint t_level = idealoffset[pair.second + 1].layer_start - target.layer_start;

        // printf("%d %d\n", s_level, t_level);

        if (d_level > s_level && d_level > t_level)
        {
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = candidate[candidate_id];
            return;
        }

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
            if((source_pixel_id / ((layer_info + source.layer_start)[d_level - 1].dimx + 1)) > (layer_info + source.layer_start)[d_level - 1].dimy){
                printf("SOURCE %d %d %d %d\n", source_pixel_id, target_pixel_id, candidate[candidate_id].level, d_level - 1);
            }
            source_pixel_box = gpu_get_pixel_box(
                gpu_get_x(source_pixel_id, (layer_info + source.layer_start)[d_level - 1].dimx),
                gpu_get_y(source_pixel_id, (layer_info + source.layer_start)[d_level - 1].dimx, (layer_info + source.layer_start)[d_level - 1].dimy),
                (layer_info + source.layer_start)[d_level - 1].mbr.low[0], (layer_info + source.layer_start)[d_level - 1].mbr.low[1],
                (layer_info + source.layer_start)[d_level - 1].step_x, (layer_info + source.layer_start)[d_level - 1].step_y);
            source_pixel_box.low[0] += 0.00001;
            source_pixel_box.low[1] += 0.00001;
            source_pixel_box.high[0] -= 0.00001;
            source_pixel_box.high[1] -= 0.00001;

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
            if((target_pixel_id / ((layer_info + target.layer_start)[d_level - 1].dimx + 1)) > (layer_info + target.layer_start)[d_level - 1].dimy){
                printf("TARGET %d %d %d %d\n", source_pixel_id, target_pixel_id, candidate[candidate_id].level, d_level);
            }
            target_pixel_box = gpu_get_pixel_box(
                gpu_get_x(target_pixel_id, (layer_info + target.layer_start)[d_level - 1].dimx),
                gpu_get_y(target_pixel_id, (layer_info + target.layer_start)[d_level - 1].dimx, (layer_info + target.layer_start)[d_level - 1].dimy),
                (layer_info + target.layer_start)[d_level - 1].mbr.low[0], (layer_info + target.layer_start)[d_level - 1].mbr.low[1],
                (layer_info + target.layer_start)[d_level - 1].step_x, (layer_info + target.layer_start)[d_level - 1].step_y);
            target_pixel_box.low[0] += 0.00001;
            target_pixel_box.low[1] += 0.00001;
            target_pixel_box.high[0] -= 0.00001;
            target_pixel_box.high[1] -= 0.00001;

            target_start_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.low[0], t_step_x, t_dimx);
            target_start_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.low[1], t_step_y, t_dimy);
            target_end_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.high[0], t_step_x, t_dimx);
            target_end_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.high[1], t_step_y, t_dimy);
        }

        // printf("%d %d %d %d\n", source_start_x, source_start_y, source_end_x, source_end_y);

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
                            double min_distance = gpu_distance(box1, box2);
                            double max_distance = gpu_max_distance(box1, box2);
                            if(max_distance <= WITHIN_DISTANCE){
                                atomicAdd(result, 1);
                                atomicExchDouble(max_box_dist + pair_id, 0.0); 
                                return;
                            }
                            if(min_distance > WITHIN_DISTANCE) continue;
            
                            int idx = atomicAdd(buffer_size, 1);
                            buffer[idx] = {id1, id2, pair_id, d_level, min_distance, max_distance};
                            atomicMinDouble(max_box_dist + pair_id, max_distance);
                        }
                    }
                }
            }
        }
    }
}

__global__ void kernel_filter(BoxDistRange *bufferinput, double *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        double left = bufferinput[bufferId].minDist;
        int pairId = bufferinput[bufferId].pairId;

        if (left < max_box_dist[pairId])
        {
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = bufferinput[bufferId];
        }
    }
}

// __global__ void statistic_size(BoxDistRange *pixpairs, uint *size, uint *pixelpairsize){
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         int pairId = pixpairs[bufferId].pairId;
//         atomicAdd(pixelpairsize + pairId, 1);
//     }
// }

// __global__ void group_by_id(BoxDistRange *bufferinput, uint *size, BoxDistRange *bufferoutput, int *pixelpairidx){
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         int pairId = bufferinput[bufferId].pairId;
//         int pos = atomicAdd(&pixelpairidx[pairId], 1);
//         bufferoutput[pos] = bufferinput[bufferId];
//     }
// }

// __global__ void kernel_sort(BoxDistRange *pixpairs, uint size, uint *pixelpairsize, int *pixelpairidx){
//     const int pairId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (pairId < size)
//     {
//         int start = pixelpairidx[pairId];
//         int size = pixelpairsize[pairId];

//         for(int i = start; i < start + size - 1; i ++){
//             int minIndex = i;
//             for(int j = i + 1; j < start + size; j ++){
//                 if(pixpairs[j].minDist < pixpairs[minIndex].minDist){
//                     minIndex = j;
//                 }
//             }

//             if(minIndex != i){
//                 BoxDistRange temp = pixpairs[i];
//                 pixpairs[i] = pixpairs[minIndex];
//                 pixpairs[minIndex] = temp;
//             }
//         }
//     }
// }

// struct CompareByMinDist {
//     __device__
//     bool operator()(const BoxDistRange& a, const BoxDistRange& b) {
//         return a.minDist < b.minDist;  // 按照 minDist 升序排序
//     }
// };

// void SortGroups(BoxDistRange *pixpairs, uint size, uint *pixelpairsize, int *pixelpairidx) {
//     // 创建 CUDA 流
//     cudaStream_t *streams = new cudaStream_t[size];
//     for (uint i = 0; i < size; i++) {
//         cudaError_t err = cudaStreamCreate(&streams[i]);
//         if (err != cudaSuccess) {
//             std::cerr << "Error creating stream: " << cudaGetErrorString(err) << std::endl;
//             // 处理错误
//         }
//     }

//     for (uint i = 0; i < size; i++) {
//         int startIdx = pixelpairidx[i];
//         int endIdx = startIdx + pixelpairsize[i];

//         thrust::device_ptr<BoxDistRange> begin = thrust::device_pointer_cast(pixpairs + startIdx);
//         thrust::device_ptr<BoxDistRange> end = thrust::device_pointer_cast(pixpairs + endIdx);

//         // 在特定的流中执行排序
//         thrust::sort(thrust::cuda::par.on(streams[i]), begin, end, CompareByMinDist());
//     }

//     // 等待所有流完成
//     for (uint i = 0; i < size; i++) {
//         cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     delete[] streams;
// }

// __global__ void kernel_merge(int *pixelpairidx, uint *pixelpairsize, BoxDistRange *pixpairs, uint pairsize, int *mergeSize, bool *resultmap)
// {
//     const int pairId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (pairId < pairsize)
//     {
//         int start = pixelpairidx[pairId];
//         int size = pixelpairsize[pairId];
//         int length = size;

//         // printf("start = %d, size = %d, length = %d\n", start, size, length);

//         if(resultmap[pairId] || size <= 0) return;

//         double left = 0x3f3f3f3f, right = 0x3f3f3f3f;
//         for(int i = start; i < start + size; i ++){
//             double mind = pixpairs[i].minDist, maxd = pixpairs[i].maxDist;
//             left = min(mind, left);
//             right = min(maxd, right);
//             double ratio = (right - mind) / (right - left);
//             // printf("ratio = %d\n", ratio);
//             if(ratio < 0.9){
//                 // printf("mind = %lf, maxd = %lfd, left = %lf, right = %lf\n", mind, maxd, left, right);
//                 length = i - start + 1;
//                 break;
//             }
//         }

//         // printf("%d in kernel %d\n", length, pairId);

//         mergeSize[pairId] = length;

//     }
// }

// __global__ void kernel_unroll(int *pixelpairidx, uint *pixelpairsize, int *mergeSize, BoxDistRange *pixpairs, Batch *pairs, uint32_t *offset, EdgeSeq *edge_sequences, Point *vertices, int pairId, int *loop, Task *tasks, uint *batch_size, bool *resultmap, double *distance)
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < mergeSize[pairId])
//     {
//         if(resultmap[pairId] || pixelpairsize[pairId] <= 0) return;
//         int bufferId = pixelpairidx[pairId] + idx;
//         int p = pixpairs[bufferId].sourcePixelId;
//         int p2 = pixpairs[bufferId].targetPixelId;
//         // printf("%d %d %d %lf %lf\n", pairId, p, p2, pixpairs[bufferId].minDist, pixpairs[bufferId].maxDist);
//         if(*loop && pixpairs[bufferId].minDist > distance[pairId]){
//             resultmap[pairId] = true;
//             return;
//         }

//         IdealOffset &source = pairs[pairId].source;
//         IdealOffset &target = pairs[pairId].target;

//         int s_num_sequence = (offset + source.offset_start)[p + 1] - (offset + source.offset_start)[p];
//         int t_num_sequence = (offset + target.offset_start)[p2 + 1] - (offset + target.offset_start)[p2];

//         for (int i = 0; i < s_num_sequence; ++ i)
//         {
//             EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(offset + source.offset_start)[p] + i];
//             for (int j = 0; j < t_num_sequence; ++j)
//             {
//                 EdgeSeq r2 = (edge_sequences + target.edge_sequences_start)[(offset + target.offset_start)[p2] + j];
//                 // if (r.length < 2 || r2.length < 2) continue;
//                 // if(gpu_point_to_point_distance(vertices[source.vertices_start + r.start], vertices[target.vertices_start + r2.start]) <= WITHIN_DISTANCE ||
//                 //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start], vertices[target.vertices_start + r2.start + r2.length - 1]) <= WITHIN_DISTANCE ||
//                 //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start + r.length - 1], vertices[target.vertices_start + r2.start]) <= WITHIN_DISTANCE ||
//                 //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start + r.length - 1], vertices[target.vertices_start + r2.start + r2.length - 1]) <= WITHIN_DISTANCE)
//                 // {
//                 //     resultmap[pairId] = true;
//                 //     return;
//                 // }
//                 int max_size = 2;
//                 for (uint s = 0; s < r.length; s += max_size)
//                 {
//                     uint end_s = min(s + max_size, r.length);
//                     for (uint t = 0; t < r2.length; t += max_size)
//                     {
//                         uint end_t = min(t + max_size, r2.length);
//                         uint idx = atomicAdd(batch_size, 1U);
//                         tasks[idx].s_start = source.vertices_start + r.start + s;
//                         tasks[idx].t_start = target.vertices_start + r2.start + t;
//                         tasks[idx].s_length = end_s - s;
//                         tasks[idx].t_length = end_t - t;
//                         tasks[idx].pair_id = pairId;
//                     }
//                 }
//             }
//         }
//     }
// }

// __global__ void kernel_refine(Task *batches, Point *vertices, uint *size, double *distance, bool *resultmap, double *max_box_dist, double *degree_per_kilometer_latitude, double *degree_per_kilometer_longitude_arr)
// {
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         uint s1 = batches[bufferId].s_start;
//         uint s2 = batches[bufferId].t_start;
//         uint len1 = batches[bufferId].s_length;
//         uint len2 = batches[bufferId].t_length;
//         int pair_id = batches[bufferId].pair_id;
//         if(resultmap[pair_id]) return;

//         double dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2, len1, len2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);

//         // double dist = gpu_point_to_point_distance(vertices + s1, vertices + s2);
//         // if(dist <= WITHIN_DISTANCE) resultmap[pair_id] = true;
//         atomicMinDouble(max_box_dist + pair_id, dist);
//         atomicMinDouble(distance + pair_id, dist);
//     }
// }

uint cuda_within_polygon(query_context *gctx)
{
    uint h_bufferinput_size, h_bufferoutput_size;

#ifdef DEBUG
    CudaTimer timer;
    timer.startTimer();
#endif
    double *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, gctx->num_pairs * sizeof(double)));

    uint8_t *d_layers = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_layers, gctx->num_pairs * sizeof(uint8_t)));

    // uint *d_level = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_level, sizeof(uint)));
    // CUDA_SAFE_CALL(cudaMemset(d_level, 0, sizeof(uint)));

    // int *d_loop = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_loop, sizeof(int)));

    // uint *h_pixelpairsize = new uint[gctx->num_pairs];
    // uint *d_pixelpairsize = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairsize, gctx->num_pairs * sizeof(uint)));

    // int *h_pixelpairidx = new int[gctx->num_pairs];
    // int *d_pixelpairidx = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairidx, gctx->num_pairs * sizeof(int)));

    // int *h_mergesize = new int[gctx->num_pairs];
    // int *d_mergesize = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_mergesize, gctx->num_pairs * sizeof(int)));

    // BoxDistRange *d_sortedpairs = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_sortedpairs, 2UL * 1024 * 1024 * 1024));
    // uint *d_sortedpairs_size = nullptr;
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_sortedpairs_size, sizeof(uint)));

    // uint h_loop = 0;
    // CUDA_SAFE_CALL(cudaMemset(d_loop, 0, sizeof(int)));

    // CUDA_SAFE_CALL(cudaMemset(d_pixelpairsize, 0, gctx->num_pairs * sizeof(uint)));
    // memset(h_pixelpairsize, 0, gctx->num_pairs * sizeof(uint));

    // CUDA_SAFE_CALL(cudaMemset(d_pixelpairidx, 0, gctx->num_pairs * sizeof(int)));
    // memset(h_pixelpairidx, 0, gctx->num_pairs * sizeof(int));

    // CUDA_SAFE_CALL(cudaMemset(d_mergesize, 0, gctx->num_pairs * sizeof(int)));
    // memset(h_mergesize, 0, gctx->num_pairs * sizeof(int));

    int grid_size_x = (gctx->num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, 1, 1);
    dim3 grid_size(grid_size_x, 1, 1);

    kernel_init<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->source_ideals.size(), (BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, gctx->num_pairs);
    cudaDeviceSynchronize();
    check_execution("kernel_init");

    // pair<uint32_t, uint32_t> *h_test_pair = new pair<uint32_t, uint32_t>[gctx->num_pairs];
    // cudaMemcpy(h_test_pair, gctx->d_candidate_pairs, gctx->num_pairs * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < gctx->num_pairs; i ++){
    //     cout << h_test_pair[i].first << " " << h_test_pair[i].second << endl;
    // }

    int h_level = 0;

    h_bufferinput_size = gctx->num_pairs;
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, sizeof(uint), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_bufferinput_size = %u\n", h_bufferinput_size);

    while(true){
        h_level ++;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_level, &h_level, sizeof(int)));

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_result);
        cudaDeviceSynchronize();
        check_execution("cal_box_distance");

// #ifdef DEBUG
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);
// #endif
        if(h_bufferinput_size == h_bufferoutput_size) {
            printf("%d %d\n", h_bufferinput_size, h_bufferoutput_size);
            break;
        }

        CUDA_SWAP_BUFFER();

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter");
// #ifdef DEBUG
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("filter h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("filter h_bufferoutput_size = %u\n", h_bufferoutput_size);
// #endif

        CUDA_SWAP_BUFFER();

    }

    // swap(d_BufferInput, d_BufferOutput);
    // swap(d_bufferinput_size, d_bufferoutput_size);
    // swap(h_bufferinput_size, h_bufferoutput_size);
    // CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

    /* TO DELETE */
    // 输出第一对polygons的具体信息
    // BoxDistRange *h_BufferInput = new BoxDistRange[h_bufferinput_size];
    // CUDA_SAFE_CALL(cudaMemcpy(h_BufferInput, d_BufferInput, h_bufferinput_size * sizeof(BoxDistRange), cudaMemcpyDeviceToHost));
    // gctx->polygon_pairs[239].first->MyPolygon::print();
    // gctx->polygon_pairs[239].first->MyRaster::print();
    // gctx->polygon_pairs[239].second->MyPolygon::print();
    // gctx->polygon_pairs[239].second->MyRaster::print();

    // for(int i = 0; i < h_bufferinput_size; i ++){
    //     if(h_BufferInput[i].pairId == 1){
    //         printf("%d\t%d\t%d\t%d\t%lf\t%lf\n", gctx->polygon_pairs[239].first->get_x(h_BufferInput[i].sourcePixelId), gctx->polygon_pairs[239].first->get_y(h_BufferInput[i].sourcePixelId),
    //         gctx->polygon_pairs[239].second->get_x(h_BufferInput[i].targetPixelId), gctx->polygon_pairs[239].second->get_y(h_BufferInput[i].targetPixelId), h_BufferInput[i].minDist, h_BufferInput[i].maxDist);
    //     }
    // }
    // return 0;
    /* TO DELETE */

    // grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // block_size.x = BLOCK_SIZE;
    // grid_size.x = grid_size_x;

    // timer.startTimer();
    // statistic_size<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_bufferinput_size, d_pixelpairsize);
    // cudaDeviceSynchronize();
    // check_execution("statistic_size");
    // timer.stopTimer();
    // printf("statistic_size: %f ms\n", timer.getElapsedTime());

    // cudaMemcpy(h_pixelpairsize, d_pixelpairsize, size * sizeof(uint), cudaMemcpyDeviceToHost);

    // for(int i = 1; i < size; i ++){
    //     h_pixelpairidx[i] = h_pixelpairidx[i - 1] + h_pixelpairsize[i - 1];

    // }

    // cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice);

    // timer.startTimer();
    // group_by_id<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_pixelpairidx);
    // cudaDeviceSynchronize();
    // check_execution("group_by_id");
    // timer.stopTimer();
    // printf("group_by_id: %f ms\n", timer.getElapsedTime());

    // h_bufferoutput_size = h_pixelpairidx[size - 1] + h_pixelpairsize[size - 1];

    // cudaMemcpy(d_bufferoutput_size, &h_bufferoutput_size, sizeof(uint), cudaMemcpyHostToDevice);

    // swap(d_BufferInput, d_BufferOutput);
    // swap(d_bufferinput_size, d_bufferoutput_size);
    // swap(h_bufferinput_size, h_bufferoutput_size);
    // CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

    // cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice);

    // timer.startTimer();
    // SortGroups((BoxDistRange *)d_BufferInput, size, h_pixelpairsize, h_pixelpairidx);
    // timer.stopTimer();
    // printf("kernel_sort: %f ms\n", timer.getElapsedTime());

    // CUDA_SAFE_CALL(cudaMemcpy(d_sortedpairs, d_BufferInput, h_bufferinput_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(d_sortedpairs_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToDevice));

    // uint h_sortpairs_size = h_bufferinput_size;

    // h_loop = 0;
    // CUDA_SAFE_CALL(cudaMemset(d_loop, 0, sizeof(int)));

    // while(true){

    //     grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //     block_size.x = BLOCK_SIZE;
    //     grid_size.x = grid_size_x;

    //     timer.startTimer();
    //     kernel_merge<<<grid_size, block_size>>>(d_pixelpairidx, d_pixelpairsize, (BoxDistRange *)d_BufferInput, size, d_mergesize, d_resultmap);
    //     cudaDeviceSynchronize();
    //     check_execution("kernel_merge");
    //     timer.stopTimer();
    //     printf("kernel merge: %f ms\n", timer.getElapsedTime());

    //     CUDA_SAFE_CALL(cudaMemcpy(h_pixelpairidx, d_pixelpairidx, size * sizeof(int), cudaMemcpyDeviceToHost));
    //     CUDA_SAFE_CALL(cudaMemcpy(h_pixelpairsize, d_pixelpairsize, size * sizeof(int), cudaMemcpyDeviceToHost));
    //     CUDA_SAFE_CALL(cudaMemcpy(h_mergesize, d_mergesize, size * sizeof(int), cudaMemcpyDeviceToHost));
    //     CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(bool), cudaMemcpyDeviceToHost));
    //     // /*  To Delete  */
    //     // printf("come on%d\n", size);
    //     // for(int i = 0; i < size;i ++) {
    //     //     if(!h_resultmap[i]){
    //     //         printf("%d\n", i);
    //     //     }
    //     // }
    //     // /*  To Delete  */

    //     timer.startTimer();
    //     cudaStream_t *streams = new cudaStream_t[size];
    //     for (int i = 0; i < size; i++) {
    //         cudaStreamCreate(&streams[i]);
    //     }

    //     for(int i = 0; i < size; i ++){
    //         grid_size_x = (h_mergesize[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //         block_size.x = BLOCK_SIZE;
    //         grid_size.x = grid_size_x;
    //         kernel_unroll<<<grid_size, block_size, 0, streams[i]>>>(d_pixelpairidx, d_pixelpairsize, d_mergesize, (BoxDistRange *)d_BufferInput, d_pairs, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, i, d_loop, (Task *)d_BufferOutput, d_bufferoutput_size, d_resultmap, d_distance);
    //         if(h_pixelpairsize[i] > 0){
    //             h_pixelpairidx[i] += h_mergesize[i];
    //             h_pixelpairsize[i] -= h_mergesize[i];
    //         }

    //     }
    //     check_execution("kernel_unroll");
    //     cudaDeviceSynchronize();
    //     timer.stopTimer();
    //     printf("kernel unroll: %f ms\n", timer.getElapsedTime());

    //     for (int i = 0; i < size; i++) {
    //         cudaStreamSynchronize(streams[i]);
    //     }

    //     for (int i = 0; i < size; i++) {
    //     cudaStreamDestroy(streams[i]);
    //     }

    //     delete[] streams;

    //     CUDA_SAFE_CALL(cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice));
    //     CUDA_SAFE_CALL(cudaMemcpy(d_pixelpairsize, h_pixelpairsize, size * sizeof(int), cudaMemcpyHostToDevice));
    //     CUDA_SAFE_CALL(cudaMemcpy(d_mergesize, h_mergesize, size * sizeof(int), cudaMemcpyHostToDevice));

    //     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    //     if(h_bufferoutput_size == 0) break;

    //     h_loop ++;
    //     CUDA_SAFE_CALL(cudaMemcpy(d_loop, &h_loop, sizeof(int), cudaMemcpyHostToDevice));

    //     /* To delete  */
    //     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

    //     printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
    //     /* To delete  */

    //     swap(d_BufferInput, d_BufferOutput);
    //     swap(d_bufferinput_size, d_bufferoutput_size);
    //     swap(h_bufferinput_size, h_bufferoutput_size);
    //     CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

    //     grid_size_x = (h_bufferinput_size + 512 - 1) / 512;
    //     block_size.x = 512;
    //     grid_size.x = grid_size_x;

    //     timer.startTimer();
    //     kernel_refine<<<grid_size, block_size>>>((Task *)d_BufferInput, gctx->d_vertices, d_bufferinput_size, d_distance, d_resultmap, d_max_box_dist, gctx->d_degree_degree_per_kilometer_latitude, gctx->degree_per_kilometer_longitude_arr);
    //     cudaDeviceSynchronize();
    //     check_execution("kernel_refine");
    //     timer.stopTimer();
    //     printf("kernel refine: %f ms\n", timer.getElapsedTime());

    //     CUDA_SAFE_CALL(cudaMemcpy(d_bufferinput_size, d_sortedpairs_size, sizeof(uint), cudaMemcpyDeviceToDevice));
    //     CUDA_SAFE_CALL(cudaMemcpy(d_BufferInput, d_sortedpairs, h_sortpairs_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));
    // }
    // duration.stopTimer();
    // printf("batch time = %lf ms\n", duration.getElapsedTime());

    // CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(bool), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < size; i++)
    // {
    //     if (h_distance[i] <= WITHIN_DISTANCE)
    //         found++;
    //     // printf("%lf\n", h_distance[i]);
    // }

    // total.startTimer();
    // printf("GPU total time = %lf ms\n", total.getElapsedTime());
    // return found;

    return 0;
}