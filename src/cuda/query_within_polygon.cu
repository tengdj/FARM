#include "geometry.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define WITHIN_DISTANCE 10

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
    double minDist;
    double maxDist;
    int pairId;
    int level = 0;
};

__global__ void kernel_init(Batch *d_pairs, RasterInfo *d_info, uint size, double *distance, double *max_box_dist)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        Batch &pair = d_pairs[pair_id];
        IdealOffset &source = pair.source;
        IdealOffset &target = pair.target;
        box &s_mbr = d_info[source.info_start].mbr;
        box &t_mbr = d_info[target.info_start].mbr;

        distance[pair_id] = gpu_max_distance(s_mbr, t_mbr);
        max_box_dist[pair_id] = DBL_MAX;
    }
}

__global__ void first_cal_box_distance(Batch *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *max_box_dist, uint *global_level, uint size, BoxDistRange *buffer, uint *buffer_size, bool *resultmap)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        Batch &pair = pairs[pair_id];
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
                if (gpu_show_status(status, source.status_start, i, source_offset) == BORDER && gpu_show_status(status, target.status_start, j, target_offset) == BORDER)
                {
                    auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                    auto target_box = gpu_get_pixel_box(gpu_get_x(j, t_dimx), gpu_get_y(j, t_dimx, t_dimy), t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                    
                    
                    double min_distance = gpu_distance(source_box, target_box);
                    double max_distance = gpu_max_distance(source_box, target_box);
                    // if(max_distance <= WITHIN_DISTANCE) resultmap[pair_id] = true;
                    int idx = atomicAdd(buffer_size, 1);
                    buffer[idx] = {i, j, min_distance, max_distance, pair_id};
                    atomicMinDouble(max_box_dist + pair_id, max_distance);
                }
            }
        }
    }
}

__global__ void cal_box_distance(BoxDistRange *candidate, Batch *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *max_box_dist, uint *global_level, uint *size, BoxDistRange *buffer, uint *buffer_size, bool *resultmap)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate_id < *size)
    {
        int source_pixel_id = candidate[candidate_id].sourcePixelId;
        int target_pixel_id = candidate[candidate_id].targetPixelId;
        int pair_id = candidate[candidate_id].pairId;
        
        if(resultmap[pair_id]) return;

        Batch &pair = pairs[pair_id];
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
            source_pixel_box.low[0] += 0.00001;
            source_pixel_box.low[1] += 0.00001;
            source_pixel_box.high[0] -= 0.00001;
            source_pixel_box.high[1] -= 0.00001;

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
            target_pixel_box.low[0] += 0.00001;
            target_pixel_box.low[1] += 0.00001;
            target_pixel_box.high[0] -= 0.00001;
            target_pixel_box.high[1] -= 0.00001;

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
                        if (gpu_show_status(status, source.status_start, id1, source_offset) == BORDER && gpu_show_status(status, target.status_start, id2, target_offset) == BORDER){
                            // printf("block_id = %u thread_id = %u pairid = %d level = %d id1 = %d id2 = %d\n", blockIdx.x, threadIdx.x, pair_id, *global_level, id1, id2);
                            auto box1 = gpu_get_pixel_box(x1, y1, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                            auto box2 = gpu_get_pixel_box(x2, y2, t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                            double min_distance = gpu_distance(box1, box2);
                            double max_distance = gpu_max_distance(box1, box2);
                            // if(max_distance <= WITHIN_DISTANCE) resultmap[pair_id] = true;
                            int idx = atomicAdd(buffer_size, 1);
                            buffer[idx] = {id1, id2, min_distance, max_distance, pair_id};
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
            // printf("id1 = %d id2 = %d\n", bufferinput[bufferId].sourcePixelId, bufferinput[bufferId].targetPixelId);
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = bufferinput[bufferId];
        }
    }
}

__global__ void statistic_size(BoxDistRange *pixpairs, uint *size, uint *pixelpairsize){
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int pairId = pixpairs[bufferId].pairId;
        atomicAdd(pixelpairsize + pairId, 1);
    }
}

__global__ void group_by_id(BoxDistRange *bufferinput, uint *size, BoxDistRange *bufferoutput, int *pixelpairidx){
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int pairId = bufferinput[bufferId].pairId;
        int pos = atomicAdd(&pixelpairidx[pairId], 1);
        bufferoutput[pos] = bufferinput[bufferId];
    }
}

__global__ void kernel_sort(BoxDistRange *pixpairs, uint size, uint *pixelpairsize, int *pixelpairidx){
    const int pairId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairId < size)
    {
        int start = pixelpairidx[pairId];
        int size = pixelpairsize[pairId];

        for(int i = start; i < start + size - 1; i ++){
            int minIndex = i;
            for(int j = i + 1; j < start + size; j ++){
                if(pixpairs[j].minDist < pixpairs[minIndex].minDist){
                    minIndex = j;
                }
            }

            if(minIndex != i){
                BoxDistRange temp = pixpairs[i];
                pixpairs[i] = pixpairs[minIndex];
                pixpairs[minIndex] = temp;
            }
        }
    }   
}

struct CompareByMinDist {
    __host__ __device__
    bool operator()(const BoxDistRange& a, const BoxDistRange& b) {
        return a.minDist < b.minDist;  // 按照 minDist 升序排序
    }
};

void SortGroups(BoxDistRange *pixpairs, uint size, uint *pixelpairsize, int *pixelpairidx){
    for (int i = 0; i < size; i++) {
        int startIdx = pixelpairidx[i];
        int endIdx = startIdx + pixelpairsize[i];

        thrust::device_ptr<BoxDistRange> begin = thrust::device_pointer_cast(pixpairs + startIdx);
        thrust::device_ptr<BoxDistRange> end = thrust::device_pointer_cast(pixpairs + endIdx);

        thrust::sort(begin, end, CompareByMinDist());      

    }
}

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

__global__ void kernel_merge(int *pixelpairidx, uint *pixelpairsize, BoxDistRange *pixpairs, uint pairsize, int *mergeSize, bool *resultmap)
{
    const int pairId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairId < pairsize)
    {
        int start = pixelpairidx[pairId];
        int size = pixelpairsize[pairId];
        int length = size;

        // printf("start = %d, size = %d, length = %d\n", start, size, length);

        if(resultmap[pairId] || size <= 0) return;

        double left = 0x3f3f3f3f, right = 0x3f3f3f3f;
        for(int i = start; i < start + size; i ++){
            double mind = pixpairs[i].minDist, maxd = pixpairs[i].maxDist;
            left = min(mind, left);
            right = min(maxd, right);
            double ratio = (right - mind) / (right - left);
            // printf("ratio = %d\n", ratio);
            if(ratio < 0.9){
                // printf("mind = %lf, maxd = %lfd, left = %lf, right = %lf\n", mind, maxd, left, right);
                length = i - start + 1;
                break;
            }
        }

        // printf("%d in kernel %d\n", length, pairId);

        mergeSize[pairId] = length;   
        
    }
}

__global__ void kernel_unroll(int *pixelpairidx, uint *pixelpairsize, int *mergeSize, BoxDistRange *pixpairs, Batch *pairs, uint16_t *offset, EdgeSeq *edge_sequences, Point *vertices, int pairId, int *loop, Task *tasks, uint *batch_size, bool *resultmap, double *distance)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mergeSize[pairId])
    {
        if(resultmap[pairId] || pixelpairsize[pairId] <= 0) return;
        int bufferId = pixelpairidx[pairId] + idx;
        int p = pixpairs[bufferId].sourcePixelId;
        int p2 = pixpairs[bufferId].targetPixelId;
        // printf("%d %d %d %lf %lf\n", pairId, p, p2, pixpairs[bufferId].minDist, pixpairs[bufferId].maxDist);
        if(*loop && pixpairs[bufferId].minDist > distance[pairId]){
            resultmap[pairId] = true;
            return;
        }
        
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
                // if (r.length < 2 || r2.length < 2) continue;
                // if(gpu_point_to_point_distance(vertices[source.vertices_start + r.start], vertices[target.vertices_start + r2.start]) <= WITHIN_DISTANCE ||
                //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start], vertices[target.vertices_start + r2.start + r2.length - 1]) <= WITHIN_DISTANCE ||
                //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start + r.length - 1], vertices[target.vertices_start + r2.start]) <= WITHIN_DISTANCE ||
                //    gpu_point_to_point_distance(vertices[source.vertices_start + r.start + r.length - 1], vertices[target.vertices_start + r2.start + r2.length - 1]) <= WITHIN_DISTANCE)
                // {
                //     resultmap[pairId] = true;
                //     return;
                // }    
                int max_size = 16;
                for (uint s = 0; s < r.length; s += max_size)
                {
                    uint end_s = min(s + max_size, r.length);
                    for (uint t = 0; t < r2.length; t += max_size)
                    {
                        uint end_t = min(t + max_size, r2.length);
                        uint idx = atomicAdd(batch_size, 1U);
                        tasks[idx].s_start = source.vertices_start + r.start + s;
                        tasks[idx].t_start = target.vertices_start + r2.start + t;
                        tasks[idx].s_length = end_s - s;
                        tasks[idx].t_length = end_t - t;
                        tasks[idx].pair_id = pairId;
                    }
                }
            }
        }
    }
}

__global__ void kernel_refine(Task *batches, Point *vertices, uint *size, double *distance, bool *resultmap, double *max_box_dist)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        uint s1 = batches[bufferId].s_start;
        uint s2 = batches[bufferId].t_start;
        uint len1 = batches[bufferId].s_length;
        uint len2 = batches[bufferId].t_length;
        int pair_id = batches[bufferId].pair_id;
        if(resultmap[pair_id]) return;

        double dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2, len1, len2);
        // double dist = gpu_point_to_point_distance(vertices + s1, vertices + s2);
        // if(dist <= WITHIN_DISTANCE) resultmap[pair_id] = true;
        atomicMinDouble(max_box_dist + pair_id, dist);
        // atomicMinDouble(distance + pair_id, dist);
    }
}

uint cuda_within_polygon(query_context *gctx)
{
    CudaTimer timer, duration, total;

    duration.startTimer();

    uint polygon_pairs_size = gctx->polygon_pairs.size();
    uint batch_size = 100000;
    int found = 0;

    log("The number of polygon pairs = %u", polygon_pairs_size);

    Batch *h_pairs = new Batch[batch_size];
    Batch *d_pairs = nullptr;

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, batch_size * sizeof(Batch)));

    double *h_distance = new double[batch_size * sizeof(double)];
    double *d_distance = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_distance, batch_size * sizeof(double)));

    double *d_max_box_dist = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, batch_size * sizeof(double)));

    uint *d_level = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_level, sizeof(uint)));

    int *d_loop = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_loop, sizeof(int)));

    bool *h_resultmap = new bool[batch_size];
    bool *d_resultmap = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, batch_size * sizeof(bool)));

    uint *h_pixelpairsize = new uint[batch_size];
    uint *d_pixelpairsize = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairsize, batch_size * sizeof(uint)));

    int *h_pixelpairidx = new int[batch_size];
    int *d_pixelpairidx = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixelpairidx, batch_size * sizeof(int)));

    int *h_mergesize = new int[batch_size];
    int *d_mergesize = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_mergesize, batch_size * sizeof(int)));

    BoxDistRange *d_sortedpairs = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sortedpairs, 2UL * 1024 * 1024 * 1024));
    uint *d_sortedpairs_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sortedpairs_size, sizeof(uint)));

    char *d_BufferInput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferInput, 8UL * 1024 * 1024 * 1024));
    uint *d_bufferinput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferinput_size, sizeof(uint)));
    uint h_bufferinput_size;

    char *d_BufferOutput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferOutput, 8UL * 1024 * 1024 * 1024));
    uint *d_bufferoutput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferoutput_size, sizeof(uint)));
    uint h_bufferoutput_size;

    total.startTimer();

    for (int i = 0; i < polygon_pairs_size; i += batch_size){  
        duration.startTimer();
        int start = i, end = min(i + batch_size, polygon_pairs_size);
        int size = end - start;
        printf("size = %d\n", size);
        for (int j = start, idx = 0; j < end; j ++, idx ++)
        {
            Ideal *source = gctx->polygon_pairs[j].first;
            Ideal *target = gctx->polygon_pairs[j].second;
            // printf("id = %d %d %d %d %d\n", i, source->get_dimx(), source->get_dimy(), target->get_dimx(), target->get_dimy());
            // source->MyPolygon::print();
            // source->MyRaster::print();
            // target->MyPolygon::print();
            // target->MyRaster::print();
            h_pairs[idx] = {*source->idealoffset, *target->idealoffset, source->get_num_layers(), target->get_num_layers()};
        }

        uint h_level = 0;
        CUDA_SAFE_CALL(cudaMemset(d_level, 0, sizeof(uint)));

        uint h_loop = 0;
        CUDA_SAFE_CALL(cudaMemset(d_loop, 0, sizeof(int)));

        CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, batch_size * sizeof(bool)));
        memset(h_resultmap, 0, batch_size * sizeof(bool));

        CUDA_SAFE_CALL(cudaMemset(d_pixelpairsize, 0, batch_size * sizeof(uint)));
        memset(h_pixelpairsize, 0, batch_size * sizeof(uint));

        CUDA_SAFE_CALL(cudaMemset(d_pixelpairidx, 0, batch_size * sizeof(int)));
        memset(h_pixelpairidx, 0, batch_size * sizeof(int));

        CUDA_SAFE_CALL(cudaMemset(d_mergesize, 0, batch_size * sizeof(int)));
        memset(h_mergesize, 0, batch_size * sizeof(int));

        CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(Batch), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(d_bufferinput_size, 0, sizeof(uint)));
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_size(BLOCK_SIZE, 1, 1);
        dim3 grid_size(grid_size_x, 1, 1);

        timer.startTimer();
        kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_info, size, d_distance, d_max_box_dist);
        cudaDeviceSynchronize();
        check_execution("kernel init");
        timer.stopTimer();
        printf("kernel init: %f ms\n", timer.getElapsedTime());

        h_level = 0;
        CUDA_SAFE_CALL(cudaMemset(d_level, 0, sizeof(uint)));

        //TODO: 限制寄存器数量
        grid_size_x = (size + 512 - 1) / 512;
        block_size.x = 512;
        grid_size.x = grid_size_x;

        timer.startTimer();
        first_cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, d_level, size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size, d_resultmap);
        cudaDeviceSynchronize();
        check_execution("first_cal_box_distance");
        timer.stopTimer();
        printf("kernel first calculate box distance: %f ms\n", timer.getElapsedTime());

// #ifdef DEBUG
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
// #endif

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

            CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(bool), cudaMemcpyDeviceToHost));

            timer.startTimer();
            cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, d_level, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size, d_resultmap);
            cudaDeviceSynchronize();
            check_execution("cal_box_distance");
            timer.stopTimer();
            printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

// #ifdef DEBUG
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
            printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);
// #endif

            if(h_bufferinput_size == h_bufferoutput_size) break;

            swap(d_BufferInput, d_BufferOutput);
            swap(d_bufferinput_size, d_bufferoutput_size);
            swap(h_bufferinput_size, h_bufferoutput_size);
            CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

            grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            block_size.x = BLOCK_SIZE;
            grid_size.x = grid_size_x;

            timer.startTimer();
            kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_max_box_dist, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
            cudaDeviceSynchronize();
            check_execution("kernel_filter");
            timer.stopTimer();
            printf("kernel filter: %f ms\n", timer.getElapsedTime());

// #ifdef DEBUG
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            printf("filter h_bufferinput_size = %u\n", h_bufferinput_size);
            printf("filter h_bufferoutput_size = %u\n", h_bufferoutput_size);
// #endif
        }

        swap(d_BufferInput, d_BufferOutput);
        swap(d_bufferinput_size, d_bufferoutput_size);
        swap(h_bufferinput_size, h_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        timer.startTimer();
        statistic_size<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_bufferinput_size, d_pixelpairsize);
        cudaDeviceSynchronize();
        check_execution("statistic_size");
        timer.stopTimer();
        printf("statistic_size: %f ms\n", timer.getElapsedTime());

        cudaMemcpy(h_pixelpairsize, d_pixelpairsize, size * sizeof(uint), cudaMemcpyDeviceToHost);

        for(int i = 1; i < size; i ++){
            h_pixelpairidx[i] = h_pixelpairidx[i - 1] + h_pixelpairsize[i - 1];
            
        }

        cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice);

        timer.startTimer();
        group_by_id<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_pixelpairidx);
        cudaDeviceSynchronize();
        check_execution("group_by_id");
        timer.stopTimer();
        printf("group_by_id: %f ms\n", timer.getElapsedTime());

        h_bufferoutput_size = h_pixelpairidx[size - 1] + h_pixelpairsize[size - 1];
        
        cudaMemcpy(d_bufferoutput_size, &h_bufferoutput_size, sizeof(uint), cudaMemcpyHostToDevice);

        swap(d_BufferInput, d_BufferOutput);
        swap(d_bufferinput_size, d_bufferoutput_size);
        swap(h_bufferinput_size, h_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice);
        
        timer.startTimer();
        SortGroups((BoxDistRange *)d_BufferInput, size, h_pixelpairsize, h_pixelpairidx);
        timer.stopTimer();
        printf("kernel_sort: %f ms\n", timer.getElapsedTime());

        CUDA_SAFE_CALL(cudaMemcpy(d_sortedpairs, d_BufferInput, h_bufferinput_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_sortedpairs_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToDevice));

        uint h_sortpairs_size = h_bufferinput_size;

        h_loop = 0;
        CUDA_SAFE_CALL(cudaMemset(d_loop, 0, sizeof(int)));

        while(true){

            grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            block_size.x = BLOCK_SIZE;
            grid_size.x = grid_size_x;

            timer.startTimer();
            kernel_merge<<<grid_size, block_size>>>(d_pixelpairidx, d_pixelpairsize, (BoxDistRange *)d_BufferInput, size, d_mergesize, d_resultmap);
            cudaDeviceSynchronize();
            check_execution("kernel_merge");
            timer.stopTimer();
            printf("kernel merge: %f ms\n", timer.getElapsedTime()); 

            // /*  To Delete  */
            CUDA_SAFE_CALL(cudaMemcpy(h_pixelpairidx, d_pixelpairidx, size * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(h_pixelpairsize, d_pixelpairsize, size * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(h_mergesize, d_mergesize, size * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(bool), cudaMemcpyDeviceToHost));

            // for(int i = 0; i < size;i ++) {
            //     if(!h_resultmap[i]){
            //         printf("%d %d %d\n", h_pixelpairidx[i], h_pixelpairsize[i], h_mergesize[i]);
            //     }
            // }
            // /*  To Delete  */

            timer.startTimer();
            for(int i = 0; i < size; i ++){ 
                grid_size_x = (h_mergesize[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
                block_size.x = BLOCK_SIZE;
                grid_size.x = grid_size_x;
                kernel_unroll<<<grid_size, block_size>>>(d_pixelpairidx, d_pixelpairsize, d_mergesize, (BoxDistRange *)d_BufferInput, d_pairs, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, i, d_loop, (Task *)d_BufferOutput, d_bufferoutput_size, d_resultmap, d_distance);
                if(h_pixelpairsize[i] > 0){
                    h_pixelpairidx[i] += h_mergesize[i];
                    h_pixelpairsize[i] -= h_mergesize[i];
                }
            }
            check_execution("kernel_unroll");
            cudaDeviceSynchronize();
            timer.stopTimer();
            printf("kernel unroll: %f ms\n", timer.getElapsedTime());

            CUDA_SAFE_CALL(cudaMemcpy(d_pixelpairidx, h_pixelpairidx, size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_pixelpairsize, h_pixelpairsize, size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_mergesize, h_mergesize, size * sizeof(int), cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            if(h_bufferoutput_size == 0) break;

            h_loop ++;
            CUDA_SAFE_CALL(cudaMemcpy(d_loop, &h_loop, sizeof(int), cudaMemcpyHostToDevice));

            /* To delete  */
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

            printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
            /* To delete  */

            swap(d_BufferInput, d_BufferOutput);
            swap(d_bufferinput_size, d_bufferoutput_size);
            swap(h_bufferinput_size, h_bufferoutput_size);
            CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

            grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            block_size.x = BLOCK_SIZE;
            grid_size.x = grid_size_x;

            timer.startTimer();
            kernel_refine<<<grid_size, block_size>>>((Task *)d_BufferInput, gctx->d_vertices, d_bufferinput_size, d_distance, d_resultmap, d_max_box_dist);
            cudaDeviceSynchronize();
            check_execution("kernel_refine");
            timer.stopTimer();
            printf("kernel refine: %f ms\n", timer.getElapsedTime());

            CUDA_SAFE_CALL(cudaMemcpy(d_bufferinput_size, d_sortedpairs_size, sizeof(uint), cudaMemcpyDeviceToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_BufferInput, d_sortedpairs, h_sortpairs_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));
        }
        duration.stopTimer();
        printf("batch time = %lf ms\n", duration.getElapsedTime());

        CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(bool), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < size; i++)
        {
            if (h_distance[i] <= WITHIN_DISTANCE)
                found++;
            // printf("%lf\n", h_distance[i]);
        }
    }
    total.startTimer();
    printf("GPU total time = %lf ms\n", total.getElapsedTime());
    return found;
}