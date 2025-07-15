#include "geometry.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>

#define WITHIN_DISTANCE 10

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

struct BinData{
    uint8_t pair_wise_fullness;
    double ratio;
};

struct BoxDistRange
{
    int sourcePixelId;
    int targetPixelId;
    int pairId;
    float minDist;
    float maxDist;
    uint8_t s_cur_level;
    uint8_t t_cur_level;

    void print() const {
        printf("%d %d %f %f\n", sourcePixelId, targetPixelId, minDist, maxDist);
    }
};

__device__ bool d_flag = false;

__global__ void kernel_init_distance(pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *layer_info, uint size, float *max_box_dist, BoxDistRange *buffer, uint *buffer_size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        pair<uint32_t, uint32_t> &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.first];
        IdealOffset &target = idealoffset[pair.second];

        int s_dimx = (layer_info + source.layer_start)[0].dimx, s_dimy = (layer_info + source.layer_start)[0].dimy;
        int t_dimx = (layer_info + target.layer_start)[0].dimx, t_dimy = (layer_info + target.layer_start)[0].dimy;

        for(int i = 0; i < s_dimx * s_dimy; i ++){
            for(int j = 0; j < t_dimx * t_dimy; j ++){
                uint idx = atomicAdd(buffer_size, 1);
                buffer[idx] = {i, j, pair_id, 0.0, FLT_MAX, 0, 0};
            }
        }
       
        // buffer[pair_id] = {0, 0, pair_id, 0.0, FLT_MAX, 0, 0};
        max_box_dist[pair_id] = FLT_MAX;
    }
}

// calculate lower bound and upper bound between box from (top down)
__global__ void iterative_filtering_step1(BoxDistRange *candidate, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, float *max_box_dist, uint *size, BoxDistRange *buffer, uint *buffer_size, float *degree_per_kilometer_latitude, float *degree_per_kilometer_longitude_arr, uint8_t category_count)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (candidate_id < *size)
    {
        d_flag = false;

        int source_pixel_id = candidate[candidate_id].sourcePixelId;
        int target_pixel_id = candidate[candidate_id].targetPixelId;
        int pair_id = candidate[candidate_id].pairId;
        uint8_t s_cur_level = candidate[candidate_id].s_cur_level;
        uint8_t t_cur_level = candidate[candidate_id].t_cur_level;

        // printf("s_cur_level = %d, t_cur_level = %d\n", s_cur_level, t_cur_level);

        pair<uint32_t, uint32_t> &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.first];
        IdealOffset &target = idealoffset[pair.second];
        uint s_level = idealoffset[pair.first + 1].layer_start - source.layer_start - 1;
        uint t_level = idealoffset[pair.second + 1].layer_start - target.layer_start - 1;
        
        // printf("s_level = %d, t_level = %d\n", s_level, t_level);

        // int source_start_x, source_start_y, source_end_x, source_end_y, target_start_x, target_start_y, target_end_x, target_end_y;
        // uint32_t source_offset, target_offset;
        // box s_mbr, t_mbr;
        // double s_step = (layer_info + source.layer_start)[s_cur_level].step_x, t_step = (layer_info + target.layer_start)[t_cur_level].step_x;
        // double s_step_x, s_step_y, t_step_x, t_step_y;
        // int s_dimx, s_dimy, t_dimx, t_dimy;
        // box source_pixel_box, target_pixel_box;

        // // printf("s_step = %lf, t_step = %lf\n", s_step, t_step);

        // if(s_cur_level < s_level){
        //     d_flag = true;
        //     s_cur_level ++;

        //     source_offset = (layer_offset + source.layer_start)[s_cur_level];
        //     s_mbr = (layer_info + source.layer_start)[s_cur_level].mbr;
        //     s_step_x = (layer_info + source.layer_start)[s_cur_level].step_x, s_step_y = (layer_info + source.layer_start)[s_cur_level].step_y;
        //     s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

        //     // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", s_mbr.low[0],s_mbr.low[1],
		// 	// 	s_mbr.high[0],s_mbr.low[1],
		// 	// 	s_mbr.high[0],s_mbr.high[1],
		// 	// 	s_mbr.low[0],s_mbr.high[1],
		// 	// 	s_mbr.low[0],s_mbr.low[1]);
            
        //     // printf("s_step_x = %lf, s_step_y = %lf, s_dimx = %d, s_dimy = %d\n", s_step_x, s_step_y, s_dimx, s_dimy);

        //     source_pixel_box = gpu_get_pixel_box(
        //         gpu_get_x(source_pixel_id, (layer_info + source.layer_start)[s_cur_level - 1].dimx),
        //         gpu_get_y(source_pixel_id, (layer_info + source.layer_start)[s_cur_level - 1].dimx, (layer_info + source.layer_start)[s_cur_level - 1].dimy),
        //         (layer_info + source.layer_start)[s_cur_level - 1].mbr.low[0], (layer_info + source.layer_start)[s_cur_level - 1].mbr.low[1],
        //         (layer_info + source.layer_start)[s_cur_level - 1].step_x, (layer_info + source.layer_start)[s_cur_level - 1].step_y);
        //     source_pixel_box.low[0] += 0.000001;
        //     source_pixel_box.low[1] += 0.000001;
        //     source_pixel_box.high[0] -= 0.000001;
        //     source_pixel_box.high[1] -= 0.000001;

        //     // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", source_pixel_box.low[0],source_pixel_box.low[1],
		// 	// 	source_pixel_box.high[0],source_pixel_box.low[1],
		// 	// 	source_pixel_box.high[0],source_pixel_box.high[1],
		// 	// 	source_pixel_box.low[0],source_pixel_box.high[1],
		// 	// 	source_pixel_box.low[0],source_pixel_box.low[1]);

        //     source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
        //     source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
        //     source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
        //     source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
        // }else{
        //     source_offset = (layer_offset + source.layer_start)[s_cur_level];
        //     s_mbr = (layer_info + source.layer_start)[s_cur_level].mbr;
        //     s_step_x = (layer_info + source.layer_start)[s_cur_level].step_x, s_step_y = (layer_info + source.layer_start)[s_cur_level].step_y;
        //     s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

        //     source_start_x = gpu_get_x(source_pixel_id, s_dimx);
        //     source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        //     source_end_x = gpu_get_x(source_pixel_id, s_dimx);
        //     source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        // }

        // if(t_cur_level < t_level){
        //     d_flag = true;
        //     t_cur_level ++;

        //     target_offset = (layer_offset + target.layer_start)[t_cur_level];
        //     t_mbr = (layer_info + target.layer_start)[t_cur_level].mbr;
        //     t_step_x = (layer_info + target.layer_start)[t_cur_level].step_x, t_step_y = (layer_info + target.layer_start)[t_cur_level].step_y;
        //     t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

        //     // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", t_mbr.low[0],t_mbr.low[1],
		// 	// 	t_mbr.high[0],t_mbr.low[1],
		// 	// 	t_mbr.high[0],t_mbr.high[1],
		// 	// 	t_mbr.low[0],t_mbr.high[1],
		// 	// 	t_mbr.low[0],t_mbr.low[1]);
            
        //     // printf("t_step_x = %lf, t_step_y = %lf, t_dimx = %d, t_dimy = %d\n", t_step_x, t_step_y, t_dimx, t_dimy);

        //     target_pixel_box = gpu_get_pixel_box(
        //         gpu_get_x(target_pixel_id, (layer_info + target.layer_start)[t_cur_level - 1].dimx),
        //         gpu_get_y(target_pixel_id, (layer_info + target.layer_start)[t_cur_level - 1].dimx, (layer_info + target.layer_start)[t_cur_level - 1].dimy),
        //         (layer_info + target.layer_start)[t_cur_level - 1].mbr.low[0], (layer_info + target.layer_start)[t_cur_level - 1].mbr.low[1],
        //         (layer_info + target.layer_start)[t_cur_level - 1].step_x, (layer_info + target.layer_start)[t_cur_level - 1].step_y);
        //     target_pixel_box.low[0] += 0.000001;
        //     target_pixel_box.low[1] += 0.000001;
        //     target_pixel_box.high[0] -= 0.000001;
        //     target_pixel_box.high[1] -= 0.000001;

        //     target_start_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.low[0], t_step_x, t_dimx);
        //     target_start_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.low[1], t_step_y, t_dimy);
        //     target_end_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.high[0], t_step_x, t_dimx);
        //     target_end_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.high[1], t_step_y, t_dimy);
        // }else{
        //     target_offset = (layer_offset + target.layer_start)[t_cur_level];
        //     t_mbr = (layer_info + target.layer_start)[t_cur_level].mbr;
        //     t_step_x = (layer_info + target.layer_start)[t_cur_level].step_x, t_step_y = (layer_info + target.layer_start)[t_cur_level].step_y;
        //     t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

        //     target_start_x = gpu_get_x(target_pixel_id, t_dimx);
        //     target_start_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
        //     target_end_x = gpu_get_x(target_pixel_id, t_dimx);
        //     target_end_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
        // }

        int source_start_x, source_start_y, source_end_x, source_end_y, target_start_x, target_start_y, target_end_x, target_end_y;
        uint32_t source_offset, target_offset;
        box s_mbr, t_mbr;
        double s_step = (layer_info + source.layer_start)[s_cur_level].step_x, t_step = (layer_info + target.layer_start)[t_cur_level].step_x;
        double s_step_x, s_step_y, t_step_x, t_step_y;
        int s_dimx, s_dimy, t_dimx, t_dimy;
        box source_pixel_box, target_pixel_box;

        // printf("s_step = %lf, t_step = %lf\n", s_step, t_step);

        if(s_cur_level < s_level && (s_step >= t_step || t_cur_level >= t_level)){
            d_flag = true;
            s_cur_level ++;

            source_offset = (layer_offset + source.layer_start)[s_cur_level];
            s_mbr = (layer_info + source.layer_start)[s_cur_level].mbr;
            s_step_x = (layer_info + source.layer_start)[s_cur_level].step_x, s_step_y = (layer_info + source.layer_start)[s_cur_level].step_y;
            s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

            // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", s_mbr.low[0],s_mbr.low[1],
			// 	s_mbr.high[0],s_mbr.low[1],
			// 	s_mbr.high[0],s_mbr.high[1],
			// 	s_mbr.low[0],s_mbr.high[1],
			// 	s_mbr.low[0],s_mbr.low[1]);
            
            // printf("s_step_x = %lf, s_step_y = %lf, s_dimx = %d, s_dimy = %d\n", s_step_x, s_step_y, s_dimx, s_dimy);

            source_pixel_box = gpu_get_pixel_box(
                gpu_get_x(source_pixel_id, (layer_info + source.layer_start)[s_cur_level - 1].dimx),
                gpu_get_y(source_pixel_id, (layer_info + source.layer_start)[s_cur_level - 1].dimx, (layer_info + source.layer_start)[s_cur_level - 1].dimy),
                (layer_info + source.layer_start)[s_cur_level - 1].mbr.low[0], (layer_info + source.layer_start)[s_cur_level - 1].mbr.low[1],
                (layer_info + source.layer_start)[s_cur_level - 1].step_x, (layer_info + source.layer_start)[s_cur_level - 1].step_y);
            source_pixel_box.low[0] += 1e-6;
            source_pixel_box.low[1] += 1e-6;
            source_pixel_box.high[0] -= 1e-6;
            source_pixel_box.high[1] -= 1e-6;

            // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", source_pixel_box.low[0],source_pixel_box.low[1],
			// 	source_pixel_box.high[0],source_pixel_box.low[1],
			// 	source_pixel_box.high[0],source_pixel_box.high[1],
			// 	source_pixel_box.low[0],source_pixel_box.high[1],
			// 	source_pixel_box.low[0],source_pixel_box.low[1]);

            source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
            source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
            source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
            source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
        }else{
            source_offset = (layer_offset + source.layer_start)[s_cur_level];
            s_mbr = (layer_info + source.layer_start)[s_cur_level].mbr;
            s_step_x = (layer_info + source.layer_start)[s_cur_level].step_x, s_step_y = (layer_info + source.layer_start)[s_cur_level].step_y;
            s_dimx = (layer_info + source.layer_start)[s_cur_level].dimx, s_dimy = (layer_info + source.layer_start)[s_cur_level].dimy;

            source_start_x = gpu_get_x(source_pixel_id, s_dimx);
            source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
            source_end_x = gpu_get_x(source_pixel_id, s_dimx);
            source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        }

        if(t_cur_level < t_level && (s_step <= t_step || s_cur_level >= s_level)){
            d_flag = true;
            t_cur_level ++;

            target_offset = (layer_offset + target.layer_start)[t_cur_level];
            t_mbr = (layer_info + target.layer_start)[t_cur_level].mbr;
            t_step_x = (layer_info + target.layer_start)[t_cur_level].step_x, t_step_y = (layer_info + target.layer_start)[t_cur_level].step_y;
            t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

            // printf("LINESTRING((%f %f, %f %f, %f %f, %f %f, %f %f))\n", t_mbr.low[0],t_mbr.low[1],
			// 	t_mbr.high[0],t_mbr.low[1],
			// 	t_mbr.high[0],t_mbr.high[1],
			// 	t_mbr.low[0],t_mbr.high[1],
			// 	t_mbr.low[0],t_mbr.low[1]);
            
            // printf("t_step_x = %lf, t_step_y = %lf, t_dimx = %d, t_dimy = %d\n", t_step_x, t_step_y, t_dimx, t_dimy);

            target_pixel_box = gpu_get_pixel_box(
                gpu_get_x(target_pixel_id, (layer_info + target.layer_start)[t_cur_level - 1].dimx),
                gpu_get_y(target_pixel_id, (layer_info + target.layer_start)[t_cur_level - 1].dimx, (layer_info + target.layer_start)[t_cur_level - 1].dimy),
                (layer_info + target.layer_start)[t_cur_level - 1].mbr.low[0], (layer_info + target.layer_start)[t_cur_level - 1].mbr.low[1],
                (layer_info + target.layer_start)[t_cur_level - 1].step_x, (layer_info + target.layer_start)[t_cur_level - 1].step_y);
            target_pixel_box.low[0] += 1e-6;
            target_pixel_box.low[1] += 1e-6;
            target_pixel_box.high[0] -= 1e-6;
            target_pixel_box.high[1] -= 1e-6;

            target_start_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.low[0], t_step_x, t_dimx);
            target_start_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.low[1], t_step_y, t_dimy);
            target_end_x = gpu_get_offset_x(t_mbr.low[0], target_pixel_box.high[0], t_step_x, t_dimx);
            target_end_y = gpu_get_offset_y(t_mbr.low[1], target_pixel_box.high[1], t_step_y, t_dimy);
        }else{
            target_offset = (layer_offset + target.layer_start)[t_cur_level];
            t_mbr = (layer_info + target.layer_start)[t_cur_level].mbr;
            t_step_x = (layer_info + target.layer_start)[t_cur_level].step_x, t_step_y = (layer_info + target.layer_start)[t_cur_level].step_y;
            t_dimx = (layer_info + target.layer_start)[t_cur_level].dimx, t_dimy = (layer_info + target.layer_start)[t_cur_level].dimy;

            target_start_x = gpu_get_x(target_pixel_id, t_dimx);
            target_start_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
            target_end_x = gpu_get_x(target_pixel_id, t_dimx);
            target_end_y = gpu_get_y(target_pixel_id, t_dimx, t_dimy);
        }


        // printf("source %d %d %d %d\n", source_start_x, source_start_y, source_end_x, source_end_y);
        // printf("target %d %d %d %d\n", target_start_x, target_start_y, target_end_x, target_end_y);
        
        for (int x1 = source_start_x; x1 <= source_end_x; x1++)
        {
            for (int y1 = source_start_y; y1 <= source_end_y; y1++)
            {
                int id1 = gpu_get_id(x1, y1, s_dimx);
                auto box1 = gpu_get_pixel_box(x1, y1, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
                for (int x2 = target_start_x; x2 <= target_end_x; x2++)
                {
                    for (int y2 = target_start_y; y2 <= target_end_y; y2++)
                    {
                        int id2 = gpu_get_id(x2, y2, t_dimx);
                        if (gpu_show_status(status, source.status_start, id1, category_count, source_offset) == BORDER && gpu_show_status(status, target.status_start, id2, category_count, target_offset) == BORDER)
                        {  
                            auto box2 = gpu_get_pixel_box(x2, y2, t_mbr.low[0], t_mbr.low[1], t_step_x, t_step_y);
                            float min_distance = gpu_distance(box1, box2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                            float max_distance = gpu_max_distance(box1, box2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
                            if(max_distance <= WITHIN_DISTANCE){
                                atomicMinFloat(max_box_dist + pair_id, -1.0f);
                                return;
                            }
                            if(min_distance > WITHIN_DISTANCE) continue;
            
                            uint idx = atomicAdd(buffer_size, 1);
                            buffer[idx] = {id1, id2, pair_id, min_distance, max_distance, s_cur_level, t_cur_level};
                            atomicMinFloat(max_box_dist + pair_id, max_distance);
                        }
                    }
                }
            }
        }
    }
}

// filter candidate pixel pairs
__global__ void iterative_filtering_step2(BoxDistRange *bufferinput, float *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
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
        }else if(a.minDist != b.minDist){
            return a.minDist < b.minDist; 
        }else{
            return a.maxDist < b.maxDist;
        }

        
    }
};

__global__ void initialize_bindata(BoxDistRange *pixpairs, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint size, BinData *bindata, uint *bindata_size, uint8_t category_count, uint8_t bin_count){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < size){

        // 计算pa.low pa.high pb.low pb.high pa.area pb.area
        int pa = pixpairs[x].sourcePixelId;
        int pb = pixpairs[x].targetPixelId;
        int pair_id = pixpairs[x].pairId;

        pair<uint32_t, uint32_t> pair = pairs[pair_id];
        uint32_t src_idx = pair.first;
        uint32_t tar_idx = pair.second;
        IdealOffset source = idealoffset[src_idx];
        IdealOffset target = idealoffset[tar_idx];

        uint8_t pa_fullness = (status + source.status_start)[pa], pb_fullness = (status + target.status_start)[pb];
        double pa_pixelArea = info[src_idx].step_x * info[src_idx].step_y;
        double pb_pixelArea = info[tar_idx].step_x * info[tar_idx].step_y;
        double pa_low = gpu_decode_fullness(pa_fullness, pa_pixelArea, category_count, true);
        double pa_high = gpu_decode_fullness(pa_fullness, pa_pixelArea, category_count, false);
        double pa_apx = (pa_low + pa_high) / 2;
        double pb_low = gpu_decode_fullness(pb_fullness, pb_pixelArea, category_count, true);
        double pb_high = gpu_decode_fullness(pb_fullness, pb_pixelArea, category_count, false);
        double pb_apx = (pb_low + pb_high) / 2;
        // double ratio = 


        // 计算出当前pair的pair wise fullness

    //     uint8_t pair_wise_fullness = gpu_encode_fullness(pa_apx + pb_apx, pa_pixelArea + pb_pixelArea, bin_count);

    //     // 计算ratio

    //     int idx = atomicAdd(bindata_size, 1);
    //     bindata[idx] = {pair_wise_fullness, }
    // }
}

__global__ void kernel_merge(BoxDistRange *pixpairs, int *pixelpairidx, int *pixelpairsize, uint pairsize, BoxDistRange* buffer, uint *buffer_size, float *max_box_dist)
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

// __global__ void kernel_unroll_within_polygon(BoxDistRange *pixpairs, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, uint32_t *es_offset, EdgeSeq *edge_sequences, uint* size, Task *tasks, uint *task_size)
// {
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         int p = pixpairs[bufferId].sourcePixelId;
//         int p2 = pixpairs[bufferId].targetPixelId;
//         int pairId = pixpairs[bufferId].pairId;

//         pair<uint32_t, uint32_t> &pair = pairs[pairId];
//         IdealOffset &source = idealoffset[pair.first];
//         IdealOffset &target = idealoffset[pair.second];

//         // printf("%d %d %d\n", pair.first, (es_offset + source.offset_start)[p], (es_offset + source.offset_start)[p + 1]);

//         // if((es_offset + source.offset_start)[p] >= (es_offset + source.offset_start)[p + 1]){
//         //     printf("ERROR %d %d %d\n", pair.first, (es_offset + source.offset_start)[p], (es_offset + source.offset_start)[p + 1]);
//         // }

//         int s_num_sequence = (es_offset + source.offset_start)[p + 1] - (es_offset + source.offset_start)[p];
//         int t_num_sequence = (es_offset + target.offset_start)[p2 + 1] - (es_offset + target.offset_start)[p2];

//         // printf("t_num_sequence = %d\n", t_num_sequence);
//         // printf("s_num_sequence = %d\n", s_num_sequence);
//         // printf("s_num_sequence = %d t_num_sequence = %d\n", s_num_sequence, t_num_sequence);

//         for (int i = 0; i < s_num_sequence; ++ i)
//         {
//             EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[p] + i];
//             for (int j = 0; j < t_num_sequence; ++j)
//             {
//                 EdgeSeq r2 = (edge_sequences + target.edge_sequences_start)[(es_offset + target.offset_start)[p2] + j];
//                 // printf("r.length = %d, r2.length = %d\n", r.length, r2.length);
//                 // atomicAdd(task_size, r.length * r2.length);
//                 int max_size = 32;
//                 for (uint s = 0; s < r.length; s += max_size)
//                 {
//                     uint end_s = min(s + max_size, r.length);
//                     for (uint t = 0; t < r2.length; t += max_size)
//                     {
//                         uint end_t = min(t + max_size, r2.length);
//                         int idx = atomicAdd(task_size, 1U);
//                         tasks[idx] = {source.vertices_start + r.start + s, target.vertices_start + r2.start + t, end_s - s, end_t - t, pairId};
//                     }
//                 }
//            }
//         }
//     }
// }

__global__ void kernel_unroll_within_polygon(
    BoxDistRange *pixpairs, 
    pair<uint32_t, uint32_t> *pairs, 
    IdealOffset *idealoffset, 
    uint32_t *es_offset, 
    EdgeSeq *edge_sequences, 
    uint32_t *size, 
    Task *tasks, 
    uint32_t *task_size)
{
    // Shared memory for frequently accessed data
    extern __shared__ char shared_mem[];
    BoxDistRange *shared_pixpair = (BoxDistRange*)shared_mem;
    IdealOffset *shared_source = (IdealOffset*)(shared_pixpair + 1);
    IdealOffset *shared_target = (IdealOffset*)(shared_source + 1);

    // Thread and block indices
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Load pixel pair into shared memory (first thread in block)
    if (bufferId < *size && tid == 0) {
        shared_pixpair[0] = pixpairs[bufferId];
    }
    __syncthreads();

    if (bufferId >= *size) return;

    // Extract pixel pair information
    int p = shared_pixpair[0].sourcePixelId;
    int p2 = shared_pixpair[0].targetPixelId;
    int pairId = shared_pixpair[0].pairId;

    // Load pair and offset data
    pair<uint32_t, uint32_t> pair = pairs[pairId];
    if (tid == 0) {
        shared_source[0] = idealoffset[pair.first];
        shared_target[0] = idealoffset[pair.second];
    }
    __syncthreads();

    // Calculate edge sequence counts
    int s_num_sequence = (es_offset + shared_source[0].offset_start)[p + 1] - 
                         (es_offset + shared_source[0].offset_start)[p];
    int t_num_sequence = (es_offset + shared_target[0].offset_start)[p2 + 1] - 
                         (es_offset + shared_target[0].offset_start)[p2];

    // Dynamic task size based on sequence length
    const uint32_t max_size = (s_num_sequence * t_num_sequence > 1024) ? 16 : 32;

    // Pre-calculate total tasks for this pixel pair
    uint32_t total_tasks = 0;
    for (int i = 0; i < s_num_sequence; ++i) {
        EdgeSeq r = (edge_sequences + shared_source[0].edge_sequences_start)[
            (es_offset + shared_source[0].offset_start)[p] + i];
        for (int j = 0; j < t_num_sequence; ++j) {
            EdgeSeq r2 = (edge_sequences + shared_target[0].edge_sequences_start)[
                (es_offset + shared_target[0].offset_start)[p2] + j];
            total_tasks += ((r.length + max_size - 1) / max_size) * 
                          ((r2.length + max_size - 1) / max_size);
        }
    }

    // Allocate task indices using atomic operation (first thread only)
    uint32_t base_task_idx = 0;
    if (tid == 0) {
        base_task_idx = atomicAdd(task_size, total_tasks);
    }
    __syncthreads();

    // Broadcast base_task_idx to all threads in block
    uint32_t *shared_base_idx = (uint32_t*)(shared_target + 1);
    if (tid == 0) {
        shared_base_idx[0] = base_task_idx;
    }
    __syncthreads();

    // Generate tasks
    uint32_t current_task_idx = shared_base_idx[0];
    for (int i = 0; i < s_num_sequence; ++i) {
        EdgeSeq r = (edge_sequences + shared_source[0].edge_sequences_start)[
            (es_offset + shared_source[0].offset_start)[p] + i];
        for (int j = 0; j < t_num_sequence; ++j) {
            EdgeSeq r2 = (edge_sequences + shared_target[0].edge_sequences_start)[
                (es_offset + shared_target[0].offset_start)[p2] + j];

            for (uint32_t s = 0; s < r.length; s += max_size) {
                uint32_t end_s = min(s + max_size, r.length);
                for (uint32_t t = 0; t < r2.length; t += max_size) {
                    uint32_t end_t = min(t + max_size, r2.length);
                    if (tid == 0) {
                        tasks[current_task_idx] = {
                            shared_source[0].vertices_start + r.start + s,
                            shared_target[0].vertices_start + r2.start + t,
                            end_s - s,
                            end_t - t,
                            pairId
                        };
                        current_task_idx++;
                    }
                }
            }
        }
    }

    // Update task_size if necessary (first thread only)
    if (tid == 0 && current_task_idx > shared_base_idx[0]) {
        atomicMax(task_size, current_task_idx);
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

        float dist = gpu_segment_to_segment_within_batch(vertices + s1, vertices + s2, len1, len2, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);
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

    kernel_init_distance<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_layer_info, batch_size, d_max_box_dist, (BoxDistRange *)gctx->d_BufferInput, gctx->d_bufferinput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_init");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));

    int round = 0;
    bool h_flag;
    while(true){
        printf("Iterative Filtering Round: %d\n", ++ round);

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        iterative_filtering_step1<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr, gctx->category_count);
        cudaDeviceSynchronize();
        check_execution("iterative_filtering_step1");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("calculate box distance h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("calculate box distance h_bufferoutput_size = %u\n", h_bufferoutput_size);

        CUDA_SWAP_BUFFER();

        // PrintBuffer((BoxDistRange*)gctx->d_BufferInput, h_bufferinput_size);
        // printf("----------------------------------------------------------------------------\n");
        
        cudaMemcpyFromSymbol(&h_flag, d_flag, sizeof(bool));
        if(!h_flag) break;
        if(h_bufferinput_size == 0) return;

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        iterative_filtering_step2<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("iterative_filtering_step2");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("filter h_bufferinput_size = %u\n", h_bufferinput_size);
        printf("filter h_bufferoutput_size = %u\n", h_bufferoutput_size);

        CUDA_SWAP_BUFFER();


        // PrintBuffer((BoxDistRange*)gctx->d_BufferInput, h_bufferinput_size);
        // printf("----------------------------------------------------------------------------\n");

        if(h_bufferinput_size == 0) return;
    }

    // CUDA_SWAP_BUFFER();

    assert(h_bufferinput_size > 0);

    int num_pixel_pairs = h_bufferinput_size;

    thrust::device_ptr<BoxDistRange> begin = thrust::device_pointer_cast((BoxDistRange*)gctx->d_BufferInput);
    thrust::device_ptr<BoxDistRange> end = thrust::device_pointer_cast((BoxDistRange*)gctx->d_BufferInput + h_bufferinput_size);
    thrust::sort(thrust::device, begin, end, CompareKeyValuePairs());

    // PrintBuffer((BoxDistRange*)gctx->d_BufferInput, h_bufferinput_size);

    thrust::device_vector<int> d_indices(num_pixel_pairs);
    thrust::sequence(d_indices.begin(), d_indices.end());

    thrust::device_vector<int> pair_ids(num_pixel_pairs);
 	thrust::transform(begin, end, pair_ids.begin(), 
        [] __device__(const BoxDistRange &r){
            return r.pairId;});

    thrust::device_vector<int> d_flags(num_pixel_pairs);
    thrust::adjacent_difference(thrust::device, pair_ids.begin(), pair_ids.end(), d_flags.begin());


    thrust::transform(d_flags.begin(), d_flags.end(), d_flags.begin(),
        [] __device__(int x){ return x != 0 ? 1 : 0; });

    d_flags[0] = 1;	

    uint num_groups = thrust::count(d_flags.begin(), d_flags.end(), 1);

    thrust::device_vector<int> d_starts(num_groups + 1, num_pixel_pairs);

    thrust::copy_if(thrust::device,
        d_indices.begin(), d_indices.end(),
        d_flags.begin(), d_starts.begin(),
        thrust::identity<int>());

    int* d_start_ptr = thrust::raw_pointer_cast(d_starts.data());

    // free up
    thrust::device_vector<int>().swap(d_indices);
    thrust::device_vector<int>().swap(pair_ids);
    thrust::device_vector<int>().swap(d_flags);

    // thrust::host_vector<int> h_starts = d_starts;
	// std::cout << "\nStart positions:\n";
    // for (int i : h_starts) std::cout << i << " ";
    // std::cout << std::endl;

    // return ;

    BoxDistRange* d_pixpairs = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixpairs, h_bufferinput_size * sizeof(BoxDistRange)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pixpairs, gctx->d_BufferInput, h_bufferinput_size * sizeof(BoxDistRange), cudaMemcpyDeviceToDevice));

    int *d_end_ptr = nullptr; 
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_end_ptr, (num_groups + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_end_ptr, d_start_ptr, (num_groups + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

    while(true){
        printf("nun_groups = %d\n", num_groups);

        grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto merge_start = std::chrono::high_resolution_clock::now();
        kernel_merge<<<grid_size, block_size>>>(d_pixpairs, d_start_ptr, d_end_ptr, num_groups, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, d_max_box_dist);
        cudaDeviceSynchronize();
        check_execution("kernel_merge"); 
        auto merge_end = std::chrono::high_resolution_clock::now();
        auto merge_duration = std::chrono::duration_cast<std::chrono::milliseconds>(merge_end - merge_start);

        std::cout << "merge运行时间: " << merge_duration.count() << " 毫秒" << std::endl;

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("h_bufferoutput_size = %d\n", h_bufferoutput_size);

        if(h_bufferoutput_size == 0) break;

        CUDA_SWAP_BUFFER();

        unsigned long long *d_test = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_test, sizeof(unsigned long long)));

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto unroll_start = std::chrono::high_resolution_clock::now();
        kernel_unroll_within_polygon<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_unroll_within_polygon");
        auto unroll_end = std::chrono::high_resolution_clock::now();
        auto unroll_duration = std::chrono::duration_cast<std::chrono::milliseconds>(unroll_end - unroll_start);

        std::cout << "unroll运行时间: " << unroll_duration.count() << " 毫秒" << std::endl;

        unsigned long long h_test;
        CUDA_SAFE_CALL(cudaMemcpy(&h_test, d_test, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        printf("test: %llu\n", h_test);
   
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        printf("h_bufferoutput_size = %d\n", h_bufferoutput_size);

        CUDA_SWAP_BUFFER();

        grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto refine_start = std::chrono::high_resolution_clock::now();
        kernel_refine_within_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, d_max_box_dist, gctx->d_degree_degree_per_kilometer_latitude, gctx->d_degree_per_kilometer_longitude_arr);
        cudaDeviceSynchronize();
        check_execution("kernel_refine_within_polygon");
        auto refine_end = std::chrono::high_resolution_clock::now();
        auto refine_duration = std::chrono::duration_cast<std::chrono::milliseconds>(refine_end - refine_start);

        std::cout << "refine运行时间: " << refine_duration.count() << " 毫秒" << std::endl;

    }

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    statistic_result_polygon<<<grid_size, block_size>>>(d_max_box_dist, batch_size, gctx->d_result);
    cudaDeviceSynchronize();
    check_execution("statistic_result");

    // PrintBuffer((float*)d_max_box_dist, batch_size);

    uint h_result;
    CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
    gctx->found += h_result;

    CUDA_SAFE_CALL(cudaFree(d_pixpairs));
    CUDA_SAFE_CALL(cudaFree(d_max_box_dist));

    return;
}