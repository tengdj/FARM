#include "geometry.cuh"
#include "Ideal.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>
#include <thrust/count.h>

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


struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

__global__ void kernel_filter_intersect(pair<uint32_t,uint32_t>* pairs, IdealOffset *idealoffset,
                                             RasterInfo *info, uint8_t *status, uint size, 
                                             PixPair *pixpairs, uint *pp_size, uint8_t category_count, bool *res)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;  

	const pair<uint32_t, uint32_t> pair = pairs[x];
    const uint32_t src_idx = pair.first;
	const uint32_t tar_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const IdealOffset target = idealoffset[tar_idx];

	const box s_mbr = info[src_idx].mbr, t_mbr = info[tar_idx].mbr;				
	const double s_step_x = info[src_idx].step_x, s_step_y = info[src_idx].step_y; 
	const int s_dimx = info[src_idx].dimx, s_dimy = info[src_idx].dimy;			 
	const double t_step_x = info[tar_idx].step_x, t_step_y = info[tar_idx].step_y; 
	const int t_dimx = info[tar_idx].dimx, t_dimy = info[tar_idx].dimy;			

	int i_min = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx);
	int i_max = gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx);
	int j_min = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy);
	int j_max = gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy);

	for (int i = i_min; i <= i_max; i++)
	{
		for (int j = j_min; j <= j_max; j++)
		{
			int pa = gpu_get_id(i, j, s_dimx);
			PartitionStatus source_status = gpu_show_status(status, source.status_start, pa, category_count);
            if(source_status == OUT) continue;

			box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

            bx.low[0] += 1e-6;
            bx.low[1] += 1e-6;
            bx.high[0] -= 1e-6;
            bx.high[1] -= 1e-6;

			int _i_min = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx);
			int _i_max = gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx);
			int _j_min = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy);
			int _j_max = gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy);

			for (int _i = _i_min; _i <= _i_max; _i++)
			{
				for (int _j = _j_min; _j <= _j_max; _j++)
				{
					int pb = gpu_get_id(_i, _j, t_dimx);

					PartitionStatus target_status = gpu_show_status(status, target.status_start, pb, category_count);

					if (target_status == OUT) continue;

                    if(source_status == IN || target_status == IN){
                        res[x] = true;
                    }else{
                        assert(source_status == BORDER && target_status == BORDER);
                        int idx = atomicAdd(pp_size, 1U);
                        pixpairs[idx] = {pa, pb, x};
                    }
                }
			}
		}
	}
    return;
}

__global__ void kernel_calculate_probability(PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint *size, PixelPairWithProb *p_pixpairs, uint *p_pixpairs_size, uint8_t category_count, bool *res){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < *size){
        int pa = pixpairs[x].source_pixid;
        int pb = pixpairs[x].target_pixid;
        int pair_id = pixpairs[x].pair_id;

        if(res[pair_id]) return;

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
        double probability = (pa_apx + pb_apx) / max(pa_pixelArea, pb_pixelArea);

        if(probability > 1.0){
            res[pair_id] = true;
            return;
        }

        int idx = atomicAdd(p_pixpairs_size, 1U);
        p_pixpairs[idx] = {probability, pa, pb, pair_id};
    }
}

__global__ void kernel_merge_intersect(PixelPairWithProb *pixpairs, int *pixelpairidx, int *pixelpairsize, int pairsize, PixPair *buffer, uint *buffer_size, bool *res, double threshold){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairsize && pixelpairsize[tid + 1] - pixelpairidx[tid] > 0)
    {
        int start = pixelpairidx[tid];
        int end = pixelpairidx[tid + 1];
        int pairId = pixpairs[start].pair_id;

        if(res[pairId]){
            pixelpairidx[tid] = end;
            return;
        }
        
        double obj_prob = 1;
        for(int i = start; i < end; i ++){
            double pix_prob = pixpairs[i].probability;
            assert(pix_prob > 0.0);
            obj_prob = obj_prob * (1 - pix_prob);
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = {pixpairs[i].source_pixid, pixpairs[i].target_pixid, pairId};
            if(1 - obj_prob >= threshold) { 
                pixelpairidx[tid] = i + 1;
                return;
            } 
        }
        pixelpairidx[tid] = end;
    }
}

__global__ void kernel_unroll_intersect(PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs,
											 IdealOffset *idealoffset, uint8_t *status,
											 uint32_t *es_offset, EdgeSeq *edge_sequences,
											 uint *size, Task *tasks, uint *task_size)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= *size) return;
	
	int p = pixpairs[idx].source_pixid;
	int p2 = pixpairs[idx].target_pixid;
	int pair_id = pixpairs[idx].pair_id;

	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t src_idx = pair.first;
    const uint32_t tar_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
    const IdealOffset target = idealoffset[tar_idx];

	uint s_offset_start = source.offset_start;
	uint t_offset_start = target.offset_start;
	uint s_edge_sequences_start = source.edge_sequences_start;
	uint t_edge_sequences_start = target.edge_sequences_start;

	int s_num_sequence = (es_offset + s_offset_start)[p + 1] - (es_offset + s_offset_start)[p];
	int t_num_sequence = (es_offset + t_offset_start)[p2 + 1] - (es_offset + t_offset_start)[p2];
	uint s_vertices_start = source.vertices_start;
	uint t_vertices_start = target.vertices_start;

	const int max_size = 16;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
		for (int j = 0; j < t_num_sequence; ++j)
		{
	 		EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
			for (uint s = 0; s < r.length; s += max_size)
			{
				uint end_s = min(s + max_size, r.length);
				for (uint t = 0; t < r2.length; t += max_size)
				{
					uint end_t = min(t + max_size, r2.length);

					uint idx_task = atomicAdd(task_size, 1U);
					tasks[idx_task].s_start = s_vertices_start + r.start + s;
					tasks[idx_task].t_start = t_vertices_start + r2.start + t;
					tasks[idx_task].s_length = end_s - s;
					tasks[idx_task].t_length = end_t - t;
					tasks[idx_task].pair_id = pair_id;
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

    if(gpu_segment_intersect_batch((d_vertices + s1), (d_vertices + s2), len1, len2)){
	    res[pair_id] = true;
    }

    return;
}


__global__ void kernel_refinement_intersect2(PixPair *pixpairs, pair<uint32_t,uint32_t> *pairs, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status,
											 uint32_t *es_offset, EdgeSeq *edge_sequences, Point *d_vertices, 
                                             uint32_t *gridline_offset, double *gridline_nodes, uint *size, bool *res)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;

	int pair_id = pixpairs[x].pair_id;

    pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t src_idx = info[pair.first].step_x > info[pair.second].step_x ? pair.first : pair.second;
    const uint32_t tar_idx = info[pair.first].step_x > info[pair.second].step_x ? pair.second : pair.first;
	int p = info[pair.first].step_x > info[pair.second].step_x ? pixpairs[x].source_pixid : pixpairs[x].target_pixid;
	int p2 = info[pair.first].step_x > info[pair.second].step_x ? pixpairs[x].target_pixid : pixpairs[x].source_pixid;

	const IdealOffset source = idealoffset[src_idx];
    const IdealOffset target = idealoffset[tar_idx];

    uint s_offset_start = source.offset_start;
	uint t_offset_start = target.offset_start;
	uint s_edge_sequences_start = source.edge_sequences_start;
	uint t_edge_sequences_start = target.edge_sequences_start;

	int s_num_sequence = (es_offset + s_offset_start)[p + 1] - (es_offset + s_offset_start)[p];
	int t_num_sequence = (es_offset + t_offset_start)[p2 + 1] - (es_offset + t_offset_start)[p2];
	uint s_vertices_start = source.vertices_start;
	uint t_vertices_start = target.vertices_start;

    for(int i = 0; i < s_num_sequence; i ++){
        EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
		for (int j = 0; j < t_num_sequence; ++j){
	 		EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
            if(gpu_segment_intersect_batch((d_vertices + s_vertices_start + r.start), (d_vertices + t_vertices_start + r2.start), r.length, r2.length)){
                res[pair_id] = true;
                // return;
            }
        }	
    }

    const box mbr = info[src_idx].mbr;
    const double step_x = info[src_idx].step_x;
    const double step_y = info[src_idx].step_y;
    const int dimx = info[src_idx].dimx;
    const int dimy = info[src_idx].dimy;

    for(int j = 0; j < t_num_sequence; j ++){
        EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
        Point pt = (d_vertices + t_vertices_start)[r2.start];
        int xoff = gpu_get_offset_x(mbr.low[0], pt.x, step_x, dimx);
        int yoff = gpu_get_offset_y(mbr.low[1], pt.y, step_y, dimy);
        const box bx = gpu_get_pixel_box(xoff, yoff, mbr.low[0], mbr.low[1], step_x, step_y);

        bool ret = false;
        // if(pair_id == 11 && abs(pt.x - 83.872009) < 1e-9) std::printf("POINT (%lf %lf) ret = %d\n", pt.x, pt.y, ret);
        for(int i = 0; i < s_num_sequence; i ++){
            EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
            for(uint s = 0; s < r.length; s ++){
                Point v1 = (d_vertices + s_vertices_start + r.start)[s];
                Point v2 = (d_vertices + s_vertices_start + r.start)[s + 1];

                if ((v1.y >= pt.y) != (v2.y >= pt.y)){
                    double int_x = (v2.x - v1.x) * (pt.y - v1.y) / (v2.y - v1.y) + v1.x;
                    if(pt.x <= int_x && int_x <= bx.high[0]){
                        ret = !ret;
                    }
                }
            }           
        }
        
        int nc = 0;
        const uint32_t gridline_start = source.gridline_offset_start;
        const uint32_t i_start = (gridline_offset + gridline_start)[xoff + 1];
        const uint32_t i_end = (gridline_offset + gridline_start)[xoff + 2];

        nc = binary_search_count((gridline_nodes + source.gridline_nodes_start), i_start, i_end, pt.y);
        // if(pair_id == 466776) std::printf("POINT (%lf %lf) ret = %d nc = %d\n", pt.x, pt.y, ret, nc);
        if(nc % 2 == 1){
            ret = !ret;
        }
        // if(pair_id == 15) printf("ret = %d\n", ret);
        if(ret) {
            res[pair_id] = true;
            // return;
        }
    }

    return;
}

__global__ void statistic_result(bool *res, uint size, uint *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;
	if(res[x] == true) atomicAdd(result, 1);
}

void cuda_intersect(query_context *gctx)
{
	size_t batch_size = gctx->index_end - gctx->index;
    uint h_bufferinput_size, h_bufferoutput_size;
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

    bool *d_res = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_res, batch_size * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMemset(d_res, 0, batch_size * sizeof(bool)));

	/*1. Raster Model Filtering*/
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_filter_intersect<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, batch_size, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size, gctx->category_count, d_res);
    cudaDeviceSynchronize();
    check_execution("kernel_filter_intersect");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_calculate_probability<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, gctx->d_bufferinput_size, (PixelPairWithProb*)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->category_count, d_res);
    cudaDeviceSynchronize();
    check_execution("kernel_calculate_probability");

    CUDA_SWAP_BUFFER();

    if(!gctx->use_approximation){

        int num_pixel_pairs = h_bufferinput_size;

        thrust::device_ptr<PixelPairWithProb> begin = thrust::device_pointer_cast((PixelPairWithProb*)gctx->d_BufferInput);
        thrust::device_ptr<PixelPairWithProb> end = thrust::device_pointer_cast((PixelPairWithProb*)gctx->d_BufferInput + h_bufferinput_size);
        thrust::sort(thrust::device, begin, end, 
            [] __device__(const PixelPairWithProb &a, const PixelPairWithProb &b) {
                if(a.pair_id != b.pair_id){
                    return a.pair_id < b.pair_id;
                }else{
                    return a.probability > b.probability;
                }
        });

        thrust::device_vector<int> d_indices(num_pixel_pairs);
        thrust::sequence(d_indices.begin(), d_indices.end());

        thrust::device_vector<int> pair_ids(num_pixel_pairs);
        thrust::transform(begin, end, pair_ids.begin(), 
            [] __device__(const PixelPairWithProb &r){
                return r.pair_id;});

        thrust::device_vector<int> d_flags(num_pixel_pairs);
        thrust::adjacent_difference(thrust::device, pair_ids.begin(), pair_ids.end(), d_flags.begin());
        
        thrust::transform(d_flags.begin(), d_flags.end(), d_flags.begin(),
            [] __device__(int x){ return x != 0 ? 1 : 0; });

        d_flags[0] = 1;	

        int num_groups = thrust::count(d_flags.begin(), d_flags.end(), 1);

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

        PixelPairWithProb* d_pixpairs = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pixpairs, h_bufferinput_size * sizeof(PixelPairWithProb)));
        CUDA_SAFE_CALL(cudaMemcpy(d_pixpairs, gctx->d_BufferInput, h_bufferinput_size * sizeof(PixelPairWithProb), cudaMemcpyDeviceToDevice));

        int *d_end_ptr = nullptr; 
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_end_ptr, (num_groups + 1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(d_end_ptr, d_start_ptr, (num_groups + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

        int iterative_round = 0;
        while(true){
            iterative_round ++;
            grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;

            kernel_merge_intersect<<<grid_size, block_size>>>(d_pixpairs, d_start_ptr, d_end_ptr, num_groups, (PixPair*)gctx->d_BufferOutput, gctx->d_bufferoutput_size, d_res, gctx->merge_threshold);
            cudaDeviceSynchronize();
            check_execution("kernel_calculate_probability");

            CUDA_SWAP_BUFFER();
            if(h_bufferinput_size == 0) break;

            // 2. Unroll Refinement

            grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            kernel_unroll_intersect<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
            // kernel_refinement_intersect2<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferinput_size, d_res);
            cudaDeviceSynchronize();
            check_execution("kernel_unroll_intersect");

            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            /*3. Refinement step*/

            grid_size = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            kernel_refinement_intersect<<<grid_size, block_size>>>((Task *)gctx->d_BufferOutput, gctx->d_vertices, gctx->d_bufferoutput_size, d_res);
            cudaDeviceSynchronize();
            check_execution("kernel_refinement_intersect");

            CUDA_SWAP_BUFFER();
        }
    }

    grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    statistic_result<<<grid_size, block_size>>>(d_res, batch_size, gctx->d_result);
    cudaDeviceSynchronize();
    check_execution("statistic_result"); 

	uint h_result;
	CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
	gctx->found += h_result;
    printf("num_intersect = %d\n", gctx->found);
    return;
}
