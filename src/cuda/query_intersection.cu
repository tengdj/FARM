#include "geometry.cuh"
#include "Ideal.h"
#include "shared_kernels.cuh" 
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

// flags: 0(not contain), 1(maybe contain), 2(contain)
__global__ void kernel_filter_contain_polygon(pair<uint32_t,uint32_t>* pairs, IdealOffset *idealoffset,
                                             RasterInfo *info, uint8_t *status, uint size, 
                                             PixPair *pixpairs, uint *pp_size, int8_t *flags)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size || flags[x] != 1) return;  

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
			int p = gpu_get_id(i, j, s_dimx);
			uint8_t source_status = gpu_show_status(status, source.status_start, p);

			box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

			int _i_min = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx);
			int _i_max = gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx);
			int _j_min = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy);
			int _j_max = gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy);

			for (int _i = _i_min; _i <= _i_max; _i++)
			{
				for (int _j = _j_min; _j <= _j_max; _j++)
				{
					int p2 = gpu_get_id(_i, _j, t_dimx);

					uint8_t target_status = gpu_show_status(status, target.status_start, p2);

					if (source_status == BORDER && target_status == BORDER)
					{
						int idx = atomicAdd(pp_size, 1U);
						
						pixpairs[idx].source_pixid = p;
						pixpairs[idx].target_pixid = p2;
						pixpairs[idx].pair_id = x;
					}
				}
			}
		}
	}
    return;
}

__global__ void kernel_unroll_contain_polygon(PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs,
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

	// printf("(es_offset + t_offset_start)[p2] = %u, (es_offset + t_offset_start)[p2 + 1] = %u\n", (es_offset + t_offset_start)[p2], (es_offset + t_offset_start)[p2 + 1]);
	uint s_vertices_start = source.vertices_start;
	uint t_vertices_start = target.vertices_start;

	const int max_size = 4;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
		for (int j = 0; j < t_num_sequence; ++j)
		{
	 		EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
			// printf("r.length = %d, r2.length = %d\n", r.length, r2.length);
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

__global__ void kernel_refinement_contain_polygon(Task *tasks, Point *d_vertices, uint *size, Intersection* intersections, uint* num)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;
	
	uint s1 = tasks[x].s_start;
	uint s2 = tasks[x].t_start;
	uint len1 = tasks[x].s_length;
	uint len2 = tasks[x].t_length;
	int pair_id = tasks[x].pair_id;

	gpu_segment_intersect_batch(d_vertices, s1, s2, s1 + len1, s2 + len2, pair_id, intersections, num);
}

struct CompareSourceIntersections {
	__host__ __device__
	bool operator()(const Intersection& a, const Intersection& b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.edge_source_id != b.edge_source_id){
			return a.edge_source_id < b.edge_source_id;
		}
        return a.t < b.t; 
    }
};

struct CompareTargetIntersections {
    __host__ __device__
    bool operator()(const Intersection& a, const Intersection& b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.edge_target_id != b.edge_target_id){
			return a.edge_target_id < b.edge_target_id;
		}
        return a.u < b.u; 
    }
};

struct IntersectionEqual {
    __host__ __device__
    bool operator()(const Intersection& a, const Intersection& b) const {
        // return a.pair_id == b.pair_id && a.edge_source_id == b.edge_source_id && a.edge_target_id == b.edge_target_id && a.t == b.t && a.u == b.u;
		return a.p == b.p;
	}
};


__global__ void find_inters_per_pair(Intersection *intersections, uint* size, uint *inters_per_pair){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < *size)
    {
        int pair_id = intersections[x].pair_id;
        atomicAdd(inters_per_pair + pair_id, 1);
    }
}

// 还需要整理
__global__ void make_segments(Intersection *intersections, uint *num_intersections, Segment *segments, uint *num_segments, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, uint *inters_per_pair, bool is_source)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < *num_intersections){
		int pair_id = intersections[x].pair_id;
		if(x + 1 >= *num_intersections || intersections[x].pair_id != intersections[x + 1].pair_id){
			Intersection a = intersections[x];
			Intersection b = intersections[x + 1 - inters_per_pair[pair_id]];
			if(a.p == b.p) return;
			int a_edge_id = is_source ? a.edge_source_id : a.edge_target_id;
			int b_edge_id = is_source ? b.edge_source_id : b.edge_target_id;
			float a_param = is_source ? a.t : a.u;
			float b_param = is_source ? b.t : b.u;
			if(fabs(a_param - 1.0) < eps) a_edge_id ++;
			if(fabs(b_param) < eps) b_edge_id --;
			int idx = atomicAdd(num_segments, 1);
			if(a_edge_id == b_edge_id){
				segments[idx] = {is_source, a.p, b.p, -1, -1, pair_id};
			}else{
				segments[idx] = {is_source, a.p, b.p, a_edge_id + 1, b_edge_id, pair_id};
			}
		}else{
			Intersection a = intersections[x];
			Intersection b = intersections[x + 1];
			if(a.p == b.p) printf("CHECK PAIR: %d POINT(%lf %lf) POINT(%lf %lf)\n", pair_id, a.p.x, a.p.y, b.p.x, b.p.y);
			int a_edge_id = is_source ? a.edge_source_id : a.edge_target_id;
			int b_edge_id = is_source ? b.edge_source_id : b.edge_target_id;
			float a_param = is_source ? a.t : a.u;
			float b_param = is_source ? b.t : b.u;
			int idx = atomicAdd(num_segments, 1);
			if(a_edge_id != b_edge_id){
				if(fabs(a_param - 1) < eps) a_edge_id ++;
				if(fabs(b_param) < eps) b_edge_id --;
				if(a_edge_id == b_edge_id){
					segments[idx] = {is_source, a.p, b.p, -1, -1, pair_id};
				}else{
					segments[idx] = {is_source, a.p, b.p, a_edge_id + 1, b_edge_id, pair_id};					
				}
			}else{	
				segments[idx] = {is_source, a.p, b.p, -1, -1, pair_id};
			}	
		}
	}
}

void cuda_contain_polygon(query_context *gctx)
{
	size_t batch_size = gctx->index_end - gctx->index;
    uint h_bufferinput_size, h_bufferoutput_size;
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

	int threadsPerBlock = 256;
	int blocksPerGrid = batch_size; // 每个块处理一个多边形对

	auto comparePolygons_start = std::chrono::high_resolution_clock::now();
	comparePolygons<<<blocksPerGrid, threadsPerBlock>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_vertices, batch_size, gctx->d_flags);
	cudaDeviceSynchronize();
    check_execution("comparePolygons");
	auto comparePolygons_end = std::chrono::high_resolution_clock::now();
	auto comparePolygons_duration = std::chrono::duration_cast<std::chrono::milliseconds>(comparePolygons_end - comparePolygons_start);
	std::cout << "comparePolygons time: " << comparePolygons_duration.count() << " ms" << std::endl;

	/*1. Raster Model Filtering*/
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

	auto filter_start = std::chrono::high_resolution_clock::now();
    kernel_filter_contain_polygon<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, batch_size, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size, gctx->d_flags);
    cudaDeviceSynchronize();
    check_execution("kernel_filter_contain_polygon");
	auto filter_end = std::chrono::high_resolution_clock::now();
	auto filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_end - filter_start);
	std::cout << "filter time: " << filter_duration.count() << " ms" << std::endl;

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferinput_size);

	if(h_bufferinput_size == 0) return;
	
    /*2. Unroll Refinement*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	auto unroll_start = std::chrono::high_resolution_clock::now();
    kernel_unroll_contain_polygon<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_unroll_contain_polygon");
	auto unroll_end = std::chrono::high_resolution_clock::now();
	auto unroll_duration = std::chrono::duration_cast<std::chrono::milliseconds>(unroll_end - unroll_start);
	std::cout << "unroll time: " << unroll_duration.count() << " ms" << std::endl;

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferoutput_size);

  	CUDA_SWAP_BUFFER();
  
    /*3. Refinement step*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	auto refine_start = std::chrono::high_resolution_clock::now();
    kernel_refinement_contain_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, (Intersection *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_contain_polygon");
	auto refine_end = std::chrono::high_resolution_clock::now();
	auto refine_duration = std::chrono::duration_cast<std::chrono::milliseconds>(refine_end - refine_start);
	std::cout << "refine time: " << refine_duration.count() << " ms" << std::endl;

	CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferoutput_size);

	CUDA_SWAP_BUFFER();

	if(h_bufferinput_size == 0) return;

	auto segment_start = std::chrono::high_resolution_clock::now();

	// check source polygon edges 
    thrust::device_ptr<Intersection> begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    thrust::device_ptr<Intersection> end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);

	// 排序并去重
	thrust::sort(thrust::device, begin, end, 
    [] __device__(const Intersection &a, const Intersection &b) {
		if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else{
			return a.p < b.p;
		}
    });

	auto new_end = thrust::unique(begin, end,
    [] __device__(const Intersection &a, const Intersection &b) {
        return a.pair_id == b.pair_id && a.p == b.p;
    });

    h_bufferinput_size = thrust::distance(begin, new_end);
	end = begin + h_bufferinput_size;

	auto &num_intersections = h_bufferinput_size;

	// 更新设备内存
	CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, 
							sizeof(uint), cudaMemcpyHostToDevice));

	// end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);
    // thrust::sort(thrust::device, begin, end, CompareSourceIntersections());

	// PrintBuffer((Intersection *)gctx->d_BufferInput, num_intersections);

	grid_size = (num_intersections + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint *d_inters_per_pair = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_inters_per_pair, num_intersections * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_inters_per_pair, 0, num_intersections * sizeof(uint)));

	find_inters_per_pair<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, d_inters_per_pair);
	cudaDeviceSynchronize();
    check_execution("kernel_find_inters_per_pair");

	thrust::sort(thrust::device, begin, end, CompareSourceIntersections());
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, d_inters_per_pair, true);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	// check target polygon edges 

	begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + num_intersections);
    thrust::sort(thrust::device, begin, end, CompareTargetIntersections());
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, d_inters_per_pair, false);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	CUDA_SWAP_BUFFER();
			
	uint &num_segments = h_bufferinput_size;

	// PrintBuffer((Segment *)gctx->d_BufferInput, num_segments);
	printf("num_segqments = %d\n", num_segments);

	if(num_segments == 0) return;
	
	uint8_t *pip = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **) &pip, num_segments * sizeof(uint8_t)));
	CUDA_SAFE_CALL(cudaMemset(pip, 0, num_segments * sizeof(uint8_t)));

	grid_size = (num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;

	kernel_filter_segment_contain<<<grid_size, block_size>>>(
		(Segment *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, 
		gctx->d_idealoffset, gctx->d_info, gctx->d_status, 
		gctx->d_vertices, num_segments, pip, 
		(PixMapping *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
	cudaDeviceSynchronize();
    check_execution("kernel_filter_segment_contain");
    
	CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

	grid_size = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	kernel_refinement_segment_contain<<<grid_size, block_size>>>(
		(PixMapping *)gctx->d_BufferOutput, (Segment *)gctx->d_BufferInput,
		gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, 
		gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, 
		gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferoutput_size, 
		pip);
	cudaDeviceSynchronize();
    check_execution("kernel_refinement_segment_contain");	

	auto segment_end = std::chrono::high_resolution_clock::now();
	auto segment_duration = std::chrono::duration_cast<std::chrono::milliseconds>(segment_end - segment_start);
	std::cout << "segment time: " << segment_duration.count() << " ms" << std::endl;

	// printf("---------------------------------------------------------------------------------------------------\n");
	
    // // PrintBuffer((Segment *)gctx->d_BufferInput, num_segments);
	
	// // PrintBuffer((bool *) pip, num_segments);
	
	// uint8_t* h_Buffer = new uint8_t[num_segments];
    // CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	// // int _sum = 0;
    // for (int i = 0; i < num_segments; i++) {
	// 	// _sum += h_Buffer[i];
	// 	printf("%d ", h_Buffer[i]);
	// 	if(i % 10 == 9) printf("\n");
    // }

	// // printf("sum = %d\n", _sum);

	// printf("---------------------------------------------------------------------------------------------------\n");

	auto transfer_start = std::chrono::high_resolution_clock::now();
	
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&gctx->segments, num_segments * sizeof(Segment), cudaHostAllocDefault));
	gctx->num_segments = num_segments;
	CUDA_SAFE_CALL(cudaMemcpy(gctx->segments, (Segment *)gctx->d_BufferInput, num_segments * sizeof(Segment), cudaMemcpyDeviceToHost));
    
	CUDA_SAFE_CALL(cudaHostAlloc((void**)&gctx->pip, num_segments * sizeof(uint8_t), cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaMemcpy(gctx->pip, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_inters_per_pair));
	CUDA_SAFE_CALL(cudaFree(pip));

	auto transfer_end = std::chrono::high_resolution_clock::now();
	auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(transfer_end - transfer_start);
	std::cout << "transfer time: " << transfer_duration.count() << " ms" << std::endl;
    return;
}
