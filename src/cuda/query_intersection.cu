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

	uint itn = 0, etn = 0;	 
	bool flag_out = false; 

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

			itn += (source_status == IN);
            etn += (source_status == OUT);

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

                    flag_out = ((source_status == OUT) && (target_status == IN));

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

	// uint total_pixels = (i_max - i_min + 1) * (j_max - j_min + 1);

	// bool is_contained = (itn == total_pixels); 
	// bool is_outside = (etn == total_pixels); 
    
    // if(is_outside || flag_out) flags[x] = 0;
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

__global__ void kernel_refinement_contain_polygon(Task *tasks, Point *d_vertices, uint *size, int8_t *flags, uint *result, Intersection* intersections, uint* num)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;
	
	uint s1 = tasks[x].s_start;
	uint s2 = tasks[x].t_start;
	uint len1 = tasks[x].s_length;
	uint len2 = tasks[x].t_length;
	int pair_id = tasks[x].pair_id;

	bool should_process = (flags[pair_id] == 1);

	bool has_intersection = should_process && gpu_segment_intersect_batch(d_vertices, s1, s2, s1 + len1, s2 + len2, pair_id, intersections, num);

	if (has_intersection) flags[pair_id] = 0;
}

__global__ void statistic_result(int8_t *flags, uint size, uint *result){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;
	if(flags[x] == 1) atomicAdd(result, 1);
}

struct CompareSourceIntersections {
    __host__ __device__
    bool operator()(const Intersection& a, const Intersection& b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.edge_source_id != b.edge_source_id){
			return a.edge_source_id < b.edge_source_id;
		}else if (a.t != b.t){
        	return a.t < b.t; 
		}else if(a.edge_target_id != b.edge_target_id){
			return a.edge_target_id < b.edge_target_id;
		}else{
			return a.u < b.u;
		}
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
        return a.pair_id == b.pair_id && a.edge_source_id == b.edge_source_id && a.edge_target_id == b.edge_target_id && a.t == b.t && a.u == b.u;
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
			int a_edge_id = is_source ? a.edge_source_id : a.edge_target_id;
			int b_edge_id = is_source ? b.edge_source_id : b.edge_target_id;
			double a_param = is_source ? a.t : a.u;
			double b_param = is_source ? b.t : b.u;
			if(fabs(a_param - 1) < 1e-9) a_edge_id ++;
			if(fabs(b_param) < 1e-9) b_edge_id --;
			int idx = atomicAdd(num_segments, 1);
			segments[idx] = {is_source, a.p, b.p, a_edge_id + 1, b_edge_id, pair_id};
		}else{
			Intersection a = intersections[x];
			Intersection b = intersections[x + 1];
			int a_edge_id = is_source ? a.edge_source_id : a.edge_target_id;
			int b_edge_id = is_source ? b.edge_source_id : b.edge_target_id;
			double a_param = is_source ? a.t : a.u;
			double b_param = is_source ? b.t : b.u;
			int idx = atomicAdd(num_segments, 1);
			if(a_edge_id != b_edge_id){
				if(fabs(a_param - 1) < 1e-9) a_edge_id ++;
				if(fabs(b_param) < 1e-9) b_edge_id --;
				if(a_edge_id + 1 <= b_edge_id)
					segments[idx] = {is_source, a.p, b.p, a_edge_id + 1, b_edge_id, pair_id};
				else
					segments[idx] = {is_source, a.p, b.p, -1, -1, pair_id};
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

	comparePolygons<<<blocksPerGrid, threadsPerBlock>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_vertices, batch_size, gctx->d_flags);
	cudaDeviceSynchronize();
    check_execution("comparePolygons");

	/*1. Raster Model Filtering*/
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

    kernel_filter_contain_polygon<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, batch_size, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size, gctx->d_flags);
    cudaDeviceSynchronize();
    check_execution("kernel_filter_contain_polygon");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferinput_size);

	if(h_bufferinput_size == 0) return;
	
    /*2. Unroll Refinement*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_unroll_contain_polygon<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_unroll_contain_polygon");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferoutput_size);
    
    CUDA_SWAP_BUFFER();
    /*3. Refinement step*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_refinement_contain_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, gctx->d_flags, gctx->d_result, (Intersection *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_contain_polygon");

	CUDA_SWAP_BUFFER();

	if(h_bufferinput_size == 0) return;

	printf("num_intersections = %d\n", h_bufferinput_size);
	// check source polygon edges 
    thrust::device_ptr<Intersection> begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    thrust::device_ptr<Intersection> end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);
    thrust::sort(thrust::device, begin, end, CompareSourceIntersections());
	auto new_end = thrust::unique(begin, end, IntersectionEqual());
	h_bufferinput_size = thrust::distance(begin, new_end);

	end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);

	thrust::device_vector<Point> d_intersection_points(h_bufferinput_size);
    thrust::transform(
		begin, 
		end, 
		d_intersection_points.begin(),
        [] __device__(const Intersection &intr){ 
			return intr.p; });

	thrust::device_vector<Point> d_unique_point_values(h_bufferinput_size);
    thrust::device_vector<int> d_point_counts(h_bufferinput_size);

	auto unique_new_end = thrust::reduce_by_key(
        d_intersection_points.begin(), d_intersection_points.end(),
        thrust::constant_iterator<int>(1),
        d_unique_point_values.begin(),
        d_point_counts.begin(),
        [] __device__(const Point &a, const Point &b)
        { return a == b; });

	int unique_count = unique_new_end.first - d_unique_point_values.begin();
    d_unique_point_values.resize(unique_count);
    d_point_counts.resize(unique_count);

	thrust::device_vector<bool> d_remove_flags(unique_count);
    thrust::transform(d_point_counts.begin(), d_point_counts.end(), d_remove_flags.begin(),
                      [] __device__(int count)
                      { return count == 2; });
		
 	thrust::device_vector<Point> d_points_to_remove;
	int count_to_remove = thrust::count_if(
        d_point_counts.begin(),
        d_point_counts.end(),
        [] __device__ (int count) { return count == 2; }
    );

	d_points_to_remove.resize(count_to_remove);

	auto points_end = thrust::copy_if(
        d_unique_point_values.begin(), 
        d_unique_point_values.end(),
        d_point_counts.begin(),
        d_points_to_remove.begin(),
        [] __device__ (int count) { return count == 2; }
    );

	size_t points_to_remove_count = points_end - d_points_to_remove.begin();
	
	auto result_end = thrust::remove_if(
        begin, end,
        [points_to_remove = thrust::raw_pointer_cast(d_points_to_remove.data()),
         count = points_to_remove_count] __device__ (const Intersection& intr) {
            // 检查当前交点的点是否在需要删除的点列表中
            for (int i = 0; i < count; i++) {
                if (intr.p == points_to_remove[i]) {
                    return true;  // 删除此交点
                }
            }
            return false;  // 保留此交点
        }
    );

	h_bufferinput_size = thrust::distance(begin, result_end);

	end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);
	new_end = thrust::unique(
		begin, 
		end, 
		[] __device__(const Intersection &a, const Intersection &b){ 
			return a.p == b.p; });

	h_bufferinput_size = thrust::distance(begin, new_end);
	uint &num_intersections = h_bufferinput_size;
	CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, sizeof(uint), cudaMemcpyHostToDevice));

	// PrintBuffer((Intersection *)gctx->d_BufferInput, num_intersections);

	grid_size = (num_intersections + BLOCK_SIZE - 1) / BLOCK_SIZE;
	printf("num_intersections = %d\n", num_intersections);
    uint *d_inters_per_pair = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_inters_per_pair, num_intersections * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_inters_per_pair, 0, num_intersections * sizeof(uint)));

	find_inters_per_pair<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, d_inters_per_pair);
	cudaDeviceSynchronize();
    check_execution("kernel_find_inters_per_pair");
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, d_inters_per_pair, true);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	// check target polygon edges 

	begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + num_intersections);
    thrust::sort(thrust::device, begin, end, CompareTargetIntersections());

    // PrintBuffer((Intersection *)gctx->d_BufferInput, num_intersections);
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, d_inters_per_pair, false);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	CUDA_SWAP_BUFFER();
			
	uint &num_segments = h_bufferinput_size;
	
	printf("check %u\n", num_segments);
	
	uint8_t *pip = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **) &pip, num_segments * sizeof(uint8_t)));
	CUDA_SAFE_CALL(cudaMemset(pip, 0, num_segments * sizeof(uint8_t)));


	grid_size = (num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;

	kernel_filter_segment_contain<<<grid_size, block_size>>>(
		(Segment *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, 
		gctx->d_idealoffset, gctx->d_info, gctx->d_status, 
		gctx->d_vertices, gctx->d_bufferinput_size, pip, 
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

	// printf("---------------------------------------------------------------------------------------------------\n");
	
    // PrintBuffer((Segment *)gctx->d_BufferInput, num_segments);
	
	// // PrintBuffer((bool *) pip, num_segments);
	
	// uint8_t* h_Buffer = new uint8_t[num_segments];
    // CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	// int _sum = 0;
    // for (int i = 0; i < num_segments; i++) {
	// 	// if(h_Buffer[i] == 2) _sum ++;
	// 	std::cout << (int)h_Buffer[i] << " ";
	// 	if ((i + 1) % 5 == 0) printf("\n");
    // }
    // printf("\n");

	// // printf("sum = %d\n", _sum);

	// printf("---------------------------------------------------------------------------------------------------\n");

	
	gctx->segments = new Segment[num_segments];
	gctx->num_segments = num_segments;
	CUDA_SAFE_CALL(cudaMemcpy(gctx->segments, (Segment *)gctx->d_BufferInput, num_segments * sizeof(Segment), cudaMemcpyDeviceToHost));
    
    gctx->pip = new uint8_t[num_segments];
	CUDA_SAFE_CALL(cudaMemcpy(gctx->pip, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_inters_per_pair));
	CUDA_SAFE_CALL(cudaFree(pip));
    return;
}
