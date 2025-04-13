#include "geometry.cuh"

__global__ void kernel_filter_contain(pair<uint32_t,uint32_t>* pairs, Point *points, 
	    							 IdealOffset *idealoffset, RasterInfo *info, 
									 uint8_t *status, uint size, uint *result, 
									 PixMapping *ptpixpairs, uint *pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;

	const pair<uint32_t, uint32_t> pair = pairs[x];
	const uint32_t src_idx = pair.first;
	const uint32_t pt_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const Point p = points[pt_idx];

	const box s_mbr = info[src_idx].mbr;
	const double s_step_x = info[src_idx].step_x;
	const double s_step_y = info[src_idx].step_y;
	const int s_dimx = info[src_idx].dimx;
	const int s_dimy = info[src_idx].dimy;

	const int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
	const int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
	const int target = gpu_get_id(xoff, yoff, s_dimx);

	const PartitionStatus st = gpu_show_status(status, source.status_start, target);

	const bool is_in = (st == IN);
	const bool is_out = (st == OUT);
	const bool is_border = !(is_in || is_out);
    
	if (is_border) {
		uint idx = atomicAdd(pp_size, 1U);
		ptpixpairs[idx].pair_id = x;
		ptpixpairs[idx].pix_id = target;
	}

	atomicAdd(result, (uint)(is_in));
}

__global__ void kernel_refinement_contain(pair<uint32_t, uint32_t> *pairs,
											PixMapping *ptpixpairs, Point *points,
											IdealOffset *idealoffset, RasterInfo *info,
											uint32_t *es_offset, EdgeSeq *edge_sequences,
											Point *vertices, uint32_t *gridline_offset,
											double *gridline_nodes, uint *size, uint *result)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size)
		return;

	const int pair_id = ptpixpairs[x].pair_id;
	const int target = ptpixpairs[x].pix_id;
	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
	const uint32_t src_idx = pair.first;
	const uint32_t pt_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const Point p = points[pt_idx];
	const box s_mbr = info[src_idx].mbr;
	const double s_step_x = info[src_idx].step_x;
	const double s_step_y = info[src_idx].step_y;
	const int s_dimx = info[src_idx].dimx;
	const int s_dimy = info[src_idx].dimy;

	bool ret = false;

	const int xoff = gpu_get_x(target, s_dimx);
	const int yoff = gpu_get_y(target, s_dimx, s_dimy);
	const box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

	const uint32_t offset_start = source.offset_start;
	const uint32_t es_start = (es_offset + offset_start)[target];
	const uint32_t es_end = (es_offset + offset_start)[target + 1];
	const int s_num_sequence = es_end - es_start;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		const EdgeSeq r = (edge_sequences + source.edge_sequences_start)[es_start + i];
		const uint32_t vertices_start = source.vertices_start;

		for (int j = 0; j < r.length; j++)
		{
			const Point v1 = (vertices + vertices_start)[r.start + j];
			const Point v2 = (vertices + vertices_start)[r.start + j + 1];

			if ((v1.y >= p.y) != (v2.y >= p.y))
			{
				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (dy != 0.0)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if (p.x <= int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
		}
	}

	int nc = 0;
	const uint32_t gridline_start = source.gridline_offset_start;
	const uint32_t i_start = (gridline_offset + gridline_start)[xoff + 1];
	const uint32_t i_end = (gridline_offset + gridline_start)[xoff + 2];

	nc = binary_search_count((gridline_nodes + source.gridline_nodes_start), i_start, i_end, p.y);

	ret ^= (nc & 1);

	atomicAdd(result, (uint)(ret));
}

__global__ void filter_check_contain(pair<uint32_t,uint32_t>* pairs, uint source_size,
	    							 IdealOffset *idealoffset, RasterInfo *info, 
									 uint8_t *status, Point *vertices, uint size, uint8_t *flags, 
									 PixMapping *ptpixpairs, uint *pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;
	pairs[x].second += source_size;

	const pair<uint32_t, uint32_t> pair = pairs[x];
	const uint32_t src_idx = pair.first;
	const uint32_t tar_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const IdealOffset target = idealoffset[tar_idx];
	const Point p = (vertices + target.vertices_start)[0];

	const box s_mbr = info[src_idx].mbr;
	const double s_step_x = info[src_idx].step_x;
	const double s_step_y = info[src_idx].step_y;
	const int s_dimx = info[src_idx].dimx;
	const int s_dimy = info[src_idx].dimy;

	const int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
	const int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
	const int pix_id = gpu_get_id(xoff, yoff, s_dimx);

	const PartitionStatus st = gpu_show_status(status, source.status_start, pix_id);

	const bool is_in = (st == IN);
	const bool is_out = (st == OUT);
	const bool is_border = !(is_in || is_out);
    
	if (is_border) {
		uint idx = atomicAdd(pp_size, 1U);
		ptpixpairs[idx].pair_id = x;
		ptpixpairs[idx].pix_id = pix_id;
	}

	flags[x] = (uint8_t)(is_in);
}

__global__ void refine_check_contain(pair<uint32_t, uint32_t> *pairs, PixMapping *ptpixpairs,
											IdealOffset *idealoffset, RasterInfo *info,
											uint32_t *es_offset, EdgeSeq *edge_sequences,
											Point *vertices, uint32_t *gridline_offset,
											double *gridline_nodes, uint *size, uint8_t *flags)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;

	const int pair_id = ptpixpairs[x].pair_id;
	const int pix_id = ptpixpairs[x].pix_id;
	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
	const uint32_t src_idx = pair.first;
	const uint32_t tar_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const IdealOffset target = idealoffset[tar_idx];
	const Point p = (vertices + target.vertices_start)[0];
	const box s_mbr = info[src_idx].mbr;
	const double s_step_x = info[src_idx].step_x;
	const double s_step_y = info[src_idx].step_y;
	const int s_dimx = info[src_idx].dimx;
	const int s_dimy = info[src_idx].dimy;

	bool ret = false;

	const int xoff = gpu_get_x(pix_id, s_dimx);
	const int yoff = gpu_get_y(pix_id, s_dimx, s_dimy);
	const box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

	const uint32_t offset_start = source.offset_start;
	const uint32_t es_start = (es_offset + offset_start)[pix_id];
	const uint32_t es_end = (es_offset + offset_start)[pix_id + 1];
	const int s_num_sequence = es_end - es_start;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		const EdgeSeq r = (edge_sequences + source.edge_sequences_start)[es_start + i];
		const uint32_t vertices_start = source.vertices_start;

		for (int j = 0; j < r.length; j++)
		{
			const Point v1 = (vertices + vertices_start)[r.start + j];
			const Point v2 = (vertices + vertices_start)[r.start + j + 1];

			if ((v1.y >= p.y) != (v2.y >= p.y))
			{
				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (dy != 0.0)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if (p.x <= int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
		}
	}

	int nc = 0;
	const uint32_t gridline_start = source.gridline_offset_start;
	const uint32_t i_start = (gridline_offset + gridline_start)[xoff + 1];
	const uint32_t i_end = (gridline_offset + gridline_start)[xoff + 2];

	nc = binary_search_count((gridline_nodes + source.gridline_nodes_start), i_start, i_end, p.y);

	ret ^= (nc & 1);

	flags[pair_id] = (uint8_t)(ret);
}

void cuda_contain(query_context *gctx, bool polygon)
{
    uint h_bufferinput_size = 0;

    CUDA_SAFE_CALL(cudaMemset(gctx->d_result, 0, sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));

	// Filtering Step
    const int block_size = BLOCK_SIZE;
    const int grid_size = (gctx->num_pairs + block_size - 1) / block_size;
	if(!polygon){
		kernel_filter_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs, gctx->d_points, gctx->d_idealoffset,
			gctx->d_info, gctx->d_status, gctx->num_pairs, 
			gctx->d_result, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size
		);
	}else{
		filter_check_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs, gctx->source_ideals.size(), gctx->d_idealoffset,
			gctx->d_info, gctx->d_status, gctx->d_vertices, gctx->num_pairs, 
			gctx->d_flags, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size
		);
	}
	cudaDeviceSynchronize();
    check_execution("kernel_filter_contain");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
	printf("h_bufferinput_size = %d\n", h_bufferinput_size);
    
    if (h_bufferinput_size == 0) {
        CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
        return;
    }

	// Refinement Step

    const int refine_grid_size = (h_bufferinput_size + block_size - 1) / block_size;
    if(!polygon){
		kernel_refinement_contain<<<refine_grid_size, block_size>>>(
			gctx->d_candidate_pairs, (PixMapping *)gctx->d_BufferInput,
			gctx->d_points, gctx->d_idealoffset, gctx->d_info,
			gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices,
			gctx->d_gridline_offset, gctx->d_gridline_nodes,
			gctx->d_bufferinput_size, gctx->d_result
		);
	}else{
		refine_check_contain<<<refine_grid_size, block_size>>>(
			gctx->d_candidate_pairs, (PixMapping *)gctx->d_BufferInput,
			gctx->d_idealoffset, gctx->d_info,
			gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices,
			gctx->d_gridline_offset, gctx->d_gridline_nodes,
			gctx->d_bufferinput_size, gctx->d_flags
		);
	}
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_contain");

	if(!polygon) CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));

	// uint8_t* h_Buffer = new uint8_t[gctx->num_pairs];
    // CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, gctx->d_flags, gctx->num_pairs * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	// int _sum = 0;
    // for (int i = 0; i < gctx->num_pairs; i++) {
	// 	if(h_Buffer[i] == 1) _sum ++;
	// 	std::cout << (int)h_Buffer[i] << " ";
	// 	if ((i + 1) % 5 == 0) printf("\n");
    // }
    // printf("\n");

	// printf("sum = %d\n", _sum);
    

}

// #include "geometry.cuh"

// __global__ void kernel_filter_contain(pair<uint32_t,uint32_t>* pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint size, uint *result, PixMapping *ptpixpairs, uint *d_pp_size)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (x < size)
// 	{
// 		pair<uint32_t, uint32_t> &pair = pairs[x];
// 		IdealOffset &source = idealoffset[pair.first];
// 		Point &p = points[pair.second];

// 		box &s_mbr = info[pair.first].mbr;
// 		const double &s_step_x = info[pair.first].step_x, &s_step_y = info[pair.first].step_y;
// 		const int &s_dimx = info[pair.first].dimx, &s_dimy = info[pair.first].dimy;

// 		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
// 		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
// 		int target = gpu_get_id(xoff, yoff, s_dimx);

// 		PartitionStatus st = gpu_show_status(status, source.status_start, target);

// 		bool is_in = (st == IN);
// 		bool is_out = (st == OUT);
// 		bool is_border = !(is_in || is_out);
	
// 		uint idx = 0;
// 		if (is_border) {
// 			idx = atomicAdd(d_pp_size, 1U);
// 			ptpixpairs[idx].pair_id = x;
// 			ptpixpairs[idx].pix_id = target;
// 		}
//         if(is_in) atomicAdd(result, 1);
// 		return;
// 	}
// }

// __global__ void kernel_refinement_contain(pair<uint32_t, uint32_t>* pairs, PixMapping *ptpixpairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint32_t *es_offset, EdgeSeq *edge_sequences, Point *vertices, uint32_t *gridline_offset, double *gridline_nodes, uint *size, uint *result)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (x < *size)
// 	{
// 		int pair_id = ptpixpairs[x].pair_id;
// 		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

// 		pair<uint32_t, uint32_t> &pair = pairs[pair_id];
// 		IdealOffset &source = idealoffset[pair.first];
// 		Point &p = points[pair.second];

// 		box &s_mbr = info[pair.first].mbr;
// 		const double &s_step_x = info[pair.first].step_x, &s_step_y = info[pair.first].step_y;
// 		const int &s_dimx = info[pair.first].dimx, &s_dimy = info[pair.first].dimy;

// 		bool ret = false;

// 		int xoff = gpu_get_x(target, s_dimx);
// 		int yoff = gpu_get_y(target, s_dimx, s_dimy);
// 		box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

// 		int s_num_sequence = (es_offset + source.offset_start)[target + 1] - (es_offset + source.offset_start)[target];

// 		for (int i = 0; i < s_num_sequence; ++ i)
// 		{
// 			EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[target] + i];
// 			for (int j = 0; j < r.length; j++)
// 			{
// 				if ((vertices + source.vertices_start)[r.start + j].y >= p.y != (vertices + source.vertices_start)[r.start + j + 1].y >= p.y)
// 				{
// 					double int_x =
// 						((vertices + source.vertices_start)[r.start + j + 1]
// 							 .x -
// 						 (vertices + source.vertices_start)[r.start + j].x) *
// 							(p.y -
// 							 (vertices + source.vertices_start)[r.start + j]
// 								 .y) /
// 							((vertices + source.vertices_start)[r.start + j + 1]
// 								 .y -
// 							 (vertices + source.vertices_start)[r.start + j]
// 								 .y) +
// 						(vertices + source.vertices_start)[r.start + j].x;
// 					if (p.x <= int_x && int_x <= bx.high[0])
// 					{
// 						ret = !ret;
// 					}
// 				}
// 			}
// 		}
// 		int nc = 0;

// 		uint32_t i = (gridline_offset + source.gridline_offset_start)[xoff + 1];
// 		uint32_t j = (gridline_offset + source.gridline_offset_start)[xoff + 2];
		
// 		while (i < j && (gridline_nodes + source.gridline_nodes_start)[i] <= p.y)
// 		{
// 			nc++;
// 			i++;
// 		}
// 		if (nc % 2 == 1)
// 		{
// 			ret = !ret;
// 		}
		
// 		if (ret)
// 		{
// 			atomicAdd(result, 1);
// 		}
// 	}
// }

// void cuda_contain(query_context *gctx)
// {
//     uint h_bufferinput_size;

// #ifdef DEBUG
// 	CudaTimer timer;
//     timer.startTimer();
// #endif
//     /*1. Raster Model Filtering*/

//     int grid_size_x = (gctx->num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 block_size(BLOCK_SIZE, 1, 1);
//     dim3 grid_size(grid_size_x, 1, 1);

//     kernel_filter_contain<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_status, gctx->num_pairs, gctx->d_result, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size);
//     cudaDeviceSynchronize();
//     check_execution("kernel_filter_contain");

//     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
//     if (h_bufferinput_size == 0) {
//         CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
//         return;
//     }
// #ifdef DEBUG
//     printf("h_buffer_size = %u\n", h_bufferinput_size);
// #endif
    
//     /*2. Refinement Step*/

//     grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     grid_size.x = grid_size_x;

//     kernel_refinement_contain<<<grid_size, block_size>>>(gctx->d_candidate_pairs, (PixMapping *)gctx->d_BufferInput, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferinput_size, gctx->d_result);
//     cudaDeviceSynchronize();
//     check_execution("kernel_refinement_contain");

// #ifdef DEBUG
//     timer.stopTimer();
//     printf("query time: %f ms\n", timer.getElapsedTime());
// #endif
//     CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
// 	return;
// }