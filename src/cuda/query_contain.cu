#include "geometry.cuh"

__global__ void kernel_filter_contain(pair<uint32_t,uint32_t> *pairs, Point *points, 
	    							 IdealOffset *idealoffset, RasterInfo *info, 
									 uint8_t *status, uint size, uint *result, int8_t *flags, 
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
	flags[x] = (int8_t)(!is_in);
}

__global__ void kernel_refinement_contain(pair<uint32_t, uint32_t> *pairs,
											PixMapping *ptpixpairs, Point *points,
											IdealOffset *idealoffset, RasterInfo *info,
											uint32_t *es_offset, EdgeSeq *edge_sequences,
											Point *vertices, uint32_t *gridline_offset,
											double *gridline_nodes, uint *size, uint *result, int8_t *flags)
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
	flags[pair_id] = (int8_t)(!ret);
}

__global__ void filter_check_contain(pair<uint32_t,uint32_t>* pairs, uint source_size,
	    							 IdealOffset *idealoffset, RasterInfo *info, 
									 uint8_t *status, Point *vertices, uint size, int8_t *flags, 
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

	flags[x] = (int8_t)(is_in);
}

__global__ void refine_check_contain(pair<uint32_t, uint32_t> *pairs, PixMapping *ptpixpairs,
											IdealOffset *idealoffset, RasterInfo *info,
											uint32_t *es_offset, EdgeSeq *edge_sequences,
											Point *vertices, uint32_t *gridline_offset,
											double *gridline_nodes, uint *size, int8_t *flags)
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

	flags[pair_id] = (int8_t)(ret);
}

__global__ void collect_valid_pairs(pair<uint32_t, uint32_t> *pairs, int8_t *flags, uint size, pair<uint32_t, uint32_t> *buffer, uint *buffer_size){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size && flags[x] == 0) return;

	uint idx = atomicAdd(buffer_size, 1U);
	buffer[idx].first = pairs[x].first;
	buffer[idx].second = pairs[x].second;
}

__global__ void kernel_filter_segment_contain(Segment *segments, pair<uint32_t,uint32_t> *pairs,
											  IdealOffset *idealoffset, RasterInfo *info, 
											  uint8_t *status, Point *vertices,  uint *size, bool *flags, 
											  PixMapping *ptpixpairs, uint *pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;

	Segment seg = segments[x];
	const pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
	uint32_t poly_idx = !seg.is_source ? pair.first : pair.second;
	const IdealOffset offset = idealoffset[poly_idx];
	
	Point p;
	if(seg.edge_start == -1) p = (seg.start + seg.end) * 0.5;
	else p = vertices[seg.edge_start];
	
	const box s_mbr = info[poly_idx].mbr;
	const double s_step_x = info[poly_idx].step_x;
	const double s_step_y = info[poly_idx].step_y;
	const int s_dimx = info[poly_idx].dimx;
	const int s_dimy = info[poly_idx].dimy;
	
	const int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
	const int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
	const int target = gpu_get_id(xoff, yoff, s_dimx);

	const PartitionStatus st = gpu_show_status(status, offset.status_start, target);
	
	const bool is_in = (st == IN);
	const bool is_out = (st == OUT);
	const bool is_border = !(is_in || is_out);
    
	if (is_border) {
		uint idx = atomicAdd(pp_size, 1U);
		ptpixpairs[idx].pair_id = x;
		ptpixpairs[idx].pix_id = target;
	}

	flags[x] = is_in;
}

__global__ void kernel_refinement_segment_contain(PixMapping *ptpixpairs, Segment *segments, 
												pair<uint32_t, uint32_t> *pairs,
												IdealOffset *idealoffset, RasterInfo *info,
												uint32_t *es_offset, EdgeSeq *edge_sequences,
												Point *vertices, uint32_t *gridline_offset,
												double *gridline_nodes, uint *size, bool *flags)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size)
		return;

	int seg_id = ptpixpairs[x].pair_id;
	int target = ptpixpairs[x].pix_id;
	
	Segment seg = segments[seg_id];
	const pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
	const uint32_t poly_idx = !seg.is_source ? pair.first : pair.second;
	const IdealOffset offset = idealoffset[poly_idx];
	
	Point p;
	if(seg.edge_start == -1) p = (seg.start + seg.end) * 0.5;
	else p = vertices[seg.edge_start];

	const box s_mbr = info[poly_idx].mbr;
	const double s_step_x = info[poly_idx].step_x;
	const double s_step_y = info[poly_idx].step_y;
	const int s_dimx = info[poly_idx].dimx;
	const int s_dimy = info[poly_idx].dimy;

	bool ret = false;

	const int xoff = gpu_get_x(target, s_dimx);
	const int yoff = gpu_get_y(target, s_dimx, s_dimy);
	const box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

	const uint32_t offset_start = offset.offset_start;
	const uint32_t es_start = (es_offset + offset_start)[target];
	const uint32_t es_end = (es_offset + offset_start)[target + 1];
	const int s_num_sequence = es_end - es_start;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		const EdgeSeq r = (edge_sequences + offset.edge_sequences_start)[es_start + i];
		const uint32_t vertices_start = offset.vertices_start;
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
	const uint32_t gridline_start = offset.gridline_offset_start;
	const uint32_t i_start = (gridline_offset + gridline_start)[xoff + 1];
	const uint32_t i_end = (gridline_offset + gridline_start)[xoff + 2];

	nc = binary_search_count((gridline_nodes + offset.gridline_nodes_start), i_start, i_end, p.y);

	ret ^= (nc & 1);

	flags[seg_id] = ret;
}

void cuda_contain(query_context *gctx, bool polygon)
{
	size_t batch_size = gctx->index_end - gctx->index;
    uint h_bufferinput_size = 0;
	
	// Filtering Step
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;
	if(!polygon){
		kernel_filter_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs + gctx->index, gctx->d_points, gctx->d_idealoffset,
			gctx->d_info, gctx->d_status, batch_size, 
			gctx->d_result, gctx->d_flags, (PixMapping *)gctx->d_BufferInput, 
			gctx->d_bufferinput_size
		);
	}else{
		filter_check_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs + gctx->index, gctx->source_ideals.size(), gctx->d_idealoffset,
			gctx->d_info, gctx->d_status, gctx->d_vertices, batch_size, 
			gctx->d_flags, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size
		);
	}
	cudaDeviceSynchronize();
    check_execution("kernel_filter_contain");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    
    if (h_bufferinput_size == 0) {
        CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
        return;
    }

	// Refinement Step

    grid_size = (h_bufferinput_size + block_size - 1) / block_size;
    if(!polygon){
		kernel_refinement_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs + gctx->index, (PixMapping *)gctx->d_BufferInput,
			gctx->d_points, gctx->d_idealoffset, gctx->d_info,
			gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices,
			gctx->d_gridline_offset, gctx->d_gridline_nodes,
			gctx->d_bufferinput_size, gctx->d_result, gctx->d_flags
		);
	}else{
		refine_check_contain<<<grid_size, block_size>>>(
			gctx->d_candidate_pairs + gctx->index, (PixMapping *)gctx->d_BufferInput,
			gctx->d_idealoffset, gctx->d_info,
			gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices,
			gctx->d_gridline_offset, gctx->d_gridline_nodes,
			gctx->d_bufferinput_size, gctx->d_flags
		);
	}
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_contain");

	if(gctx->query_type == contain) {
		uint h_result;
		CUDA_SAFE_CALL(cudaMemcpy(&h_result, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
		gctx->found += h_result;
	}
	// int8_t* h_Buffer = new int8_t[batch_size];
	// CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, gctx->d_flags, batch_size * sizeof(int8_t), cudaMemcpyDeviceToHost));
	// int _sum = 0;
	// for (int i = 0; i < batch_size; i++) {
	// 	if(h_Buffer[i] == 0) _sum ++;
	// 	// std::cout << (int)h_Buffer[i] << " ";
	// 	// if ((i + 1) % 5 == 0) printf("\n");
	// }
	// printf("\n");

	// printf("sum = %d\n", _sum);

}