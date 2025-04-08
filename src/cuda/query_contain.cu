#include "geometry.cuh"

__global__ void kernel_filter_contain(pair<uint32_t,uint32_t>* pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint size, uint *result, PixMapping *ptpixpairs, uint *d_pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		pair<uint32_t, uint32_t> &pair = pairs[x];
		IdealOffset &source = idealoffset[pair.first];
		Point &p = points[pair.second];

		box &s_mbr = info[pair.first].mbr;
		const double &s_step_x = info[pair.first].step_x, &s_step_y = info[pair.first].step_y;
		const int &s_dimx = info[pair.first].dimx, &s_dimy = info[pair.first].dimy;

		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
		int target = gpu_get_id(xoff, yoff, s_dimx);

		PartitionStatus st = gpu_show_status(status, source.status_start, target);

		bool is_in = (st == IN);
		bool is_out = (st == OUT);
		bool is_border = !(is_in || is_out);
	
		uint idx = 0;
		if (is_border) {
			idx = atomicAdd(d_pp_size, 1U);
			ptpixpairs[idx].pair_id = x;
			ptpixpairs[idx].pix_id = target;
		}
        if(is_in) atomicAdd(result, 1);
		return;
	}
}

__global__ void kernel_refinement_contain(pair<uint32_t, uint32_t>* pairs, PixMapping *ptpixpairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint32_t *es_offset, EdgeSeq *edge_sequences, Point *vertices, uint32_t *gridline_offset, double *gridline_nodes, uint *size, uint *result)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int pair_id = ptpixpairs[x].pair_id;
		int target = ptpixpairs[x].pix_id;

		pair<uint32_t, uint32_t> &pair = pairs[pair_id];
		IdealOffset &source = idealoffset[pair.first];
		Point &p = points[pair.second];

		box &s_mbr = info[pair.first].mbr;
		const double &s_step_x = info[pair.first].step_x, &s_step_y = info[pair.first].step_y;
		const int &s_dimx = info[pair.first].dimx, &s_dimy = info[pair.first].dimy;

		bool ret = false;

		int xoff = gpu_get_x(target, s_dimx);
		int yoff = gpu_get_y(target, s_dimx, s_dimy);
		box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

		int s_num_sequence = (es_offset + source.offset_start)[target + 1] - (es_offset + source.offset_start)[target];

		for (int i = 0; i < s_num_sequence; ++ i)
		{
			EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[target] + i];
			for (int j = 0; j < r.length; j++)
			{
				if ((vertices + source.vertices_start)[r.start + j].y >= p.y != (vertices + source.vertices_start)[r.start + j + 1].y >= p.y)
				{
					double int_x =
						((vertices + source.vertices_start)[r.start + j + 1]
							 .x -
						 (vertices + source.vertices_start)[r.start + j].x) *
							(p.y -
							 (vertices + source.vertices_start)[r.start + j]
								 .y) /
							((vertices + source.vertices_start)[r.start + j + 1]
								 .y -
							 (vertices + source.vertices_start)[r.start + j]
								 .y) +
						(vertices + source.vertices_start)[r.start + j].x;
					if (p.x <= int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
		}
		int nc = 0;

		uint32_t i = (gridline_offset + source.gridline_offset_start)[xoff + 1];
		uint32_t j = (gridline_offset + source.gridline_offset_start)[xoff + 2];
		
		while (i < j && (gridline_nodes + source.gridline_nodes_start)[i] <= p.y)
		{
			nc++;
			i++;
		}
		if (nc % 2 == 1)
		{
			ret = !ret;
		}
		
		if (ret)
		{
			atomicAdd(result, 1);
		}
	}
}

void cuda_contain(query_context *gctx)
{
    uint h_bufferinput_size;

#ifdef DEBUG
	CudaTimer timer;
    timer.startTimer();
#endif
    /*1. Raster Model Filtering*/

    int grid_size_x = (gctx->num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, 1, 1);
    dim3 grid_size(grid_size_x, 1, 1);

    kernel_filter_contain<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_status, gctx->num_pairs, gctx->d_result, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_filter_contain");

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    if (h_bufferinput_size == 0) {
        CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
        return;
    }
#ifdef DEBUG
    printf("h_buffer_size = %u\n", h_bufferinput_size);
#endif
    
    /*2. Refinement Step*/

    grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size.x = grid_size_x;

    kernel_refinement_contain<<<grid_size, block_size>>>(gctx->d_candidate_pairs, (PixMapping *)gctx->d_BufferInput, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferinput_size, gctx->d_result);
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_contain");

#ifdef DEBUG
    timer.stopTimer();
    printf("query time: %f ms\n", timer.getElapsedTime());
#endif
    CUDA_SAFE_CALL(cudaMemcpy(&gctx->found, gctx->d_result, sizeof(uint), cudaMemcpyDeviceToHost));
	return;
}