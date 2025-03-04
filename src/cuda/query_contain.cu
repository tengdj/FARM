#include "geometry.cuh"

__global__ void kernel_filter_contain(IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint size, uint8_t *resultmap, PixMapping *ptpixpairs, uint *d_pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		IdealPair &pair = pairs[x];
		IdealOffset &source = idealoffset[pair.source];
		Point &p = points[pair.target];

		box &s_mbr = info[pair.source].mbr;
		const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y;
		const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;

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
	
		resultmap[x] = is_in ? 2 : (is_out ? 0 : resultmap[x]);
	}
}

__global__ void kernel_refinement_contain(IdealPair *pairs, PixMapping *ptpixpairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint32_t *es_offset, EdgeSeq *edge_sequences, Point *vertices, uint32_t *gridline_offset, double *gridline_nodes, uint *size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int pair_id = ptpixpairs[x].pair_id;
		int target = ptpixpairs[x].pix_id;

		IdealPair &pair = pairs[pair_id];
		IdealOffset &source = idealoffset[pair.source];
		Point &p = points[pair.target];

		box &s_mbr = info[pair.source].mbr;
		const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y;
		const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;

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
			resultmap[pair_id] = 1;
		}
	}
}

uint cuda_contain(query_context *gctx)
{
	// resultmap status: 0(undecided / not contain), 1(contain), 2(contain)

	CudaTimer timer;

	uint point_polygon_pairs_size = gctx->point_polygon_pairs.size();
	uint batch_size = gctx->batch_size;
	int found = 0;

	IdealPair *h_pairs = new IdealPair[point_polygon_pairs_size];

	for (int i = 0; i < point_polygon_pairs_size; i++)
	{
		h_pairs[i].target = gctx->point_polygon_pairs[i].first;
		h_pairs[i].source = gctx->point_polygon_pairs[i].second;
	}

	IdealPair *d_pairs = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, point_polygon_pairs_size * sizeof(IdealPair)));
	CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, point_polygon_pairs_size * sizeof(IdealPair), cudaMemcpyHostToDevice));

	uint h_bufferinput_size;

	for (int i = 0; i < point_polygon_pairs_size; i += batch_size)
	{

		int start = i, end = min(i + batch_size, point_polygon_pairs_size);
		int size = end - start;

		CUDA_SAFE_CALL(cudaMemset(gctx->d_resultmap, 0, size * sizeof(uint8_t)));
		CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));

		timer.startTimer();

		/*1. Raster Model Filtering*/

		int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 block_size(BLOCK_SIZE, 1, 1);
		dim3 grid_size(grid_size_x, 1, 1);

		kernel_filter_contain<<<grid_size, block_size>>>(d_pairs + start, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_status, size, gctx->d_resultmap, (PixMapping *)gctx->d_BufferInput, gctx->d_bufferinput_size);
		cudaDeviceSynchronize();
		check_execution("kernel_filter_contain");

		CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
		printf("h_buffer_size = %u\n", h_bufferinput_size);

		if (h_bufferinput_size == 0)
		{
			if (i + batch_size >= point_polygon_pairs_size)
			{
				CUDA_SAFE_CALL(cudaMemcpy(gctx->h_resultmap, gctx->d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
				for (int idx = 0; idx < size; ++idx)
				{
					if (gctx->h_resultmap[idx] >= 1)
					{
						found++;
					}
				}
			}
			continue;
		}

		/*2. Refinement Step*/

		grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		grid_size.x = grid_size_x;

		kernel_refinement_contain<<<grid_size, block_size>>>(d_pairs + start, (PixMapping *)gctx->d_BufferInput, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferinput_size, gctx->d_resultmap);
		cudaDeviceSynchronize();
		check_execution("kernel_refinement_contain");

		timer.stopTimer();
		printf("query time: %f ms\n", timer.getElapsedTime());

		CUDA_SAFE_CALL(cudaMemcpy(gctx->h_resultmap, gctx->d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		for (int idx = 0; idx < size; ++idx)
		{
			if (gctx->h_resultmap[idx] >= 1)
			{
				found++;
			}
		}
	}

	CUDA_SAFE_CALL(cudaFree(d_pairs));
	delete[] h_pairs;

	return found;
}