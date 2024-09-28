#include "geometry.cuh"

__global__ void kernel_filter_contain(pair<Point, IdealOffset> *d_pairs, RasterInfo *d_info, uint8_t *d_status, uint size, uint8_t *resultmap, PixPair *d_ptpixpairs, uint *d_pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		pair<Point, IdealOffset> &pair = d_pairs[x];
		IdealOffset &source = pair.second;
		Point &p = pair.first;

		box &s_mbr = d_info[source.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
		int target = gpu_get_id(xoff, yoff, s_dimx);

		if (gpu_show_status(d_status, source.status_start, target) == IN)
		{
			resultmap[x] = 1;
		}
		else if (gpu_show_status(d_status, source.status_start, target) == OUT)
		{
			resultmap[x] = 2;
		}
		else
		{
			int idx = atomicAdd(d_pp_size, 1U);
			d_ptpixpairs[idx].source_pixid = x;
			d_ptpixpairs[idx].target_pixid = target;
		}
	}
}

__global__ void kernel_refinement_contain(pair<Point, IdealOffset> *d_pairs, PixPair *d_ptpixpairs, RasterInfo *d_info, uint16_t *d_offset, EdgeSeq *d_edge_sequences, Point *d_vertices, uint16_t *d_gridline_offset, double *d_gridline_nodes, uint *size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int pair_id = d_ptpixpairs[x].source_pixid;
		int target = d_ptpixpairs[x].target_pixid;

		pair<Point, IdealOffset> &pair = d_pairs[pair_id];
		IdealOffset &source = pair.second;
		Point &p = pair.first;

		box &s_mbr = d_info[source.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

		bool ret = false;

		int xoff = gpu_get_x(target, s_dimx);
		int yoff = gpu_get_y(target, s_dimx, s_dimy);
		box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

		int s_num_sequence = (d_offset + source.offset_start)[target + 1] - (d_offset + source.offset_start)[target];

		for (int i = 0; i < s_num_sequence; ++i)
		{
			EdgeSeq r = (d_edge_sequences + source.edge_sequences_start)[(d_offset + source.offset_start)[target] + i];
			for (int j = 0; j < r.length; j++)
			{
				if ((d_vertices + source.vertices_start)[r.start + j].y >= p.y != (d_vertices + source.vertices_start)[r.start + j + 1].y >= p.y)
				{
					double int_x =
						((d_vertices + source.vertices_start)[r.start + j + 1]
							 .x -
						 (d_vertices + source.vertices_start)[r.start + j].x) *
							(p.y -
							 (d_vertices + source.vertices_start)[r.start + j]
								 .y) /
							((d_vertices +
							  source.vertices_start)[r.start + j + 1]
								 .y -
							 (d_vertices + source.vertices_start)[r.start + j]
								 .y) +
						(d_vertices + source.vertices_start)[r.start + j].x;
					if (p.x <= int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
		}
		int nc = 0;
		uint16_t i = (d_gridline_offset + source.gridline_offset_start)[xoff + 1], j;
		if (xoff + 1 < s_dimx)
			j = (d_gridline_offset + source.gridline_offset_start)[xoff + 2];
		else
			j = source.gridline_offset_end - source.gridline_offset_start + 1;
		while (i < j && (d_gridline_nodes + source.gridline_nodes_start)[i] <= p.y)
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

	CudaTimer timer;

	uint size = gctx->point_polygon_pairs.size();

	printf("size = %d\n", size);

	pair<Point, IdealOffset> *h_pairs = nullptr;
	pair<Point, IdealOffset> *d_pairs = nullptr;

	h_pairs = new pair<Point, IdealOffset>[size];

	for (int i = 0; i < size; ++i)
	{
		Point *target = gctx->point_polygon_pairs[i].first;
		Ideal *source = gctx->point_polygon_pairs[i].second;
		h_pairs[i] = {*target, *source->idealoffset};
	}

	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, size * sizeof(pair<Point, IdealOffset>)));
	CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(pair<Point, IdealOffset>), cudaMemcpyHostToDevice));

	// resultmap status: 0(undecided), 1(contain), 2(not contain)
	uint8_t *d_resultmap = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, size * sizeof(uint8_t)));
	CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(uint8_t)));

	PixPair *h_ptpixpairs = new PixPair[1024 * 1024];
	PixPair *d_ptpixpairs = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_ptpixpairs, 1024 * 1024 * sizeof(PixPair)));

	uint *d_pp_size = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pp_size, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(d_pp_size, 0, sizeof(uint)));

	/*1. Raster Model Filtering*/

	int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 block_size(BLOCK_SIZE, 1, 1);
	dim3 grid_size(grid_size_x, 1, 1);

	timer.startTimer();

	kernel_filter_contain<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, size, d_resultmap, d_ptpixpairs, d_pp_size);
	cudaDeviceSynchronize();
	check_execution();

	timer.stopTimer();
	printf("kernel_filter time: %f ms\n", timer.getElapsedTime());

	uint h_pp_size;
	CUDA_SAFE_CALL(cudaMemcpy(&h_pp_size, d_pp_size, sizeof(uint), cudaMemcpyDeviceToHost));

	/*Refinement Step*/

	grid_size_x = (h_pp_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	grid_size.x = grid_size_x;

	timer.startTimer();

	kernel_refinement_contain<<<grid_size, block_size>>>(d_pairs, d_ptpixpairs, gctx->d_info, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, d_pp_size, d_resultmap);
	cudaDeviceSynchronize();
	check_execution();

	timer.stopTimer();
	printf("kernel_refinement time: %f ms\n", timer.getElapsedTime());

	uint8_t *h_resultmap = new uint8_t[size];
	CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	int found = 0;
	for (int i = 0; i < size; ++i)
	{
		if (h_resultmap[i] == 1)
			found++;
	}

	return found;
}