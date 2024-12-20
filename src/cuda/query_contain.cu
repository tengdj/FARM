#include "geometry.cuh"

struct Batch
{
	IdealOffset source;
	Point target;
};

__global__ void kernel_filter_contain(Batch *d_pairs, RasterInfo *d_info, uint8_t *d_status, uint size, uint8_t *resultmap, PixMapping *d_ptpixpairs, uint *d_pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		Batch &pair = d_pairs[x];
		IdealOffset &source = pair.source;
		Point &p = pair.target;

		box &s_mbr = d_info[source.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
		int target = gpu_get_id(xoff, yoff, s_dimx);

		// printf("x: %d, y: %d, id = %d, status: %d\n", xoff, yoff, target, gpu_show_status(d_status, source.status_start, target));

		if (gpu_show_status(d_status, source.status_start, target) == IN)
		{
			resultmap[x] = 2;
		}
		else if (gpu_show_status(d_status, source.status_start, target) == OUT)
		{
			resultmap[x] = 0;
		}
		else
		{
			int idx = atomicAdd(d_pp_size, 1U);
			d_ptpixpairs[idx].pair_id = x;
			d_ptpixpairs[idx].pix_id = target;
		}
	}
}

__global__ void kernel_refinement_contain(Batch *d_pairs, PixMapping *d_ptpixpairs, RasterInfo *d_info, uint16_t *d_offset, EdgeSeq *d_edge_sequences, Point *d_vertices, uint16_t *d_gridline_offset, double *d_gridline_nodes, uint *size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int pair_id = d_ptpixpairs[x].pair_id;
		int target = d_ptpixpairs[x].pix_id;

		Batch &pair = d_pairs[pair_id];
		IdealOffset &source = pair.source;
		Point &p = pair.target;

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

void process_chunk(int start, int end, int base, query_context *gctx, Batch *h_pairs) {
	for (int idx = start; idx < end; idx ++) {
        Point *target = gctx->point_polygon_pairs[idx].first;
        Ideal *source = gctx->point_polygon_pairs[idx].second;
		
		// if(idx - base >= gctx->point_polygon_pairs.size() / 5){
		// 	assert(idx - base < gctx->point_polygon_pairs.size() / 5);
		// }else{
	        h_pairs[idx - base] = {*source->idealoffset, *target};
		// }
	}
}

uint cuda_contain(query_context *gctx)
{
	CudaTimer timer, total, duration;
	total.startTimer();

	uint point_polygon_pairs_size = gctx->point_polygon_pairs.size();
	uint batch_size = point_polygon_pairs_size / 5;
	int found = 0;

	printf("size = %d\n", point_polygon_pairs_size);

	timer.startTimer();
	Batch *h_pairs = new Batch[point_polygon_pairs_size];
	timer.stopTimer();
	printf("cpu time: %f ms\n", timer.getElapsedTime());

	timer.startTimer();
	Batch *d_pairs = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, batch_size * sizeof(Batch)));

	timer.stopTimer();
	printf("cudaMalloc time: %f ms\n", timer.getElapsedTime());

	// resultmap status: 0(undecided), 1(contain), 2(not contain)
	uint8_t *d_resultmap = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, batch_size * sizeof(uint8_t)));
	uint8_t *h_resultmap = new uint8_t[batch_size];

	char *d_Buffer = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_Buffer, 10UL * 1024 * 1024 * 1024));
    uint *d_buffer_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_buffer_size, sizeof(uint)));
    uint h_buffer_size;

	for (int i = 0; i < point_polygon_pairs_size; i += batch_size)
	{
		
		duration.startTimer();
		int start = i, end = min(i + batch_size, point_polygon_pairs_size);
        int size = end - start;

		printf("size = %d\n", size);

		timer.startTimer();

		int num_threads = gctx->num_threads; 
		int chunk_size = (size + num_threads - 1) / num_threads; 

		std::vector<std::thread> threads;

		for (int j = 0; j < num_threads; ++ j) {
			int chunk_start = start + j * chunk_size;
			// int chunk_end = (j == (num_threads - 1)) ? end : chunk_start + chunk_size;
			int chunk_end = min(chunk_start + chunk_size, end);

			// printf("thread %d\n", i);
			// printf("%d %d %d\n", chunk_start, chunk_end, start);
			threads.emplace_back(process_chunk, chunk_start, chunk_end, start, gctx, h_pairs);
		}

		for (auto& thread : threads) {
			thread.join();
		}
		
		// int idx = 0;
		// for(int j = start; j < end; j ++)
		// {
		// 	Point *target = gctx->point_polygon_pairs[j].first;
		// 	Ideal *source = gctx->point_polygon_pairs[j].second;
		// 	h_pairs[idx ++] = {*source->idealoffset, *target};
		// }

		timer.stopTimer();
		printf("cpu time: %f ms\n", timer.getElapsedTime());

		timer.startTimer();
		CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(Batch), cudaMemcpyHostToDevice));
		timer.stopTimer();
		printf("cudaMemcpy time: %f ms\n", timer.getElapsedTime());

		CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(uint8_t)));
		CUDA_SAFE_CALL(cudaMemset(d_buffer_size, 0, sizeof(uint)));

		/*1. Raster Model Filtering*/

		int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 block_size(BLOCK_SIZE, 1, 1);
		dim3 grid_size(grid_size_x, 1, 1);

		timer.startTimer();
		kernel_filter_contain<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, size, d_resultmap, (PixMapping *)d_Buffer, d_buffer_size);
		cudaDeviceSynchronize();
		check_execution("kernel_filter_contain");
		timer.stopTimer();
		printf("kernel_filter time: %f ms\n", timer.getElapsedTime());

		CUDA_SAFE_CALL(cudaMemcpy(&h_buffer_size, d_buffer_size, sizeof(uint), cudaMemcpyDeviceToHost));
		printf("h_buffer_size = %u\n", h_buffer_size);

		if(h_buffer_size == 0) {
			if(i + batch_size >= point_polygon_pairs_size){
				CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
				for (int idx = 0; idx < size; ++ idx)
				{
					if (h_resultmap[idx] >= 1){
						found ++;
					}
					
				}

				duration.stopTimer();
				printf("round time: %f ms\n", duration.getElapsedTime());
			}
			continue;
		}

		/*2. Refinement Step*/

		grid_size_x = (h_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		grid_size.x = grid_size_x;

		timer.startTimer();
		kernel_refinement_contain<<<grid_size, block_size>>>(d_pairs, (PixMapping *)d_Buffer, gctx->d_info, gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, gctx->d_gridline_offset, gctx->d_gridline_nodes, d_buffer_size, d_resultmap);
		cudaDeviceSynchronize();
		check_execution("kernel_refinement_contain");
		timer.stopTimer();
		printf("kernel_refinement time: %f ms\n", timer.getElapsedTime());

		CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		for (int idx = 0; idx < size; ++ idx)
		{
			if (h_resultmap[idx] >= 1){
				found ++;
			}
				
		}

		duration.stopTimer();
		printf("round time: %f ms\n", duration.getElapsedTime());
	}

	total.stopTimer();
	printf("total query: %f ms\n", total.getElapsedTime());
	
	CUDA_SAFE_CALL(cudaFree(d_pairs));
	CUDA_SAFE_CALL(cudaFree(d_resultmap));
	CUDA_SAFE_CALL(cudaFree(d_buffer_size));
	CUDA_SAFE_CALL(cudaFree(d_Buffer));
	delete []h_pairs;
	delete []h_resultmap;
	return found;
}