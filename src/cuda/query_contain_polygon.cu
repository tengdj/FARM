#include "geometry.cuh"
#include "Ideal.h"

struct Batch
{
    IdealOffset source;
	IdealOffset target;
};

struct Task
{
	uint s_start = 0;
	uint t_start = 0;
	uint s_length = 0;
	uint t_length = 0;
	int pair_id = 0;
};

__global__ void kernel_filter_contain_polygon(Batch *d_pairs, RasterInfo *d_info, uint8_t *d_status, uint size, uint8_t *resultmap, PixPair *d_pixpairs, uint *pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		Batch &temp_pair = d_pairs[x];
		IdealOffset &source = temp_pair.source;
		IdealOffset &target = temp_pair.target;

		const box &s_mbr = d_info[source.info_start].mbr, &t_mbr = d_info[target.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;
		const double &t_step_x = d_info[target.info_start].step_x, &t_step_y = d_info[target.info_start].step_y;
		const int &t_dimx = d_info[target.info_start].dimx, &t_dimy = d_info[target.info_start].dimy;

		uint itn = 0, etn = 0;
		for (int i = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx); i <= gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx); i++)
		{
			for (int j = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy); j <= gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy); j++)
			{
				int p = gpu_get_id(i, j, s_dimx);
				if (gpu_show_status(d_status, source.status_start, p) == IN)
					itn++;
				else if (gpu_show_status(d_status, source.status_start, p) == OUT)
					etn++;

				box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

				for (int _i = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx); _i <= gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx); _i++)
				{
					for (int _j = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy); _j <= gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy); _j++)
					{
						int p2 = gpu_get_id(_i, _j, t_dimx);
						if (gpu_show_status(d_status, source.status_start, p) == OUT && gpu_show_status(d_status, target.status_start, p2) == IN)
						{
							resultmap[x] = 2;
							return;
						}
						if (gpu_show_status(d_status, source.status_start, p) == BORDER && gpu_show_status(d_status, target.status_start, p2) == BORDER)
						{
							int idx = atomicAdd(pp_size, 1U);
							d_pixpairs[idx].source_pixid = p;
							d_pixpairs[idx].target_pixid = p2;
							d_pixpairs[idx].pair_id = x;
						}
					}
				}
			}
		}
		if (itn == (gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx) - gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx) + 1) * (gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy) - gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy) + 1))
		{
			resultmap[x] = 1;
			return;
		}
		if (etn == (gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx) - gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx) + 1) * (gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy) - gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy) + 1))
		{
			resultmap[x] = 2;
			return;
		}
	}
}

__global__ void kernel_unroll_contain_polygon(PixPair *d_pixpairs, Batch *d_pairs, uint8_t *d_status, uint32_t *d_offset, EdgeSeq *d_edge_sequences, uint *size, Task *batches, uint *batch_size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int p = d_pixpairs[x].source_pixid;
		int p2 = d_pixpairs[x].target_pixid;
		int pair_id = d_pixpairs[x].pair_id;
		if (resultmap[pair_id] != 0)
			return;

		Batch temp_pair = d_pairs[pair_id];
		IdealOffset source = temp_pair.source;
		IdealOffset target = temp_pair.target;

		if (gpu_show_status(d_status, source.status_start, p) == BORDER && gpu_show_status(d_status, target.status_start, p2) == BORDER)
		{

			uint s_offset_start = source.offset_start, t_offset_start = target.offset_start;
			uint s_edge_sequences_start = source.edge_sequences_start, t_edge_sequences_start = target.edge_sequences_start;
			int s_num_sequence = (d_offset + s_offset_start)[p + 1] - (d_offset + s_offset_start)[p];
			int t_num_sequence = (d_offset + t_offset_start)[p2 + 1] - (d_offset + t_offset_start)[p2];
			uint s_vertices_start = source.vertices_start, t_vertices_start = target.vertices_start;

			for (int i = 0; i < s_num_sequence; ++i)
			{
				EdgeSeq r = (d_edge_sequences + s_edge_sequences_start)[(d_offset + s_offset_start)[p] + i];
				for (int j = 0; j < t_num_sequence; ++j)
				{
					EdgeSeq r2 = (d_edge_sequences + t_edge_sequences_start)[(d_offset + t_offset_start)[p2] + j];
					int max_size = 32;
					for (uint s = 0; s < r.length; s += max_size)
					{
						uint end_s = min(s + max_size, r.length);
						for (uint t = 0; t < r2.length; t += max_size)
						{
							uint end_t = min(t + max_size, r2.length);
							uint idx = atomicAdd(batch_size, 1U);
							batches[idx].s_start = s_vertices_start + r.start + s;
							batches[idx].t_start = t_vertices_start + r2.start + t;
							batches[idx].s_length = end_s - s;
							batches[idx].t_length = end_t - t;
							batches[idx].pair_id = pair_id;
						}
					}
				}
			}
		}
	}
}

__global__ void kernel_refinement_contain_polygon(Task *batches, Point *d_vertices, uint *size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		uint s1 = batches[x].s_start;
		uint s2 = batches[x].t_start;
		uint len1 = batches[x].s_length;
		uint len2 = batches[x].t_length;
		int pair_id = batches[x].pair_id;
		if (resultmap[pair_id] != 0)
			return;

		if (segment_intersect_batch((d_vertices + s1), (d_vertices + s2), len1, len2))
		{
			resultmap[pair_id] = 3;
			return;
		}
	}
}

void process_chunk_contain_polygon(int start, int end, int base, query_context *gctx, Batch *h_pairs) {
	for (int idx = start; idx < end; idx ++) {
        Ideal *source = gctx->polygon_pairs[idx].first;
        Ideal *target = gctx->polygon_pairs[idx].second;
		
	    h_pairs[idx - base] = {*source->idealoffset, *target->idealoffset};
	}
}


uint cuda_contain_polygon(query_context *gctx)
{

	CudaTimer timer;

    uint polygon_pairs_size = gctx->polygon_pairs.size();
	uint batch_size = gctx->batch_size ? gctx->batch_size : gctx->polygon_pairs.size();
    int found = 0;

	timer.startTimer();
	Batch *h_pairs = new Batch[batch_size];
	uint8_t *h_resultmap = new uint8_t[batch_size];
	timer.stopTimer();
	printf("CPU Malloc time: %f ms\n", timer.getElapsedTime());

	timer.startTimer();
	Batch *d_pairs = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, batch_size * sizeof(Batch)));

	// resultmap status: 0(undecided / not contain), 1(contain), 2(contain)
	uint8_t *d_resultmap = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, batch_size * sizeof(uint8_t)));

    char *d_BufferInput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferInput, 8UL * 1024 * 1024 * 1024));
    uint *d_bufferinput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferinput_size, sizeof(uint)));
    uint h_bufferinput_size;

    char *d_BufferOutput = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferOutput, 8UL * 1024 * 1024 * 1024));
    uint *d_bufferoutput_size = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferoutput_size, sizeof(uint)));
    uint h_bufferoutput_size;

	timer.stopTimer();
	printf("cudaMalloc time: %f ms\n", timer.getElapsedTime());

    for (int i = 0; i < polygon_pairs_size; i += batch_size){
        int start = i, end = min(i + batch_size, polygon_pairs_size);
        int size = end - start;

		timer.startTimer();

		int num_threads = gctx->num_threads; 
		int chunk_size = (size + num_threads - 1) / num_threads; 

        std::vector<std::thread> threads;

        for (int j = 0; j < num_threads; ++ j) {
			int chunk_start = start + j * chunk_size;
			int chunk_end = min(chunk_start + chunk_size, end);

			threads.emplace_back(process_chunk_contain_polygon, chunk_start, chunk_end, start, gctx, h_pairs);
		}

        for (auto& thread : threads) {
			thread.join();
		}

		timer.stopTimer();
		printf("preprocessing for GPU (cpu time): %f ms\n", timer.getElapsedTime());

		timer.startTimer();
		CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(Batch), cudaMemcpyHostToDevice));
		timer.stopTimer();
		printf("cudaMemcpy time: %f ms\n", timer.getElapsedTime());

        timer.startTimer();
		CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(uint8_t)));
        CUDA_SAFE_CALL(cudaMemset(d_bufferinput_size, 0, sizeof(uint)));
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));
		timer.stopTimer();
		printf("cudaMemset time: %f ms\n", timer.getElapsedTime());

        timer.startTimer();

        /*1. Raster Model Filtering*/

        int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_size(BLOCK_SIZE, 1, 1);
        dim3 grid_size(grid_size_x, 1, 1);

        kernel_filter_contain_polygon<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, size, d_resultmap, (PixPair *)d_BufferInput, d_bufferinput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter_contain_polygon");

        /*2. Unroll Refinement*/

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_size.x = grid_size_x;

        kernel_unroll_contain_polygon<<<grid_size, block_size>>>((PixPair *)d_BufferInput, d_pairs, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, d_bufferinput_size, (Task *)d_BufferOutput, d_bufferoutput_size, d_resultmap);
        cudaDeviceSynchronize();
        check_execution("kernel_unroll_contain_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

        swap(d_BufferInput, d_BufferOutput);
        swap(d_bufferinput_size, d_bufferoutput_size);
        swap(h_bufferinput_size, h_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

        /*3. Refinement step*/

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_size.x = grid_size_x;

        kernel_refinement_contain_polygon<<<grid_size, block_size>>>((Task *)d_BufferInput, gctx->d_vertices, d_bufferinput_size, d_resultmap);
        cudaDeviceSynchronize();
        check_execution("kernel_refinement_contain_polygon");

        timer.stopTimer();
        printf("query time: %f ms\n", timer.getElapsedTime());

        CUDA_SAFE_CALL(cudaMemcpy(h_resultmap, d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        int found = 0;
        for (int i = 0; i < size; ++i)
        {
            if (h_resultmap[i] == 1)
                found++;
            if (h_resultmap[i] == 0)
            {
                Ideal *source = gctx->polygon_pairs[i].first;
                Ideal *target = gctx->polygon_pairs[i].second;
                Point p(target->getx(0), target->gety(0));
                if (source->contain(p, gctx))
                {
                    found++;
                }
            }
        }
    }

    delete[] h_pairs;
    delete[] h_resultmap;

    CUDA_SAFE_CALL(cudaFree(d_pairs));
    CUDA_SAFE_CALL(cudaFree(d_resultmap));
    CUDA_SAFE_CALL(cudaFree(d_BufferInput));
    CUDA_SAFE_CALL(cudaFree(d_BufferOutput));
    CUDA_SAFE_CALL(cudaFree(d_bufferinput_size));
    CUDA_SAFE_CALL(cudaFree(d_bufferoutput_size));

	return found;
}
