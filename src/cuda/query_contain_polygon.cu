#include "geometry.cuh"
#include "Ideal.h"

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

__global__ void kernel_filter_contain_polygon(IdealPair *pairs, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, uint size, uint8_t *resultmap, PixPair *pixpairs, uint *pp_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		IdealPair &pair = pairs[x];
		IdealOffset &source = idealoffset[pair.source];
		IdealOffset &target = idealoffset[pair.target];

		const box &s_mbr = info[pair.source].mbr, &t_mbr = info[pair.target].mbr;
		const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y;
		const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;
		const double &t_step_x = info[pair.target].step_x, &t_step_y = info[pair.target].step_y;
		const int &t_dimx = info[pair.target].dimx, &t_dimy = info[pair.target].dimy;

		uint itn = 0, etn = 0;
		for (int i = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx); i <= gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx); i++)
		{
			for (int j = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy); j <= gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy); j++)
			{
				int p = gpu_get_id(i, j, s_dimx);
				if (gpu_show_status(status, source.status_start, p) == IN)
					itn++;
				else if (gpu_show_status(status, source.status_start, p) == OUT)
					etn++;

				box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

				for (int _i = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx); _i <= gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx); _i++)
				{
					for (int _j = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy); _j <= gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy); _j++)
					{
						int p2 = gpu_get_id(_i, _j, t_dimx);
						if (gpu_show_status(status, source.status_start, p) == OUT && gpu_show_status(status, target.status_start, p2) == IN)
						{
							resultmap[x] = 2;
							return;
						}
						if (gpu_show_status(status, source.status_start, p) == BORDER && gpu_show_status(status, target.status_start, p2) == BORDER)
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

__global__ void kernel_unroll_contain_polygon(PixPair *pixpairs, IdealPair *pairs, IdealOffset *idealoffset, uint8_t *status, uint32_t *es_offset, EdgeSeq *edge_sequences, uint *size, Task *tasks, uint *task_size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int p = pixpairs[x].source_pixid;
		int p2 = pixpairs[x].target_pixid;
		int pair_id = pixpairs[x].pair_id;
		if (resultmap[pair_id] != 0)
			return;

		IdealPair &pair = pairs[pair_id];
		IdealOffset &source = idealoffset[pair.source];
		IdealOffset &target = idealoffset[pair.target];

		if (gpu_show_status(status, source.status_start, p) == BORDER && gpu_show_status(status, target.status_start, p2) == BORDER)
		{
			uint s_offset_start = source.offset_start, t_offset_start = target.offset_start;
			uint s_edge_sequences_start = source.edge_sequences_start, t_edge_sequences_start = target.edge_sequences_start;
			int s_num_sequence = (es_offset + s_offset_start)[p + 1] - (es_offset + s_offset_start)[p];
			int t_num_sequence = (es_offset + t_offset_start)[p2 + 1] - (es_offset + t_offset_start)[p2];
			uint s_vertices_start = source.vertices_start, t_vertices_start = target.vertices_start;

			for (int i = 0; i < s_num_sequence; ++i)
			{
				EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
				for (int j = 0; j < t_num_sequence; ++j)
				{
					EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
					int max_size = 8;
					for (uint s = 0; s < r.length; s += max_size)
					{
						uint end_s = min(s + max_size, r.length);
						for (uint t = 0; t < r2.length; t += max_size)
						{
							uint end_t = min(t + max_size, r2.length);
							uint idx = atomicAdd(task_size, 1U);
							tasks[idx].s_start = s_vertices_start + r.start + s;
							tasks[idx].t_start = t_vertices_start + r2.start + t;
							tasks[idx].s_length = end_s - s;
							tasks[idx].t_length = end_t - t;
							tasks[idx].pair_id = pair_id;
						}
					}
				}
			}
		}
	}
}

__global__ void kernel_refinement_contain_polygon(Task *tasks, Point *d_vertices, uint *size, uint8_t *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		uint s1 = tasks[x].s_start;
		uint s2 = tasks[x].t_start;
		uint len1 = tasks[x].s_length;
		uint len2 = tasks[x].t_length;
		int pair_id = tasks[x].pair_id;
		if (resultmap[pair_id] != 0)
			return;

		if (segment_intersect_batch((d_vertices + s1), (d_vertices + s2), len1, len2))
		{
			resultmap[pair_id] = 3;
			return;
		}
	}
}

uint cuda_contain_polygon(query_context *gctx)
{

    CudaTimer timer;

    uint polygon_pairs_size = gctx->polygon_pairs.size();
    printf("SIZE = %d\n", polygon_pairs_size);
    uint batch_size = gctx->batch_size;
    int found = 0;

    IdealPair *h_pairs = new IdealPair[polygon_pairs_size];

    for (int i = 0; i < polygon_pairs_size; i++)
    {
        h_pairs[i].target = gctx->polygon_pairs[i].first;
        h_pairs[i].source = gctx->polygon_pairs[i].second;
    }

    IdealPair *d_pairs = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, polygon_pairs_size * sizeof(IdealPair)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, polygon_pairs_size * sizeof(IdealPair), cudaMemcpyHostToDevice));

    uint h_bufferinput_size, h_bufferoutput_size;

    for (int i = 0; i < polygon_pairs_size; i += batch_size)
    {
        int start = i, end = min(i + batch_size, polygon_pairs_size);
        int size = end - start;

        CUDA_SAFE_CALL(cudaMemset(gctx->d_resultmap, 0, size * sizeof(uint8_t)));
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

        timer.startTimer();

        /*1. Raster Model Filtering*/

        int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_size(BLOCK_SIZE, 1, 1);
        dim3 grid_size(grid_size_x, 1, 1);

        kernel_filter_contain_polygon<<<grid_size, block_size>>>(d_pairs + start, gctx->d_idealoffset, gctx->d_info, gctx->d_status, size, gctx->d_resultmap, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_filter_contain_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
		printf("h_buffer_size = %u\n", h_bufferinput_size);

        /*2. Unroll Refinement*/

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_size.x = grid_size_x;

        kernel_unroll_contain_polygon<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, d_pairs + start, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_resultmap);
        cudaDeviceSynchronize();
        check_execution("kernel_unroll_contain_polygon");

        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
		printf("h_buffer_size = %u\n", h_bufferoutput_size);
        
        /*3. Refinement step*/

        grid_size_x = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_size.x = grid_size_x;

        kernel_refinement_contain_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferOutput, gctx->d_vertices, gctx->d_bufferoutput_size, gctx->d_resultmap);
        cudaDeviceSynchronize();
        check_execution("kernel_refinement_contain_polygon");

        timer.stopTimer();
        printf("query time: %f ms\n", timer.getElapsedTime());

        CUDA_SAFE_CALL(cudaMemcpy(gctx->h_resultmap, gctx->d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < size; ++i)
        {
            if (gctx->h_resultmap[i] == 1)
                found++;
            if (gctx->h_resultmap[i] == 0)
            {
                Ideal *source = gctx->source_ideals[(h_pairs + start)[i].source];
                Ideal *target = gctx->target_ideals[(h_pairs + start)[i].target - gctx->source_ideals.size()];
                Point p(target->getx(0), target->gety(0));
                if (source->contain(p, gctx))
                {
                    found++;
                }
            }
        }
    }

    delete[] h_pairs;

    CUDA_SAFE_CALL(cudaFree(d_pairs));

    return found;
}
