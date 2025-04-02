// #include "geometry.cuh"
// #include "Ideal.h"

// struct Task
// {
//     uint s_start = 0;
//     uint t_start = 0;
//     uint s_length = 0;
//     uint t_length = 0;
//     int pair_id = 0;
// };

// __global__ void kernel_filter_contain_polygon(IdealPair *pairs, IdealOffset *idealoffset, RasterInfo *info,
// 											  uint8_t *status, uint size, uint *result,
// 											  PixPair *pixpairs, uint *pp_size)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (x >= size) return; 

// 	IdealPair &pair = pairs[x];
// 	IdealOffset &source = idealoffset[pair.source];
// 	IdealOffset &target = idealoffset[pair.target];

// 	const box &s_mbr = info[pair.source].mbr, &t_mbr = info[pair.target].mbr;				
// 	const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y; 
// 	const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;			 
// 	const double &t_step_x = info[pair.target].step_x, &t_step_y = info[pair.target].step_y; 
// 	const int &t_dimx = info[pair.target].dimx, &t_dimy = info[pair.target].dimy;			

// 	uint itn = 0, etn = 0;	 
// 	uint8_t flag_out_in = 0; 
// 	uint border_count = 0;	

// 	int i_min = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0], s_step_x, s_dimx);
// 	int i_max = gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0], s_step_x, s_dimx);
// 	int j_min = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1], s_step_y, s_dimy);
// 	int j_max = gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1], s_step_y, s_dimy);

// 	for (int i = i_min; i <= i_max; i++)
// 	{
// 		for (int j = j_min; j <= j_max; j++)
// 		{
// 			int p = gpu_get_id(i, j, s_dimx);
// 			uint8_t source_status = gpu_show_status(status, source.status_start, p);

// 			itn += (source_status == IN);
// 			etn += (source_status == OUT);

// 			box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

// 			int _i_min = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx);
// 			int _i_max = gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx);
// 			int _j_min = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy);
// 			int _j_max = gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy);

// 			for (int _i = _i_min; _i <= _i_max; _i++)
// 			{
// 				for (int _j = _j_min; _j <= _j_max; _j++)
// 				{
// 					int p2 = gpu_get_id(_i, _j, t_dimx);
// 					uint8_t target_status = gpu_show_status(status, target.status_start, p2);

// 					flag_out_in |= ((source_status == OUT) & (target_status == IN));

// 					if (source_status == BORDER && target_status == BORDER)
// 					{
// 						int idx = atomicAdd(pp_size, 1U);
// 						pixpairs[idx].source_pixid = p;
// 						pixpairs[idx].target_pixid = p2;
// 						pixpairs[idx].pair_id = x;
// 						border_count++;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	uint total_pixels = (i_max - i_min + 1) * (j_max - j_min + 1);

// 	bool is_contained = (itn == total_pixels); 
//     if(is_contained) atomicAdd(result, 1);
//     return;
// }

// __global__ void kernel_unroll_contain_polygon(PixPair *pixpairs,
// 											  IdealPair *pairs,
// 											  IdealOffset *idealoffset,
// 											  uint8_t *status,
// 											  uint32_t *es_offset,
// 											  EdgeSeq *edge_sequences,
// 											  uint *size,
// 											  Task *tasks,
// 											  uint *task_size,
// 											  uint8_t *resultmap)
// {
// 	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 	if(idx >= *size || resultmap[pixpairs[idx].pair_id] != 0) return;

// 	int p = pixpairs[idx].source_pixid;
// 	int p2 = pixpairs[idx].target_pixid;
// 	int pair_id = pixpairs[idx].pair_id;

// 	IdealPair pair = pairs[pair_id];
// 	IdealOffset source = idealoffset[pair.source];
// 	IdealOffset target = idealoffset[pair.target];

// 	uint s_offset_start = source.offset_start;
// 	uint t_offset_start = target.offset_start;
// 	uint s_edge_sequences_start = source.edge_sequences_start;
// 	uint t_edge_sequences_start = target.edge_sequences_start;

// 	int s_num_sequence = (es_offset + s_offset_start)[p + 1] - (es_offset + s_offset_start)[p];
// 	int t_num_sequence = (es_offset + t_offset_start)[p2 + 1] - (es_offset + t_offset_start)[p2];
// 	uint s_vertices_start = source.vertices_start;
// 	uint t_vertices_start = target.vertices_start;

// 	const int max_size = 8;

// 	for (int i = 0; i < s_num_sequence; ++i)
// 	{
// 		EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
// 		for (int j = 0; j < t_num_sequence; ++j)
// 		{
// 			EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
// 			for (uint s = 0; s < r.length; s += max_size)
// 			{
// 				uint end_s = min(s + max_size, r.length);
// 				for (uint t = 0; t < r2.length; t += max_size)
// 				{
// 					uint end_t = min(t + max_size, r2.length);

// 					uint idx_task = atomicAdd(task_size, 1U);
// 					tasks[idx_task].s_start = s_vertices_start + r.start + s;
// 					tasks[idx_task].t_start = t_vertices_start + r2.start + t;
// 					tasks[idx_task].s_length = end_s - s;
// 					tasks[idx_task].t_length = end_t - t;
// 					tasks[idx_task].pair_id = pair_id;
// 				}
// 			}
// 		}
// 	}
// }

// __global__ void kernel_refinement_contain_polygon(Task *tasks, Point *d_vertices, uint *size, uint8_t *resultmap)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (x >= *size) return;
	
// 	uint s1 = tasks[x].s_start;
// 	uint s2 = tasks[x].t_start;
// 	uint len1 = tasks[x].s_length;
// 	uint len2 = tasks[x].t_length;
// 	int pair_id = tasks[x].pair_id;

// 	bool should_process = (resultmap[pair_id] == 0);

// 	bool has_intersection = should_process && gpu_segment_intersect_batch((d_vertices + s1), (d_vertices + s2), len1, len2);

// 	if (has_intersection) {
//         resultmap[pair_id] = 3;
//     }
// }

// uint cuda_contain_polygon(query_context *gctx)
// {
//     uint h_bufferinput_size, h_bufferoutput_size;
// #ifdef DEBUG
//     CudaTimer timer;
//     timer.startTimer();
// #endif
//     /*1. Raster Model Filtering*/

//     int grid_size_x = (gctx->num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 block_size(BLOCK_SIZE, 1, 1);
//     dim3 grid_size(grid_size_x, 1, 1);

//     kernel_filter_contain_polygon<<<grid_size, block_size>>>(d_pairs + start, gctx->d_idealoffset, gctx->d_info, gctx->d_status, size, gctx->d_resultmap, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size);
//     cudaDeviceSynchronize();
//     check_execution("kernel_filter_contain_polygon");

//     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
//     printf("h_buffer_size = %u\n", h_bufferinput_size);

//     /*2. Unroll Refinement*/

//     grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     grid_size.x = grid_size_x;

//     starter.startTimer();
//     kernel_unroll_contain_polygon<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, d_pairs + start, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_resultmap);
//     cudaDeviceSynchronize();
//     check_execution("kernel_unroll_contain_polygon");
//     starter.stopTimer();
//     printf("unroll time: %f ms\n", starter.getElapsedTime());

//     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
//     printf("h_buffer_size = %u\n", h_bufferoutput_size);
    
//     /*3. Refinement step*/

//     grid_size_x = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     grid_size.x = grid_size_x;

//     kernel_refinement_contain_polygon<<<grid_size, block_size>>>((Task *)gctx->d_BufferOutput, gctx->d_vertices, gctx->d_bufferoutput_size, gctx->d_resultmap);
//     cudaDeviceSynchronize();
//     check_execution("kernel_refinement_contain_polygon");

//     timer.stopTimer();
//     printf("query time: %f ms\n", timer.getElapsedTime());

//     CUDA_SAFE_CALL(cudaMemcpy(gctx->h_resultmap, gctx->d_resultmap, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

//     for (int i = 0; i < size; ++i)
//     {
//         if (gctx->h_resultmap[i] == 1)
//             found++;
//         if (gctx->h_resultmap[i] == 0)
//         {
//             Ideal *source = gctx->source_ideals[(h_pairs + start)[i].source];
//             Ideal *target = gctx->target_ideals[(h_pairs + start)[i].target - gctx->source_ideals.size()];
//             Point p(target->getx(0), target->gety(0));
//             if (source->contain(p, gctx))
//             {
//                 found++;
//             }
//         }
//     }

//     return found;
// }
