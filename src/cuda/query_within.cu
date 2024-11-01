// #include "geometry.cuh"

// #define WITHIN_DISTANCE 10

// struct Batch
// {
//     uint s_start = 0;
//     uint s_length = 0;
//     int pair_id = 0;
// };

// struct BoxDistRange
// {
//     int sourcePixelId;
//     double minDist;
//     double maxDist; // maxDist is not nessnary
//     int pairId;
//     int level = 0;
// };

// __global__ void kernel_init(PointPolygonPair *d_pairs, RasterInfo *d_info, uint size, double *distance, double *min_box_dist, double *max_box_dist)
// {
// 	const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (pair_id < size)
// 	{
// 		PointPolygonPair &pair = d_pairs[pair_id];
// 		IdealOffset &source = pair.source;
// 		Point &p = pair.target;
// 		box &s_mbr = d_info[source.info_start].mbr;

// 		distance[pair_id] = gpu_max_distance(p, s_mbr);
//         min_box_dist[pair_id] = DBL_MAX;
//         max_box_dist[pair_id] = DBL_MAX;		
// 	}
// }

// __global__ void first_cal_box_distance(PointPolygonPair *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint size, BoxDistRange *buffer, uint *buffer_size)
// {
//     const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (pair_id < size)
//     {
//         PointPolygonPair &pair = pairs[pair_id];
//         IdealOffset &source = pair.source;
//         Point &target = pair.target;     

//         uint16_t source_offset = (layer_offset + source.layer_offset_start)[*global_level];

//         box &s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr;
//         const double &s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, &s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
//         const int &s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, &s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;
//         for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i++)
//         {
// 			// printf("STATUS: %d\n", gpu_show_status(status, source.status_start, i, source_offset));
// 			if (gpu_show_status(status, source.status_start, i, source_offset) == BORDER)
// 			{
// 				auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
// 				double min_distance = gpu_distance(source_box, target);
// 				double max_distance = gpu_max_distance(target, source_box);
// 				int idx = atomicAdd(buffer_size, 1);
// 				buffer[idx] = {i, min_distance, max_distance, pair_id};
// 				atomicMinDouble(min_box_dist + pair_id, min_distance);
// 				atomicMinDouble(max_box_dist + pair_id, max_distance);
// 			}
//         }
//     }
// }

// __global__ void cal_box_distance(BoxDistRange *candidate, PointPolygonPair *pairs, RasterInfo *layer_info, uint16_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint *size, BoxDistRange *buffer, uint *buffer_size)
// {
//     const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (candidate_id < *size)
//     {
//         int source_pixel_id = candidate[candidate_id].sourcePixelId;
//         int pair_id = candidate[candidate_id].pairId;

//         PointPolygonPair &pair = pairs[pair_id];
//         IdealOffset &source = pair.source;
//         Point &target = pair.target;
//         uint level = pair.level;

//         if(*global_level > level){
//             int idx = atomicAdd(buffer_size, 1);
//             buffer[idx] = candidate[candidate_id];
//             return;
//         }

//         int source_start_x, source_start_y, source_end_x, source_end_y;
//         uint16_t source_offset;
//         box s_mbr;
//         double s_step_x, s_step_y;
//         int s_dimx, s_dimy;
//         box source_pixel_box;

//         if(*global_level > level) {
//             source_offset = (layer_offset + source.layer_offset_start)[level];
//             s_mbr = (layer_info + source.layer_info_start)[level].mbr;
//             s_step_x = (layer_info + source.layer_info_start)[level].step_x, s_step_y = (layer_info + source.layer_info_start)[level].step_y;
//             s_dimx = (layer_info + source.layer_info_start)[level].dimx, s_dimy = (layer_info + source.layer_info_start)[level].dimy;

//             source_start_x = gpu_get_x(source_pixel_id, s_dimx);
//             source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
//             source_end_x = gpu_get_x(source_pixel_id, s_dimx);
//             source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
//         }else{      
//             source_offset = (layer_offset + source.layer_offset_start)[*global_level];
//             s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr;
//             s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
//             s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;
//             source_pixel_box = gpu_get_pixel_box(
//                 gpu_get_x(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx), 
//                 gpu_get_y(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx, (layer_info + source.layer_info_start)[*global_level-1].dimy),
//                 (layer_info + source.layer_info_start)[*global_level-1].mbr.low[0], (layer_info + source.layer_info_start)[*global_level-1].mbr.low[1],
//                 (layer_info + source.layer_info_start)[*global_level-1].step_x, (layer_info + source.layer_info_start)[*global_level-1].step_y);
//             source_pixel_box.low[0] += 0.0001;
//             source_pixel_box.low[1] += 0.0001;
//             source_pixel_box.high[0] -= 0.0001;
//             source_pixel_box.high[1] -= 0.0001;

//             source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
//             source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
//             source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
//             source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
//         }

      

//         for(int x = source_start_x; x <= source_end_x; x ++){
//             for(int y = source_start_y; y <= source_end_y; y ++){
//                 int id = gpu_get_id(x, y, s_dimx);
// 				if (gpu_show_status(status, source.status_start, id, source_offset) == BORDER){
// 					auto bx = gpu_get_pixel_box(x, y, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
// 					double min_distance = gpu_distance(bx, target);
// 					double max_distance = gpu_max_distance(target, bx);
// 					int idx = atomicAdd(buffer_size, 1);
// 					buffer[idx] = {id, min_distance, max_distance, pair_id};
// 					atomicMinDouble(min_box_dist + pair_id, min_distance);
// 					atomicMinDouble(max_box_dist + pair_id, max_distance);
// 				}
//             }
//         }
//     }
// }

// __global__ void kernel_filter(BoxDistRange *bufferinput, double *min_box_dist, double *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
// {
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         double left = bufferinput[bufferId].minDist;
//         int pairId = bufferinput[bufferId].pairId;

//         if (left < max_box_dist[pairId])
//         {
//             int idx = atomicAdd(bufferoutput_size, 1);
//             bufferoutput[idx] = bufferinput[bufferId];
//         }
//     }
// }

// __global__ void kernel_unroll(BoxDistRange *pixpairs, PointPolygonPair *pairs, uint16_t *offset, EdgeSeq *edge_sequences, uint *size, Batch *batches, uint *batch_size)
// {
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         int pairId = pixpairs[bufferId].pairId;
//         int p = pixpairs[bufferId].sourcePixelId;

//         IdealOffset &source = pairs[pairId].source;
//         Point &p2 = pairs[pairId].target;

//         int s_num_sequence = (offset + source.offset_start)[p + 1] - (offset + source.offset_start)[p];

//         for (int i = 0; i < s_num_sequence; ++i)
//         {
//             EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(offset + source.offset_start)[p] + i];
// 			if (r.length < 2) continue;
// 			int max_size = 32;
// 			for (uint s = 0; s < r.length; s += max_size)
// 			{
// 				uint end_s = min(s + max_size, r.length);
// 				uint idx = atomicAdd(batch_size, 1U);

// 				batches[idx].s_start = source.vertices_start + r.start + s;
// 				batches[idx].s_length = end_s - s;
// 				batches[idx].pair_id = pairId;
// 			}
//         }
//     }
// }

// __global__ void kernel_refine(Batch *batches, PointPolygonPair *pairs, Point *vertices, uint *size, double *distance)
// {
//     const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (bufferId < *size)
//     {
//         uint s = batches[bufferId].s_start;
//         uint len = batches[bufferId].s_length;
//         int pair_id = batches[bufferId].pair_id;

// 		Point &target = pairs[pair_id].target;

//         double dist = gpu_point_to_segment_within_batch(target, vertices + s, len);

//         atomicMinDouble(distance + pair_id, dist);
//     }
// }

// uint cuda_within(query_context *gctx)
// {
// 	CudaTimer timer, duration;

// 	duration.startTimer();

// 	size_t size = gctx->point_polygon_pairs.size();

// 	printf("SIZE = %u\n", size);

// 	PointPolygonPair *h_pairs = new PointPolygonPair[size];
// 	PointPolygonPair *d_pairs = nullptr;

// 	for (int i = 0; i < size; ++i)
// 	{
// 		Point *target = gctx->point_polygon_pairs[i].first;
// 		Ideal *source = gctx->point_polygon_pairs[i].second;
// 		h_pairs[i] = {*source->idealoffset, *target, source->get_num_layers()};
// 	}

// 	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, size * sizeof(PointPolygonPair)));
// 	CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(PointPolygonPair), cudaMemcpyHostToDevice));

// 	double *h_distance = new double[size * sizeof(double)];
// 	double *d_distance = nullptr;
// 	CUDA_SAFE_CALL(cudaMalloc((void **)&d_distance, size * sizeof(double)));

//     double *d_min_box_dist = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_min_box_dist, size * sizeof(double)));	

//     double *d_max_box_dist = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_max_box_dist, size * sizeof(double)));

//     char *d_BufferInput = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferInput, 4UL * 1024 * 1024 * 1024));
//     uint *d_bufferinput_size = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferinput_size, sizeof(uint)));
//     CUDA_SAFE_CALL(cudaMemset(d_bufferinput_size, 0, sizeof(uint)));
//     uint h_bufferinput_size;

//     char *d_BufferOutput = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_BufferOutput, 4UL * 1024 * 1024 * 1024));
//     uint *d_bufferoutput_size = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_bufferoutput_size, sizeof(uint)));
//     CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));
//     uint h_bufferoutput_size;

// 	int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
// 	dim3 block_size(BLOCK_SIZE, 1, 1);
// 	dim3 grid_size(grid_size_x, 1, 1);

// 	timer.startTimer();

// 	kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_info, size, d_distance, d_min_box_dist, d_max_box_dist);
// 	cudaDeviceSynchronize();
// 	check_execution("kernel init");

// 	timer.stopTimer();
// 	printf("distance initialize time: %f ms\n", timer.getElapsedTime());

// 	uint h_level = 0;
//     uint *d_level = nullptr;
//     CUDA_SAFE_CALL(cudaMalloc((void **)&d_level, sizeof(uint)));
//     CUDA_SAFE_CALL(cudaMemset(d_level, 0, sizeof(uint)));

// 	// grid_size_x = (size + 512 - 1) / 512;
// 	// block_size.x = 512;
// 	// grid_size.x = grid_size_x;

//     timer.startTimer();
//     first_cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_min_box_dist, d_max_box_dist, d_level, size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
//     cudaDeviceSynchronize();
//     check_execution("first_cal_box_distance");
//     timer.stopTimer();
//     printf("kernel first calculate box distance: %f ms\n", timer.getElapsedTime());

// 	/* To delete  */
//     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
//     CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
//     printf("h_bufferinput_size = %u\n", h_bufferinput_size);
//     printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
//     /*   To delete  */

//     while(true){
//         h_level ++;
//         CUDA_SAFE_CALL(cudaMemcpy(d_level, &h_level, sizeof(uint), cudaMemcpyHostToDevice));
//         if(h_level > gctx->num_layers) break;

//         std::swap(d_BufferInput, d_BufferOutput);
//         std::swap(d_bufferinput_size, d_bufferoutput_size);
//         std::swap(h_bufferinput_size, h_bufferoutput_size);
//         CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

//         grid_size_x = (h_bufferinput_size + 512 - 1) / 512;
//         block_size.x = 512;
//         grid_size.x = grid_size_x;

//         timer.startTimer();
//         cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_pairs, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, d_min_box_dist, d_max_box_dist, d_level, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
//         cudaDeviceSynchronize();
//         check_execution("cal_box_distance");
//         timer.stopTimer();
//         printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

// 		// /* To delete  */
// 		CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
// 		CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
// 		// printf("h_bufferinput_size = %u\n", h_bufferinput_size);
// 		// printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
// 		// /*   To delete  */

//         if(h_bufferinput_size == h_bufferoutput_size) break;

//         std::swap(d_BufferInput, d_BufferOutput);
//         std::swap(d_bufferinput_size, d_bufferoutput_size);
//         std::swap(h_bufferinput_size, h_bufferoutput_size);
//         CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

//         grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//         block_size.x = BLOCK_SIZE;
//         grid_size.x = grid_size_x;

//         timer.startTimer();
//         kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_min_box_dist, d_max_box_dist, d_bufferinput_size, (BoxDistRange *)d_BufferOutput, d_bufferoutput_size);
//         cudaDeviceSynchronize();
//         check_execution("kernel_filter");
//         timer.stopTimer();
//         printf("kernel filter: %f ms\n", timer.getElapsedTime());
//     }

//     swap(d_BufferInput, d_BufferOutput);
//     swap(d_bufferinput_size, d_bufferoutput_size);
//     swap(h_bufferinput_size, h_bufferoutput_size);
//     CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

//     grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     block_size.x = BLOCK_SIZE;
//     grid_size.x = grid_size_x;

//     timer.startTimer();

//     kernel_unroll<<<grid_size, block_size>>>((BoxDistRange *)d_BufferInput, d_pairs, gctx->d_offset, gctx->d_edge_sequences, d_bufferinput_size, (Batch *)d_BufferOutput, d_bufferoutput_size);
//     cudaDeviceSynchronize();
//     check_execution("kernel_unroll");

//     timer.stopTimer();
//     printf("kernel unroll: %f ms\n", timer.getElapsedTime());

//     swap(d_BufferInput, d_BufferOutput);
//     swap(d_bufferinput_size, d_bufferoutput_size);
//     swap(h_bufferinput_size, h_bufferoutput_size);
//     CUDA_SAFE_CALL(cudaMemset(d_bufferoutput_size, 0, sizeof(uint)));

//     grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     block_size.x = BLOCK_SIZE;
//     grid_size.x = grid_size_x;

//     timer.startTimer();

//     kernel_refine<<<grid_size, block_size>>>((Batch *)d_BufferInput, d_pairs, gctx->d_vertices, d_bufferinput_size, d_distance);
//     cudaDeviceSynchronize();
//     check_execution("kernel_refine");

//     timer.stopTimer();
//     printf("kernel refine: %f ms\n", timer.getElapsedTime());

//     duration.stopTimer();
//     printf("kernel total time = %lf ms\n", duration.getElapsedTime());

// 	CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
// 	int found = 0;
// 	for (int i = 0; i < size; i++)
// 	{
// 		if (h_distance[i] <= WITHIN_DISTANCE)
// 			found++;
// 		printf("%lf\n", h_distance[i]);
// 	}

// 	return found;
// }