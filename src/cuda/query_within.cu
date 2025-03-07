#include "geometry.cuh"

#define WITHIN_DISTANCE 10

struct Task
{
    uint s_start = 0;
    uint s_length = 0;
    int pair_id = 0;
};

struct BoxDistRange
{
    int sourcePixelId;
    double minDist;
    double maxDist; // maxDist is not nessnary
    int pairId;
    int level = 0;
};

__global__ void kernel_init(IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint size, double *distance, double *min_box_dist, double *max_box_dist)
{
	const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair_id < size)
	{
		IdealPair &pair = pairs[pair_id];
		IdealOffset &source = idealoffset[pair.source];
		Point &p = points[pair.target];

		box &s_mbr = info[pair.source].mbr;

		distance[pair_id] = gpu_max_distance(p, s_mbr);
        min_box_dist[pair_id] = DBL_MAX;
        max_box_dist[pair_id] = DBL_MAX;		
	}
}

__global__ void cal_box_distance(IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *info, uint8_t *status, double *min_box_dist, double *max_box_dist, uint size, BoxDistRange *buffer, uint *buffer_size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        IdealPair &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.source];
        Point &target = points[pair.target];     

        box &s_mbr = info[pair.source].mbr;
        const double &s_step_x = info[pair.source].step_x, &s_step_y = info[pair.source].step_y;
        const int &s_dimx = info[pair.source].dimx, &s_dimy = info[pair.source].dimy;

        for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i++)
        {
			// printf("STATUS: %d\n", gpu_show_status(status, source.status_start, i, source_offset));
			if (gpu_show_status(status, source.status_start, i) == BORDER)
			{
				auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
				double min_distance = gpu_distance(source_box, target);
				double max_distance = gpu_max_distance(target, source_box);
				int idx = atomicAdd(buffer_size, 1);
				buffer[idx] = {i, min_distance, max_distance, pair_id};
				atomicMinDouble(min_box_dist + pair_id, min_distance);
				atomicMinDouble(max_box_dist + pair_id, max_distance);
			}
        }
    }
}

__global__ void h_first_cal_box_distance(IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint size, BoxDistRange *buffer, uint *buffer_size)
{
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id < size)
    {
        IdealPair &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.source];
        Point &target = points[pair.target];    
        
        uint32_t source_offset = (layer_offset + source.layer_offset_start)[*global_level];

        box &s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr;
        const double &s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, &s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
        const int &s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, &s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;

        for (int i = 0; i < (s_dimx + 1) * (s_dimy + 1); i++)
        {
			// printf("STATUS: %d\n", gpu_show_status(status, source.status_start, i, source_offset));
			if (gpu_show_status(status, source.status_start, i, source_offset) == BORDER)
			{
				auto source_box = gpu_get_pixel_box(gpu_get_x(i, s_dimx), gpu_get_y(i, s_dimx, s_dimy), s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
				double min_distance = gpu_distance(source_box, target);
				double max_distance = gpu_max_distance(target, source_box);
				int idx = atomicAdd(buffer_size, 1);
				buffer[idx] = {i, min_distance, max_distance, pair_id};
				atomicMinDouble(min_box_dist + pair_id, min_distance);
				atomicMinDouble(max_box_dist + pair_id, max_distance);
			}
        }
    }
}

__global__ void h_cal_box_distance(BoxDistRange *candidate, IdealPair *pairs, Point *points, IdealOffset *idealoffset, RasterInfo *layer_info, uint32_t *layer_offset, uint8_t *status, double *min_box_dist, double *max_box_dist, uint *global_level, uint *size, BoxDistRange *buffer, uint *buffer_size)
{
    const int candidate_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate_id < *size)
    {
        int source_pixel_id = candidate[candidate_id].sourcePixelId;
        int pair_id = candidate[candidate_id].pairId;

        IdealPair &pair = pairs[pair_id];
        IdealOffset &source = idealoffset[pair.source];
        Point &target = points[pair.target];
        uint level = pair.level;

        if(*global_level > level){
            int idx = atomicAdd(buffer_size, 1);
            buffer[idx] = candidate[candidate_id];
            return;
        }

        int source_start_x, source_start_y, source_end_x, source_end_y;
        uint32_t source_offset;
        box s_mbr;
        double s_step_x, s_step_y;
        int s_dimx, s_dimy;
        box source_pixel_box;

        if(*global_level > level) {
            source_offset = (layer_offset + source.layer_offset_start)[level];
            s_mbr = (layer_info + source.layer_info_start)[level].mbr;
            s_step_x = (layer_info + source.layer_info_start)[level].step_x, s_step_y = (layer_info + source.layer_info_start)[level].step_y;
            s_dimx = (layer_info + source.layer_info_start)[level].dimx, s_dimy = (layer_info + source.layer_info_start)[level].dimy;

            source_start_x = gpu_get_x(source_pixel_id, s_dimx);
            source_start_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
            source_end_x = gpu_get_x(source_pixel_id, s_dimx);
            source_end_y = gpu_get_y(source_pixel_id, s_dimx, s_dimy);
        }else{      
            source_offset = (layer_offset + source.layer_offset_start)[*global_level];
            s_mbr = (layer_info + source.layer_info_start)[*global_level].mbr;
            s_step_x = (layer_info + source.layer_info_start)[*global_level].step_x, s_step_y = (layer_info + source.layer_info_start)[*global_level].step_y;
            s_dimx = (layer_info + source.layer_info_start)[*global_level].dimx, s_dimy = (layer_info + source.layer_info_start)[*global_level].dimy;
            source_pixel_box = gpu_get_pixel_box(
                gpu_get_x(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx), 
                gpu_get_y(source_pixel_id, (layer_info + source.layer_info_start)[*global_level-1].dimx, (layer_info + source.layer_info_start)[*global_level-1].dimy),
                (layer_info + source.layer_info_start)[*global_level-1].mbr.low[0], (layer_info + source.layer_info_start)[*global_level-1].mbr.low[1],
                (layer_info + source.layer_info_start)[*global_level-1].step_x, (layer_info + source.layer_info_start)[*global_level-1].step_y);
            source_pixel_box.low[0] += 0.0001;
            source_pixel_box.low[1] += 0.0001;
            source_pixel_box.high[0] -= 0.0001;
            source_pixel_box.high[1] -= 0.0001;

            source_start_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.low[0], s_step_x, s_dimx);
            source_start_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.low[1], s_step_y, s_dimy);
            source_end_x = gpu_get_offset_x(s_mbr.low[0], source_pixel_box.high[0], s_step_x, s_dimx);
            source_end_y = gpu_get_offset_y(s_mbr.low[1], source_pixel_box.high[1], s_step_y, s_dimy);
        }

      

        for(int x = source_start_x; x <= source_end_x; x ++){
            for(int y = source_start_y; y <= source_end_y; y ++){
                int id = gpu_get_id(x, y, s_dimx);
				if (gpu_show_status(status, source.status_start, id, source_offset) == BORDER){
					auto bx = gpu_get_pixel_box(x, y, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);
					double min_distance = gpu_distance(bx, target);
					double max_distance = gpu_max_distance(target, bx);
					int idx = atomicAdd(buffer_size, 1);
					buffer[idx] = {id, min_distance, max_distance, pair_id};
					atomicMinDouble(min_box_dist + pair_id, min_distance);
					atomicMinDouble(max_box_dist + pair_id, max_distance);
				}
            }
        }
    }
}



__global__ void kernel_filter(BoxDistRange *bufferinput, double *max_box_dist, uint *size, BoxDistRange *bufferoutput, uint *bufferoutput_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        double left = bufferinput[bufferId].minDist;
        int pairId = bufferinput[bufferId].pairId;

        if (left < max_box_dist[pairId])
        {
            int idx = atomicAdd(bufferoutput_size, 1);
            bufferoutput[idx] = bufferinput[bufferId];
        }
    }
}

__global__ void kernel_unroll(BoxDistRange *pixpairs, IdealPair *pairs, Point *points, IdealOffset *idealoffset, uint32_t *es_offset, EdgeSeq *edge_sequences, uint *size, Task *batches, uint *batch_size)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        int pairId = pixpairs[bufferId].pairId;
        int p = pixpairs[bufferId].sourcePixelId;

        IdealPair &pair = pairs[pairId];
        IdealOffset &source = idealoffset[pair.source];
        Point &p2 = points[pair.target];

        int s_num_sequence = (es_offset + source.offset_start)[p + 1] - (es_offset + source.offset_start)[p];

        for (int i = 0; i < s_num_sequence; ++i)
        {
            EdgeSeq r = (edge_sequences + source.edge_sequences_start)[(es_offset + source.offset_start)[p] + i];
			if (r.length < 2) continue;
			int max_size = 8;
			for (uint s = 0; s < r.length; s += max_size)
			{
				uint end_s = min(s + max_size, r.length);
				uint idx = atomicAdd(batch_size, 1U);

				batches[idx].s_start = source.vertices_start + r.start + s;
				batches[idx].s_length = end_s - s;
				batches[idx].pair_id = pairId;
			}
        }
    }
}

__global__ void kernel_refine(Task *tasks, IdealPair *pairs, Point *points, Point *vertices, uint *size, double *distance,  double *degree_per_kilometer_latitude, double *degree_per_kilometer_longitude_arr)
{
    const int bufferId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bufferId < *size)
    {
        uint s = tasks[bufferId].s_start;
        uint len = tasks[bufferId].s_length;
        int pair_id = tasks[bufferId].pair_id;

        IdealPair &pair = pairs[pair_id];
		Point &target = points[pair.target];

        double dist = gpu_point_to_segment_within_batch(target, vertices + s, len, degree_per_kilometer_latitude, degree_per_kilometer_longitude_arr);

        atomicMinDouble(distance + pair_id, dist);
    }
}

uint cuda_within(query_context *gctx)
{
	CudaTimer timer;

    uint point_polygon_pairs_size = gctx->point_polygon_pairs.size();
	uint batch_size = gctx->batch_size;
    int found = 0;

	printf("SIZE = %u\n", point_polygon_pairs_size);

	// IdealPair *h_pairs = new IdealPair[point_polygon_pairs_size];
    
	// for (int i = 0; i < point_polygon_pairs_size; i++)
	// {
	// 	h_pairs[i].target = gctx->point_polygon_pairs[i].first;
	// 	h_pairs[i].source = gctx->point_polygon_pairs[i].second;
	// }

    // IdealPair *d_pairs = nullptr;
	// CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, point_polygon_pairs_size * sizeof(IdealPair)));
	// CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, point_polygon_pairs_size * sizeof(IdealPair), cudaMemcpyHostToDevice));

	uint h_bufferinput_size, h_bufferoutput_size; 

    for(int i = 0; i < point_polygon_pairs_size; i += batch_size)
    {
		int start = i, end = min(i + batch_size, point_polygon_pairs_size);
		int size = end - start;

        IdealPair *h_pairs = new IdealPair[size];
    
        for (int k = 0; k < size; k++)
        {
            h_pairs[k].target = gctx->point_polygon_pairs[k + start].first;
            h_pairs[k].source = gctx->point_polygon_pairs[k + start].second;
            h_pairs[k].level = gctx->source_ideals[h_pairs[k].source]->get_num_layers();
        }

        IdealPair *d_pairs = nullptr;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_pairs, size * sizeof(IdealPair)));
        CUDA_SAFE_CALL(cudaMemcpy(d_pairs, h_pairs, size * sizeof(IdealPair), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemset(gctx->d_resultmap, 0, size * sizeof(uint8_t)));
		CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
		CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
        if(gctx->use_hierachy){
            gctx->h_level = 0;
            CUDA_SAFE_CALL(cudaMemset(gctx->d_level, 0, sizeof(uint)));
        }
        printf("size = %d\n", size);

        int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_size(BLOCK_SIZE, 1, 1);
        dim3 grid_size(grid_size_x, 1, 1);

        timer.startTimer();
        kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_info, size, gctx->d_distance, gctx->d_min_box_dist, gctx->d_max_box_dist);
        cudaDeviceSynchronize();
        check_execution("kernel init");
        timer.stopTimer();
        printf("distance initialize time: %f ms\n", timer.getElapsedTime());

        if(gctx->use_hierachy){
            printf("level: %d\n", gctx->h_level);

            timer.startTimer();
            h_first_cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, gctx->d_min_box_dist, gctx->d_max_box_dist, gctx->d_level, size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
            cudaDeviceSynchronize();
            check_execution("h_first_cal_box_distance");
            timer.stopTimer();
            printf("kernel first calculate box distance: %f ms\n", timer.getElapsedTime());

            /* To delete  */
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
            /*   To delete  */

            while(true){
                gctx->h_level ++;
                printf("level: %d\n", gctx->h_level);
                CUDA_SAFE_CALL(cudaMemcpy(gctx->d_level, &gctx->h_level, sizeof(uint), cudaMemcpyHostToDevice));
                if(gctx->h_level > gctx->num_layers) break;
    
                std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);
                std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);
                CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
        
                grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                block_size.x = BLOCK_SIZE;
                grid_size.x = grid_size_x;
        
                timer.startTimer();
                h_cal_box_distance<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_layer_info, gctx->d_layer_offset, gctx->d_status, gctx->d_min_box_dist, gctx->d_max_box_dist, gctx->d_level, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
                cudaDeviceSynchronize();
                check_execution("h_cal_box_distance");
                timer.stopTimer();
                printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

                CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
		        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        
                printf("h_bufferoutput_size = %u\n", h_bufferoutput_size);
                if(h_bufferinput_size == h_bufferoutput_size) break;

                std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);
                std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);
                CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
        
                grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                block_size.x = BLOCK_SIZE;
                grid_size.x = grid_size_x;
        
                timer.startTimer();
                kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
                cudaDeviceSynchronize();
                check_execution("kernel_filter");
                timer.stopTimer();
                printf("kernel filter: %f ms\n", timer.getElapsedTime());
            }
        }else{
            timer.startTimer();
            cal_box_distance<<<grid_size, block_size>>>(d_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_info, gctx->d_status, gctx->d_min_box_dist, gctx->d_max_box_dist, size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
            cudaDeviceSynchronize();
            check_execution("first_cal_box_distance");
            timer.stopTimer();
            printf("kernel calculate box distance: %f ms\n", timer.getElapsedTime());

            std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);
            std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);
            CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));
    
            printf("filter bufferinput_size = %d\n", h_bufferinput_size);
    
            grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            block_size.x = BLOCK_SIZE;
            grid_size.x = grid_size_x;
    
            timer.startTimer();
            kernel_filter<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, gctx->d_max_box_dist, gctx->d_bufferinput_size, (BoxDistRange *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
            cudaDeviceSynchronize();
            check_execution("kernel_filter");
            timer.stopTimer();
            printf("kernel filter: %f ms\n", timer.getElapsedTime());
        }

        std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);
        std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

        printf("refine bufferinput_size = %d\n", h_bufferinput_size);

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        timer.startTimer();
        kernel_unroll<<<grid_size, block_size>>>((BoxDistRange *)gctx->d_BufferInput, d_pairs, gctx->d_points, gctx->d_idealoffset, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
        cudaDeviceSynchronize();
        check_execution("kernel_unroll");
        timer.stopTimer();
        printf("kernel unroll: %f ms\n", timer.getElapsedTime());

        std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);
        std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

        grid_size_x = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        block_size.x = BLOCK_SIZE;
        grid_size.x = grid_size_x;

        timer.startTimer();

        kernel_refine<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, d_pairs, gctx->d_points, gctx->d_vertices, gctx->d_bufferinput_size, gctx->d_distance, gctx->d_degree_degree_per_kilometer_latitude, gctx->degree_per_kilometer_longitude_arr);
        cudaDeviceSynchronize();
        check_execution("kernel_refine");

        timer.stopTimer();
        printf("kernel refine: %f ms\n", timer.getElapsedTime());

        double *h_distance = new double[size * sizeof(double)];
        CUDA_SAFE_CALL(cudaMemcpy(h_distance, gctx->d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
    
        for (int i = 0; i < size; i++)
        {
            if (h_distance[i] <= WITHIN_DISTANCE)
                found++;
        }

        CUDA_SAFE_CALL(cudaFree(d_pairs));

        delete []h_pairs;
        delete []h_distance;

    }

	return found;
}