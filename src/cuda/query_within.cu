#include "geometry.cuh"

#define WITHIN_DISTANCE 10

struct Batch
{
	uint start = 0;
	uint length = 0;
	int pair_id = 0;
};

struct test
{
	int pair_id;
	unsigned short pix_id1;
	unsigned short pix_id2;
};

__global__ void kernel_init(pair<Point, IdealOffset> *d_pairs, Idealinfo *d_info, uint size, double *distance)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		pair<Point, IdealOffset> &pair = d_pairs[x];
		IdealOffset &source = pair.second;
		Point &p = pair.first;
		box &s_mbr = d_info[source.info_start].mbr;

		distance[x] = gpu_max_distance(p, s_mbr);
		// printf("DIST: %lf\n", dist);
	}
}

__global__ void kernel_filter(pair<Point, IdealOffset> *d_pairs, Idealinfo *d_info, uint8_t *d_status, PixMapping *d_ptpixpairs, uint *d_pp_size, uint size, int *step, bool *resultmap)
{
	const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair_id < size)
	{
		if (resultmap[pair_id] != 0)
			return;
		pair<Point, IdealOffset> &pair = d_pairs[pair_id];
		IdealOffset &source = pair.second;
		Point &p = pair.first;

		box &s_mbr = d_info[source.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
		int closest = gpu_get_closest_pixel(xoff, yoff, s_dimx, s_dimy);

		if (*step == 0)
		{
			if (gpu_show_status(d_status, source.status_start, closest) == BORDER)
			{
				int idx = atomicAdd(d_pp_size, 1);
				d_ptpixpairs[idx] = {pair_id, closest};
			}
		}
		else
		{
			int closest_x = gpu_get_x(closest, s_dimx);
			int closest_y = gpu_get_y(closest, s_dimx, s_dimy);
			int ymin = max(0, closest_y - *step);
			int ymax = min(s_dimy, closest_y + *step);

			// left scan
			if (closest_x - *step >= 0)
			{
				for (int y = ymin; y <= ymax; y++)
				{
					int id = gpu_get_id(closest_x - *step, y, s_dimx);
					if (gpu_show_status(d_status, source.status_start, id) == BORDER)
					{
						int idx = atomicAdd(d_pp_size, 1);
						d_ptpixpairs[idx] = {pair_id, id};
					}
				}
			}
			// right scan
			if (closest_x + *step <= s_dimx)
			{
				for (int y = ymin; y <= ymax; y++)
				{
					int id = gpu_get_id(closest_x + *step, y, s_dimx);
					if (gpu_show_status(d_status, source.status_start, id) == BORDER)
					{
						int idx = atomicAdd(d_pp_size, 1);
						d_ptpixpairs[idx] = {pair_id, id};
					}
				}
			}
			// skip the first if there is left scan
			int xmin = max(0, closest_x - *step + (closest_x - *step >= 0));
			// skip the last if there is right scan
			int xmax = min(s_dimx, closest_x + *step - (closest_x + *step <= s_dimx));

			// bottom scan
			if (closest_y - *step >= 0)
			{
				for (int x = xmin; x <= xmax; x++)
				{
					int id = gpu_get_id(x, closest_y - *step, s_dimx);
					if (gpu_show_status(d_status, source.status_start, id) == BORDER)
					{
						int idx = atomicAdd(d_pp_size, 1);
						d_ptpixpairs[idx] = {pair_id, id};
					}
				}
			}
			// top scan
			if (closest_y + *step <= s_dimy)
			{
				for (int x = xmin; x <= xmax; x++)
				{
					int id = gpu_get_id(x, closest_y + *step, s_dimx);
					if (gpu_show_status(d_status, source.status_start, id) == BORDER)
					{
						int idx = atomicAdd(d_pp_size, 1);
						d_ptpixpairs[idx] = {pair_id, id};
					}
				}
			}
		}
	}
}

__global__ void kernel_unroll(PixMapping *d_ptpixpairs, pair<Point, IdealOffset> *d_pairs, uint16_t *d_offset, EdgeSeq *d_edge_sequences, uint *size, Batch *d_batch, uint *d_batch_size)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		int pair_id = d_ptpixpairs[x].pair_id;
		int cur = d_ptpixpairs[x].pix_id;

		pair<Point, IdealOffset> &pair = d_pairs[pair_id];
		IdealOffset &source = pair.second;

		int s_num_sequence = (d_offset + source.offset_start)[cur + 1] - (d_offset + source.offset_start)[cur];

		for (int i = 0; i < s_num_sequence; i++)
		{
			EdgeSeq r = (d_edge_sequences + source.edge_sequences_start)[(d_offset + source.offset_start)[cur] + i];

			int max_size = 32;
			for (int j = 0; j < r.length; j += max_size)
			{
				uint end = min(j + max_size, r.length);
				int idx = atomicAdd(d_batch_size, 1U);
				d_batch[idx] = {r.start + j, end - j, pair_id};
			}
		}
	}
}

__global__ void kernel_refinement(Batch *d_batch, pair<Point, IdealOffset> *d_pairs, Point *d_vertices, uint *size, double *distance, bool *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < *size)
	{
		uint s = d_batch[x].start;
		uint len = d_batch[x].length;
		int pair_id = d_batch[x].pair_id;
		if (resultmap[pair_id])
			return;

		pair<Point, IdealOffset> &pair = d_pairs[pair_id];
		IdealOffset &source = pair.second;
		Point &p = pair.first;

		for (int i = 0; i < len; i++)
		{
			double dist = gpu_point_to_segment_distance(p, (d_vertices + source.vertices_start)[s + i], (d_vertices + source.vertices_start)[s + i + 1]);
			atomicMinDouble(distance + pair_id, dist);
			if (distance[pair_id] <= WITHIN_DISTANCE)
			{
				resultmap[pair_id] = true;
				return;
			}
		}
	}
}

__global__ void kernel_check_exit(pair<Point, IdealOffset> *d_pairs, Idealinfo *d_info, uint8_t *d_status, PixMapping *d_ptpixpairs, uint *d_pp_size, uint size, int *step, double *distance, bool *resultmap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size)
	{
		if (resultmap[x])
			return;
		pair<Point, IdealOffset> &pair = d_pairs[x];
		IdealOffset &source = pair.second;
		Point &p = pair.first;

		box &s_mbr = d_info[source.info_start].mbr;
		const double &s_step_x = d_info[source.info_start].step_x, &s_step_y = d_info[source.info_start].step_y;
		const int &s_dimx = d_info[source.info_start].dimx, &s_dimy = d_info[source.info_start].dimy;

		int xoff = gpu_get_offset_x(s_mbr.low[0], p.x, s_step_x, s_dimx);
		int yoff = gpu_get_offset_y(s_mbr.low[1], p.y, s_step_y, s_dimy);
		int closest = gpu_get_closest_pixel(xoff, yoff, s_dimx, s_dimy);

		double mindist = DBL_MAX;

		int closest_x = gpu_get_x(closest, s_dimx);
		int closest_y = gpu_get_y(closest, s_dimx, s_dimy);
		int ymin = max(0, closest_y - *step);
		int ymax = min(s_dimy, closest_y + *step);

		// left scan
		if (closest_x - *step >= 0)
		{
			double x = gpu_get_pixel_box(closest_x - *step, ymin, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[0];
			double y1 = gpu_get_pixel_box(closest_x - *step, ymin, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[1];
			double y2 = gpu_get_pixel_box(closest_x - *step, ymax, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[1];

			Point p1 = Point(x, y1);
			Point p2 = Point(x, y2);
			mindist = min(mindist, gpu_point_to_segment_distance(p, p1, p2));
		}
		// right scan
		if (closest_x + *step <= s_dimx)
		{
			double x = gpu_get_pixel_box(closest_x + *step, ymin, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[0];
			double y1 = gpu_get_pixel_box(closest_x + *step, ymin, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[1];
			double y2 = gpu_get_pixel_box(closest_x + *step, ymax, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[1];

			Point p1 = Point(x, y1);
			Point p2 = Point(x, y2);
			mindist = min(mindist, gpu_point_to_segment_distance(p, p1, p2));
		}

		// skip the first if there is left scan
		int xmin = max(0, closest_x - *step + (closest_x - *step >= 0));
		// skip the last if there is right scan
		int xmax = min(s_dimx, closest_x + *step - (closest_x + *step <= s_dimx));
		// bottom scan
		if (closest_y - *step >= 0)
		{
			double y = gpu_get_pixel_box(xmin, closest_y - *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[1];
			double x1 = gpu_get_pixel_box(xmin, closest_y - *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[0];
			double x2 = gpu_get_pixel_box(xmax, closest_y - *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[0];

			Point p1 = Point(x1, y);
			Point p2 = Point(x2, y);
			mindist = min(mindist, gpu_point_to_segment_distance(p, p1, p2));
		}
		// top scan
		if (closest_y + *step <= s_dimy)
		{
			double y = gpu_get_pixel_box(xmin, closest_y + *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[1];
			double x1 = gpu_get_pixel_box(xmin, closest_y + *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).low[0];
			double x2 = gpu_get_pixel_box(xmax, closest_y + *step, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y).high[0];

			Point p1 = Point(x1, y);
			Point p2 = Point(x2, y);
			mindist = min(mindist, gpu_point_to_segment_distance(p, p1, p2));
		}

		if (distance[x] < mindist)
		{
			resultmap[x] = true;
			return;
		}
	}
}

uint cuda_within(query_context *gctx)
{
	CudaTimer timer;

	float sum_filter = 0.0;
	float sum_unroll = 0.0;
	float sum_refinement = 0.0;
	float sum_check = 0.0;

	size_t size = gctx->point_polygon_pairs.size();

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

	double *h_distance = new double[size * sizeof(double)];
	double *d_distance = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_distance, size * sizeof(double)));

	bool *d_resultmap = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_resultmap, size * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemset(d_resultmap, 0, size * sizeof(bool)));

	PixMapping *d_ptpixpairs = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_ptpixpairs, 64 * 1024 * 1024 * sizeof(PixMapping)));
	uint h_pp_size;
	uint *d_pp_size = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_pp_size, sizeof(uint)));

	int h_step = 0;
	int *d_step = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_step, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_step, 0, sizeof(int)));

	Batch *d_batch = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_batch, 32 * 1024 * 1024 * sizeof(Batch)));
	uint *d_batch_size = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_batch_size, sizeof(uint)));
	uint h_batch_size = 0;

	int grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 block_size(BLOCK_SIZE, 1, 1);
	dim3 grid_size(grid_size_x, 1, 1);

	timer.startTimer();

	kernel_init<<<grid_size, block_size>>>(d_pairs, gctx->d_info, size, d_distance);
	cudaDeviceSynchronize();
	check_execution("kernel init");

	timer.stopTimer();
	printf("distance initialize time: %f ms\n", timer.getElapsedTime());

	while (true)
	{
		printf("STEP: %d\n", h_step);
		CUDA_SAFE_CALL(cudaMemset(d_pp_size, 0, sizeof(uint)));
		CUDA_SAFE_CALL(cudaMemset(d_batch_size, 0, sizeof(uint)));

		timer.startTimer();

		kernel_filter<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, d_ptpixpairs, d_pp_size, size, d_step, d_resultmap);
		cudaDeviceSynchronize();
		check_execution("Kernel filter");

		timer.stopTimer();
		printf("kernel filter time: %f ms\n", timer.getElapsedTime());
		sum_filter += timer.getElapsedTime();

		CUDA_SAFE_CALL(cudaMemcpy(&h_pp_size, d_pp_size, sizeof(uint), cudaMemcpyDeviceToHost));

		// printf("POINT PIXEL PAIRS SIZE: %u\n", h_pp_size);
		if (h_pp_size == 0)
			break;

		grid_size_x = (h_pp_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		grid_size.x = grid_size_x;

		timer.startTimer();

		kernel_unroll<<<grid_size, block_size>>>(d_ptpixpairs, d_pairs, gctx->d_offset, gctx->d_edge_sequences, d_pp_size, d_batch, d_batch_size);
		cudaDeviceSynchronize();
		check_execution("Kernel_refinement");

		timer.stopTimer();
		printf("kernel unroll time: %f ms\n", timer.getElapsedTime());
		sum_unroll += timer.getElapsedTime();

		CUDA_SAFE_CALL(cudaMemcpy(&h_batch_size, d_batch_size, sizeof(uint), cudaMemcpyDeviceToHost));

		grid_size_x = (h_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		grid_size.x = grid_size_x;

		timer.startTimer();

		kernel_refinement<<<grid_size, block_size>>>(d_batch, d_pairs, gctx->d_vertices, d_batch_size, d_distance, d_resultmap);
		cudaDeviceSynchronize();
		check_execution("Kernel_refinement");

		timer.stopTimer();
		printf("kernel refinement time: %f ms\n", timer.getElapsedTime());
		sum_refinement += timer.getElapsedTime();

		// if(h_step == 3) break;
		h_step++;
		CUDA_SAFE_CALL(cudaMemcpy(d_step, &h_step, sizeof(int), cudaMemcpyHostToDevice));

		grid_size_x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		grid_size.x = grid_size_x;

		timer.startTimer();

		kernel_check_exit<<<grid_size, block_size>>>(d_pairs, gctx->d_info, gctx->d_status, d_ptpixpairs, d_pp_size, size, d_step, d_distance, d_resultmap);
		cudaDeviceSynchronize();
		check_execution("Kernel check exit");

		timer.stopTimer();
		printf("kernel check exit time: %f ms\n", timer.getElapsedTime());
		sum_check += timer.getElapsedTime();
	}

	printf("kernel filter time: %f ms\n", sum_filter);
	printf("kernel unroll time: %f ms\n", sum_unroll);
	printf("kernel refinment time: %f ms\n", sum_refinement);
	printf("kernel check exit time: %f ms\n", sum_check);

	printf("average kernel filter time: %f ms\n", sum_filter / h_step);
	printf("average kernel unroll time: %f ms\n", sum_unroll / h_step);
	printf("average kernel refinment time: %f ms\n", sum_refinement / h_step);
	printf("average kernel check exit time: %f ms\n", sum_check / h_step);

	CUDA_SAFE_CALL(cudaMemcpy(h_distance, d_distance, size * sizeof(double), cudaMemcpyDeviceToHost));
	int found = 0;
	for (int i = 0; i < size; i++)
	{
		if (h_distance[i] <= WITHIN_DISTANCE)
			found++;
	}

	return found;
}