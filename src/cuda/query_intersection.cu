#include "geometry.cuh"
#include "Ideal.h"
#include "shared_kernels.cuh" 
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>
#include <thrust/unique.h>
#include <thrust/count.h>

struct Task
{
    uint s_start = 0;
    uint t_start = 0;
    uint s_length = 0;
    uint t_length = 0;
    int pair_id = 0;
};

// flags: 0(not contain), 1(maybe contain), 2(contain)
__global__ void kernel_filter_intersection(pair<uint32_t,uint32_t>* pairs, IdealOffset *idealoffset,
                                             RasterInfo *info, uint8_t *status, uint size, 
                                             PixPair *pixpairs, uint *pp_size, uint8_t category_count)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;  

	const pair<uint32_t, uint32_t> pair = pairs[x];
    const uint32_t src_idx = pair.first;
	const uint32_t tar_idx = pair.second;
	const IdealOffset source = idealoffset[src_idx];
	const IdealOffset target = idealoffset[tar_idx];

	const box s_mbr = info[src_idx].mbr, t_mbr = info[tar_idx].mbr;				
	const double s_step_x = info[src_idx].step_x, s_step_y = info[src_idx].step_y; 
	const int s_dimx = info[src_idx].dimx, s_dimy = info[src_idx].dimy;			 
	const double t_step_x = info[tar_idx].step_x, t_step_y = info[tar_idx].step_y; 
	const int t_dimx = info[tar_idx].dimx, t_dimy = info[tar_idx].dimy;		

	int i_min = gpu_get_offset_x(s_mbr.low[0], t_mbr.low[0] + 1e-6, s_step_x, s_dimx);
	int i_max = gpu_get_offset_x(s_mbr.low[0], t_mbr.high[0] - 1e-6, s_step_x, s_dimx);
	int j_min = gpu_get_offset_y(s_mbr.low[1], t_mbr.low[1] + 1e-6, s_step_y, s_dimy);
	int j_max = gpu_get_offset_y(s_mbr.low[1], t_mbr.high[1] - 1e-6, s_step_y, s_dimy);
	
	for (int i = i_min; i <= i_max; i++)
	{
		for (int j = j_min; j <= j_max; j++)
		{
			int p = gpu_get_id(i, j, s_dimx);
			PartitionStatus source_status = gpu_show_status(status, source.status_start, p, category_count);
            if(source_status != BORDER) continue;

			box bx = gpu_get_pixel_box(i, j, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

            bx.low[0] += 1e-6;
            bx.low[1] += 1e-6;
            bx.high[0] -= 1e-6;
            bx.high[1] -= 1e-6;

			int _i_min = gpu_get_offset_x(t_mbr.low[0], bx.low[0], t_step_x, t_dimx);
			int _i_max = gpu_get_offset_x(t_mbr.low[0], bx.high[0], t_step_x, t_dimx);
			int _j_min = gpu_get_offset_y(t_mbr.low[1], bx.low[1], t_step_y, t_dimy);
			int _j_max = gpu_get_offset_y(t_mbr.low[1], bx.high[1], t_step_y, t_dimy);

			for (int _i = _i_min; _i <= _i_max; _i++)
			{
				for (int _j = _j_min; _j <= _j_max; _j++)
				{
					int p2 = gpu_get_id(_i, _j, t_dimx);

					PartitionStatus target_status = gpu_show_status(status, target.status_start, p2, category_count);

					if (target_status == BORDER)
					{
						int idx = atomicAdd(pp_size, 1U);
						
                        pixpairs[idx] = {p, p2, x};
					}
				}
			}
		}
	}
    return;
}

__global__ void kernel_unroll_intersection(PixPair *pixpairs, pair<uint32_t, uint32_t> *pairs,
											 IdealOffset *idealoffset, uint8_t *status,
											 uint32_t *es_offset, EdgeSeq *edge_sequences,
											 uint *size, Task *tasks, uint *task_size)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= *size) return;
	
	int p = pixpairs[idx].source_pixid;
	int p2 = pixpairs[idx].target_pixid;
	int pair_id = pixpairs[idx].pair_id;

	const pair<uint32_t, uint32_t> pair = pairs[pair_id];
    const uint32_t src_idx = pair.first;
    const uint32_t tar_idx = pair.second;

	const IdealOffset source = idealoffset[src_idx];
    const IdealOffset target = idealoffset[tar_idx];

	uint s_offset_start = source.offset_start;
	uint t_offset_start = target.offset_start;
	uint s_edge_sequences_start = source.edge_sequences_start;
	uint t_edge_sequences_start = target.edge_sequences_start;

	int s_num_sequence = (es_offset + s_offset_start)[p + 1] - (es_offset + s_offset_start)[p];
	int t_num_sequence = (es_offset + t_offset_start)[p2 + 1] - (es_offset + t_offset_start)[p2];

	// printf("(es_offset + t_offset_start)[p2] = %u, (es_offset + t_offset_start)[p2 + 1] = %u\n", (es_offset + t_offset_start)[p2], (es_offset + t_offset_start)[p2 + 1]);
	uint s_vertices_start = source.vertices_start;
	uint t_vertices_start = target.vertices_start;

	const int max_size = 16;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		EdgeSeq r = (edge_sequences + s_edge_sequences_start)[(es_offset + s_offset_start)[p] + i];
		for (int j = 0; j < t_num_sequence; ++j)
		{
	 		EdgeSeq r2 = (edge_sequences + t_edge_sequences_start)[(es_offset + t_offset_start)[p2] + j];
			// printf("r.length = %d, r2.length = %d\n", r.length, r2.length);
			for (uint s = 0; s < r.length; s += max_size)
			{
				uint end_s = min(s + max_size, r.length);
				for (uint t = 0; t < r2.length; t += max_size)
				{
					uint end_t = min(t + max_size, r2.length);

					uint idx_task = atomicAdd(task_size, 1U);
					tasks[idx_task].s_start = s_vertices_start + r.start + s;
					tasks[idx_task].t_start = t_vertices_start + r2.start + t;
					tasks[idx_task].s_length = end_s - s;
					tasks[idx_task].t_length = end_t - t;
					tasks[idx_task].pair_id = pair_id;
	 			}
	 		}
		}
	}
}

__global__ void kernel_refinement_intersection(Task *tasks, Point *d_vertices, uint *size, Intersection* intersections, uint* num)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size) return;
	
	uint s1 = tasks[x].s_start;
	uint s2 = tasks[x].t_start;
	uint len1 = tasks[x].s_length;
	uint len2 = tasks[x].t_length;
	int pair_id = tasks[x].pair_id;

	gpu_segment_intersect_batch(d_vertices, s1, s2, s1 + len1, s2 + len2, pair_id, intersections, num);
}

struct CompareSourceIntersections {
	__host__ __device__
	bool operator()(const Intersection& a, const Intersection& b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.edge_source_id != b.edge_source_id){
			return a.edge_source_id < b.edge_source_id;
		}
        return a.t < b.t; 
    }
};

struct CompareTargetIntersections {
    __host__ __device__
    bool operator()(const Intersection& a, const Intersection& b) const {
        if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.edge_target_id != b.edge_target_id){
			return a.edge_target_id < b.edge_target_id;
		}
        return a.u < b.u; 
    }
};

struct IntersectionEqual {
    __host__ __device__
    bool operator()(const Intersection& a, const Intersection& b) const {
        // return a.pair_id == b.pair_id && a.edge_source_id == b.edge_source_id && a.edge_target_id == b.edge_target_id && a.t == b.t && a.u == b.u;
		return a.p == b.p;
	}
};


__global__ void find_inters_per_pair(Intersection *intersections, uint* size, uint *inters_per_pair){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < *size)
    {
        int pair_id = intersections[x].pair_id;
        atomicAdd(inters_per_pair + pair_id, 1);
    }
}

__global__ void make_segments(Intersection *intersections, uint *num_intersections, Segment *segments, uint *num_segments, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, Point *vertices, uint *inters_per_pair, bool is_source)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < *num_intersections){
		int pair_id = intersections[x].pair_id;
        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        const uint32_t id = is_source ? pair.first : pair.second;

        Intersection a = intersections[x];
        Intersection b = (x + 1 >= *num_intersections || intersections[x].pair_id != intersections[x + 1].pair_id) ? intersections[x + 1 - inters_per_pair[pair_id]] : intersections[x + 1];

        if(a.p == b.p) return;

        int a_edge_id = is_source ? a.edge_source_id : a.edge_target_id;
        int b_edge_id = is_source ? b.edge_source_id : b.edge_target_id;
        float a_param = is_source ? a.t : a.u;
        float b_param = is_source ? b.t : b.u;
        int num_vertices = idealoffset[id + 1].vertices_start - idealoffset[id].vertices_start;

        if(fabs(a_param - 1.0) < eps){
			a_edge_id = (a_edge_id + 1) < (idealoffset[id + 1].vertices_start - 1) ? a_edge_id + 1 : a_edge_id + 2 - num_vertices;
			a_param = 0.0;
        } 

        if(fabs(b_param) < eps){
            b_edge_id--;
			b_param = 1.0;
        }

        // if(x == 20 && pair_id == 11){
        //     printf("range %d %d\n", idealoffset[id].vertices_start, idealoffset[id + 1].vertices_start);
        // }
        // if(pair_id == 11){
        //     printf("%d %d %lf %lf\n", a_edge_id, b_edge_id, a_param, b_param);
        // }

        int idx = atomicAdd(num_segments, 1);
        segments[idx] = {is_source, a.p, b.p, 
                        (a_edge_id == b_edge_id) && (a_param < b_param) ? -1 : a_edge_id + 1,
                        (a_edge_id == b_edge_id) && (a_param < b_param) ? -1 : b_edge_id, 
                        // (a_edge_id == b_edge_id && a_param < b_param),
                        // (a_edge_id == b_edge_id && a_param < b_param), 
                        pair_id};

	}
}

struct ExtractPairId {
    __host__ __device__
    int operator()(const Segment& seg) const {
        return seg.pair_id;
    }
};

__global__ void kernel_filter_segment_contain(Segment *segments, pair<uint32_t,uint32_t> *pairs,
											  IdealOffset *idealoffset, RasterInfo *info, 
											  uint8_t *status, Point *vertices, uint size, uint8_t *flags, 
											  PixMapping *ptpixpairs, uint *pp_size, uint8_t category_count)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size) return;

	Segment seg = segments[x];
	const pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
	uint32_t poly_idx = !seg.is_source ? pair.first : pair.second;
	const IdealOffset offset = idealoffset[poly_idx];
	
	Point p;
	if(seg.edge_start == -1) p = (seg.start + seg.end) * 0.5;
	else p = vertices[seg.edge_start];
	
	const box mbr = info[poly_idx].mbr;
	const double step_x = info[poly_idx].step_x;
	const double step_y = info[poly_idx].step_y;
	const int dimx = info[poly_idx].dimx;
	const int dimy = info[poly_idx].dimy;
	
	const int xoff = gpu_get_offset_x(mbr.low[0], p.x, step_x, dimx);
	const int yoff = gpu_get_offset_y(mbr.low[1], p.y, step_y, dimy);
	const int target = gpu_get_id(xoff, yoff, dimx);

	const PartitionStatus st = gpu_show_status(status, offset.status_start, target, category_count);
	
    if (st == BORDER) {
		uint idx = atomicAdd(pp_size, 1U);
		ptpixpairs[idx].pair_id = x;
		ptpixpairs[idx].pix_id = target;
	}else{
        flags[x] = st;
    }
}

__global__ void kernel_refinement_segment_contain(PixMapping *ptpixpairs, Segment *segments, 
												pair<uint32_t, uint32_t> *pairs,
												IdealOffset *idealoffset, RasterInfo *info,
												uint32_t *es_offset, EdgeSeq *edge_sequences,
												Point *vertices, uint32_t *gridline_offset,
												double *gridline_nodes, uint *size, uint8_t *flags)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= *size)
		return;

	int seg_id = ptpixpairs[x].pair_id;
	int target = ptpixpairs[x].pix_id;
	
	Segment seg = segments[seg_id];
	const pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
	const uint32_t poly_idx = !seg.is_source ? pair.first : pair.second;
	const IdealOffset offset = idealoffset[poly_idx];
	
	Point p;
	if(seg.edge_start == -1) p = (seg.start + seg.end) * 0.5;
	else p = vertices[seg.edge_start];

	const box s_mbr = info[poly_idx].mbr;
	const double s_step_x = info[poly_idx].step_x;
	const double s_step_y = info[poly_idx].step_y;
	const int s_dimx = info[poly_idx].dimx;
	const int s_dimy = info[poly_idx].dimy;

	bool ret = false;

	const int xoff = gpu_get_x(target, s_dimx);
	const int yoff = gpu_get_y(target, s_dimx, s_dimy);
	const box bx = gpu_get_pixel_box(xoff, yoff, s_mbr.low[0], s_mbr.low[1], s_step_x, s_step_y);

	// printf("POINT (%lf %lf)\tLINESTRING((%lf %lf, %lf %lf, %lf %lf, %lf %lf, %lf %lf))\n", p.x, p.y, bx.low[0], bx.low[1], bx.high[0], bx.low[1], bx.high[0], bx.high[1], bx.low[0], bx.high[1], bx.low[0], bx.low[1]);

	const uint32_t offset_start = offset.offset_start;
	const uint32_t es_start = (es_offset + offset_start)[target];
	const uint32_t es_end = (es_offset + offset_start)[target + 1];
	const int s_num_sequence = es_end - es_start;

	for (int i = 0; i < s_num_sequence; ++i)
	{
		const EdgeSeq r = (edge_sequences + offset.edge_sequences_start)[es_start + i];
		const uint32_t vertices_start = offset.vertices_start;
		for (int j = 0; j < r.length; j++)
		{
			const Point v1 = (vertices + vertices_start)[r.start + j];
			const Point v2 = (vertices + vertices_start)[r.start + j + 1];
			if(p == v1 || p == v2){
				flags[seg_id] = 1;
				return;  // p在边界上
			}
			if ((v1.y >= p.y) != (v2.y >= p.y))
			{

				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (abs(dy) > 1e-9)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if(fabs(p.x - int_x) < 1e-9) {
						flags[seg_id] = 1;
						return;  // p在边界上
					}
					if (p.x < int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}else if (v1.y == p.y && v2.y == p.y && (v1.x >= p.x) != (v2.x >= p.x)){
                flags[seg_id] = 1;
                return;   // p在边界上
            }
		}
	}

	int nc = 0;
	const uint32_t gridline_start = offset.gridline_offset_start;
	const uint32_t i_start = (gridline_offset + gridline_start)[xoff + 1];
	const uint32_t i_end = (gridline_offset + gridline_start)[xoff + 2];

	nc = binary_search_count((gridline_nodes + offset.gridline_nodes_start), i_start, i_end, p.y);

    if(nc % 2 == 1){
        ret = !ret;
    }

    if(ret){
        flags[seg_id] = 2;
    }else{
        flags[seg_id] = 0;
    }	
}

__global__ void rebuild_polygons(Segment* segments, uint8_t* status, size_t size, pair<uint32_t, uint32_t> *pairs, IdealOffset *idealoffset, Point* vertices, int* offsets, double* area){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < size){
		int start_idx = offsets[x];
		int end_idx = offsets[x + 1];
        int num_segments = offsets[size];

        bool foundCycle = false;

		// printf("x = %d %d %d\n", x, start_idx, end_idx);
		for(int i = start_idx; i < end_idx; i ++){
            if (i < num_segments - 1 && (status[i] == 0 || status[i] == 1))
                continue;
                
            if (i == num_segments - 1 && status[i] == 0)
                continue;

            if (i == num_segments - 1 && status[i] == 2 && foundCycle){
                continue;
            }

			Segment seg = segments[i];
			// 当前segment和端点
			size_t currentSegIdx = i;
			Point currentPoint = seg.start;
			Point lastPoint = currentPoint;
			Point startPoint = currentPoint;
			// printf("START POINT (%lf %lf) %d\n", currentPoint.x, currentPoint.y, seg.pair_id);

			double a = 0.0f;
			double b = 0.0f;

			foundCycle = false;

			while(status[currentSegIdx]){
				status[currentSegIdx] = 0;
				const Segment& seg = segments[currentSegIdx];
				if(seg.edge_start != -1){
                	if(seg.edge_start <= seg.edge_end){
						for(int verId = seg.edge_start; verId <= seg.edge_end; verId ++){
							a += lastPoint.x * vertices[verId].y;
							b += lastPoint.y * vertices[verId].x;
							lastPoint = vertices[verId];
							// printf("POINT (%lf %lf) %lf %lf %d\n", vertices[verId].x, vertices[verId].y, a, b, seg.pair_id);
						}
                	}else{
						pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
						uint32_t offset_idx = seg.is_source ? pair.first : pair.second;
						for(int verId = seg.edge_start; verId < idealoffset[offset_idx + 1].vertices_start - 1; verId ++){
							a += lastPoint.x * vertices[verId].y;
							b += lastPoint.y * vertices[verId].x;
							lastPoint = vertices[verId];
							// printf("POINT (%lf %lf) %lf %lf %d\n", vertices[verId].x, vertices[verId].y, a, b, seg.pair_id);
						}
						for(int verId = idealoffset[offset_idx].vertices_start; verId <= seg.edge_end; verId ++){
							
							a += lastPoint.x * vertices[verId].y;
							b += lastPoint.y * vertices[verId].x;
							lastPoint = vertices[verId];
							// printf("POINT (%lf %lf) %lf %lf %d\n", vertices[verId].x, vertices[verId].y, a, b, seg.pair_id);
						}
					}
				}

				// 确定segment的另一个端点
            	Point nextPoint = currentPoint == seg.start ? seg.end : seg.start;

				// 如果回到起点，我们找到了一个闭合的多边形
				if (nextPoint == startPoint) {
					// printf("lastPoint: POINT(%lf %lf) startPoint: POINT(%lf %lf)\n", lastPoint.x, lastPoint.y, startPoint.x, startPoint.y);
					a += lastPoint.x * startPoint.y;
					b += lastPoint.y * startPoint.x;
					// printf("POINT (%lf %lf) %lf %lf %d\n", startPoint.x, startPoint.y, a, b, seg.pair_id);
					foundCycle = true;
					break;
				}

				// 寻找连接到nextPoint的未使用segment
				bool foundNext = false;

				int idx = binary_search(segments, start_idx, end_idx - 1, nextPoint);
				if(idx != -1){
					uint8_t st0 = status[idx], st1 = status[idx + 1];
					if(st0 == 2 || st1 == 2){
						currentSegIdx = (st0 == 2) ? idx : idx + 1;
					}else if(st0 == 1 || st1 == 1){
						currentSegIdx = (st0 == 1) ? idx : idx + 1;
					} 
                    currentPoint = nextPoint;
                    foundNext = true;
				}

				// // 如果回到起点，我们找到了一个闭合的多边形
				// if (!foundNext && nextPoint == startPoint) {
				// 	lastPoint = currentPoint;
				// 	a += lastPoint.x * startPoint.y;
				// 	b += lastPoint.y * startPoint.x;
				// 	foundCycle = true;
				// 	break;
				// }
				
				if (!foundNext) break;

				a += lastPoint.x * currentPoint.y;
				b += lastPoint.y * currentPoint.x;
				lastPoint = currentPoint;
				// printf("POINT (%lf %lf) %lf %lf %d\n", currentPoint.x, currentPoint.y, a, b, seg.pair_id);
			}
			if (foundCycle) {
				atomicAdd(area, 0.5 * fabs(a - b));
				// printf("area: %lf\n", 0.5 * fabs(a - b));
			} 			
		}		
	}
}

void cuda_intersection(query_context *gctx)
{
	size_t batch_size = gctx->index_end - gctx->index;
    uint h_bufferinput_size, h_bufferoutput_size;
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferinput_size, 0, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));

	/*1. Raster Model Filtering*/
    const int block_size = BLOCK_SIZE;
    int grid_size = (batch_size + block_size - 1) / block_size;

	auto filter_start = std::chrono::high_resolution_clock::now();
    kernel_filter_intersection<<<grid_size, block_size>>>(gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, gctx->d_status, batch_size, (PixPair *)gctx->d_BufferInput, gctx->d_bufferinput_size, gctx->category_count);
    cudaDeviceSynchronize();
    check_execution("kernel_filter_intersection");
	auto filter_end = std::chrono::high_resolution_clock::now();
	auto filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_end - filter_start);
	std::cout << "filter time: " << filter_duration.count() << " ms" << std::endl;

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferinput_size);

	if(h_bufferinput_size == 0) return;
	
    /*2. Unroll Refinement*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	auto unroll_start = std::chrono::high_resolution_clock::now();
    kernel_unroll_intersection<<<grid_size, block_size>>>((PixPair *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_status, gctx->d_offset, gctx->d_edge_sequences, gctx->d_bufferinput_size, (Task *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_unroll_intersection");
	auto unroll_end = std::chrono::high_resolution_clock::now();
	auto unroll_duration = std::chrono::duration_cast<std::chrono::milliseconds>(unroll_end - unroll_start);
	std::cout << "unroll time: " << unroll_duration.count() << " ms" << std::endl;

    CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferoutput_size);

  	CUDA_SWAP_BUFFER();
  
    /*3. Refinement step*/

    grid_size = (h_bufferinput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	auto refine_start = std::chrono::high_resolution_clock::now();
    kernel_refinement_intersection<<<grid_size, block_size>>>((Task *)gctx->d_BufferInput, gctx->d_vertices, gctx->d_bufferinput_size, (Intersection *)gctx->d_BufferOutput, gctx->d_bufferoutput_size);
    cudaDeviceSynchronize();
    check_execution("kernel_refinement_intersection");
	auto refine_end = std::chrono::high_resolution_clock::now();
	auto refine_duration = std::chrono::duration_cast<std::chrono::milliseconds>(refine_end - refine_start);
	std::cout << "refine time: " << refine_duration.count() << " ms" << std::endl;

	CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("h_buffer_size = %u\n", h_bufferoutput_size);

	CUDA_SWAP_BUFFER();

	if(h_bufferinput_size == 0) return;

	auto segment_start = std::chrono::high_resolution_clock::now();

	// check source polygon edges 
    thrust::device_ptr<Intersection> begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    thrust::device_ptr<Intersection> end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);

	// 排序并去重
	thrust::sort(thrust::device, begin, end, 
    [] __device__(const Intersection &a, const Intersection &b) {
		if (a.pair_id != b.pair_id) {
            return a.pair_id < b.pair_id; 
        }else if (a.p.x != b.p.x){
            return a.p.x < b.p.x;
        }
        return a.p.y < b.p.y;
    });

	auto new_end = thrust::unique(begin, end,
    [] __device__(const Intersection &a, const Intersection &b) {
        return a.pair_id == b.pair_id && a.p == b.p;
    });

    h_bufferinput_size = thrust::distance(begin, new_end);
	end = begin + h_bufferinput_size;

	auto &num_intersections = h_bufferinput_size;

	// 更新设备内存
	CUDA_SAFE_CALL(cudaMemcpy(gctx->d_bufferinput_size, &h_bufferinput_size, 
							sizeof(uint), cudaMemcpyHostToDevice));

	// end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + h_bufferinput_size);
    // thrust::sort(thrust::device, begin, end, CompareSourceIntersections());

	// PrintBuffer((Intersection *)gctx->d_BufferInput, num_intersections);

	grid_size = (num_intersections + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint *d_inters_per_pair = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_inters_per_pair, batch_size * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMemset(d_inters_per_pair, 0, batch_size * sizeof(uint)));

	find_inters_per_pair<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, d_inters_per_pair);
	cudaDeviceSynchronize();
    check_execution("kernel_find_inters_per_pair");

	thrust::sort(thrust::device, begin, end, CompareSourceIntersections());
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_vertices, d_inters_per_pair, true);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	// check target polygon edges 

	begin = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput);
    end = thrust::device_pointer_cast((Intersection*)gctx->d_BufferInput + num_intersections);
    thrust::sort(thrust::device, begin, end, CompareTargetIntersections());
	
	make_segments<<<grid_size, block_size>>>((Intersection *)gctx->d_BufferInput, gctx->d_bufferinput_size, (Segment *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_vertices, d_inters_per_pair, false);
	cudaDeviceSynchronize();
    check_execution("kernel_make_segments");

	CUDA_SWAP_BUFFER();
			
	uint &num_segments = h_bufferinput_size;

	printf("num_segqments = %d\n", num_segments);

	if(num_segments == 0) return;

	thrust::device_ptr<Segment> seg_begin = thrust::device_pointer_cast((Segment*)gctx->d_BufferInput);
    thrust::device_ptr<Segment> seg_end = thrust::device_pointer_cast((Segment *)gctx->d_BufferInput + num_segments);
    thrust::sort(thrust::device, seg_begin, seg_end, 
		[] __device__(const Segment &a, const Segment &b) {
			if(a.pair_id != b.pair_id){
				return a.pair_id < b.pair_id;
			}else if(fabs(a.start.x - b.start.x) >= 1e-9){
				return a.start.x < b.start.x;
			}else if(fabs(a.start.y - b.start.y) >= 1e-9){
				return a.start.y < b.start.y;
			}else if(fabs(a.end.x - b.end.x) >= 1e-9){
				return a.end.x < b.end.x;
			}else if(fabs(a.end.y - b.end.y) >= 1e-9){
				return a.end.y < b.end.y;
			}else{
				return a.is_source < b.is_source;
			}
		});

	// PrintBuffer((Segment *)gctx->d_BufferInput, num_segments);
    // return;

	// 生成索引 [0, 1, 2, ..., n-1]
    thrust::device_vector<int> d_indices(num_segments);
    thrust::sequence(d_indices.begin(), d_indices.end());

	// 提取pair_id
	thrust::device_vector<int> pair_ids(num_segments);
	thrust::transform(seg_begin, seg_end, pair_ids.begin(), ExtractPairId());

	// 创建布尔掩码：每个不同值的起点为 true
    thrust::device_vector<int> d_flags(num_segments);
    thrust::adjacent_difference(thrust::device, pair_ids.begin(), pair_ids.end(), d_flags.begin());

    // 把非零变成1（表示新组开始）
    thrust::transform(d_flags.begin(), d_flags.end(), d_flags.begin(),
        [] __device__(int x){ return x != 0 ? 1 : 0; });

	// 修复第一个元素（应始终为true）
    d_flags[0] = 1;	

	// 拷贝新组的索引和对应值
    int num_groups = thrust::count(d_flags.begin(), d_flags.end(), 1);

	thrust::device_vector<int> d_starts(num_groups + 1, num_segments);
	
    thrust::copy_if(thrust::device,
					d_indices.begin(), d_indices.end(),
                    d_flags.begin(), d_starts.begin(),
                    thrust::identity<int>());

	// thrust::host_vector<int> h_starts = d_starts;
	// std::cout << "\nStart positions:\n";
    // for (int i : h_starts) std::cout << i << " ";
    // std::cout << std::endl;
	
	uint8_t *pip = nullptr;
	CUDA_SAFE_CALL(cudaMalloc((void **) &pip, num_segments * sizeof(uint8_t)));
	CUDA_SAFE_CALL(cudaMemset(pip, 0, num_segments * sizeof(uint8_t)));

	grid_size = (num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;

	kernel_filter_segment_contain<<<grid_size, block_size>>>(
		(Segment *)gctx->d_BufferInput, gctx->d_candidate_pairs + gctx->index, 
		gctx->d_idealoffset, gctx->d_info, gctx->d_status, 
		gctx->d_vertices, num_segments, pip, 
		(PixMapping *)gctx->d_BufferOutput, gctx->d_bufferoutput_size, gctx->category_count);
	cudaDeviceSynchronize();
    check_execution("kernel_filter_segment_contain");
    
	CUDA_SAFE_CALL(cudaMemcpy(&h_bufferoutput_size, gctx->d_bufferoutput_size, sizeof(uint), cudaMemcpyDeviceToHost));

	grid_size = (h_bufferoutput_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	kernel_refinement_segment_contain<<<grid_size, block_size>>>(
		(PixMapping *)gctx->d_BufferOutput, (Segment *)gctx->d_BufferInput,
		gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_info, 
		gctx->d_offset, gctx->d_edge_sequences, gctx->d_vertices, 
		gctx->d_gridline_offset, gctx->d_gridline_nodes, gctx->d_bufferoutput_size, 
		pip);
	cudaDeviceSynchronize();
    check_execution("kernel_refinement_segment_contain");	

	auto segment_end = std::chrono::high_resolution_clock::now();
	auto segment_duration = std::chrono::duration_cast<std::chrono::milliseconds>(segment_end - segment_start);
	std::cout << "segment time: " << segment_duration.count() << " ms" << std::endl;

	// printf("---------------------------------------------------------------------------------------------------\n");
	
    // PrintBuffer((Segment *)gctx->d_BufferInput, num_segments);
	
	// // PrintBuffer((bool *) pip, num_segments);
	
	// uint8_t* h_Buffer = new uint8_t[num_segments];
    // CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));
	// // int _sum = 0;
    // for (int i = 0; i < num_segments; i++) {
	// 	// _sum += h_Buffer[i];
	// 	printf("%d ", h_Buffer[i]);
	// 	if(i % 10 == 9) printf("\n");
    // }

	// // printf("sum = %d\n", _sum);

    // return;

	// printf("---------------------------------------------------------------------------------------------------\n");
	// auto transfer_start = std::chrono::high_resolution_clock::now();

	// int* start_ptr = thrust::raw_pointer_cast(d_starts.data());
	// double *d_area = nullptr;
	// CUDA_SAFE_CALL(cudaMalloc((void **) &d_area, sizeof(double)));

	// grid_size = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// rebuild_polygons<<<grid_size, block_size>>>((Segment *)gctx->d_BufferInput, pip, num_groups, gctx->d_candidate_pairs + gctx->index, gctx->d_idealoffset, gctx->d_vertices, start_ptr, d_area);
	// cudaDeviceSynchronize();
    // check_execution("kernel_refinement_segment_contain");	

	// double h_area;
	// CUDA_SAFE_CALL(cudaMemcpy(&h_area, d_area, sizeof(double), cudaMemcpyDeviceToHost));
	// printf("area = %lf\n", h_area);
	// gctx->area += h_area;

    // auto transfer_end = std::chrono::high_resolution_clock::now();
	// auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(transfer_end - transfer_start);
	// std::cout << "transfer time: " << transfer_duration.count() << " ms" << std::endl;
	
	// CUDA_SAFE_CALL(cudaHostAlloc((void**)&gctx->segments, num_segments * sizeof(Segment), cudaHostAllocDefault));
	// gctx->num_segments = num_segments;
	// CUDA_SAFE_CALL(cudaMemcpy(gctx->segments, (Segment *)gctx->d_BufferInput, num_segments * sizeof(Segment), cudaMemcpyDeviceToHost));
    
	// CUDA_SAFE_CALL(cudaHostAlloc((void**)&gctx->pip, num_segments * sizeof(uint8_t), cudaHostAllocDefault));
	// CUDA_SAFE_CALL(cudaMemcpy(gctx->pip, pip, num_segments * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_inters_per_pair));
	CUDA_SAFE_CALL(cudaFree(pip));
	// CUDA_SAFE_CALL(cudaFree(d_area));
    return;
}
