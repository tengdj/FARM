#include "../include/Ideal.h"

namespace {

double fast_box_max_distance(const box &source, const box &target, bool geography)
{
	float source_x[2] = {(float)source.low[0], (float)source.high[0]};
	float source_y[2] = {(float)source.low[1], (float)source.high[1]};
	float target_x[2] = {(float)target.low[0], (float)target.high[0]};
	float target_y[2] = {(float)target.low[1], (float)target.high[1]};
	double max_dist_sq = 0.0;
	double longitude_degree = geography ? degree_per_kilometer_longitude(source.low[1]) : 1.0;

	for(int sx = 0; sx < 2; sx ++){
		for(int sy = 0; sy < 2; sy ++){
			for(int tx = 0; tx < 2; tx ++){
				for(int ty = 0; ty < 2; ty ++){
					double dx = source_x[sx] - target_x[tx];
					double dy = source_y[sy] - target_y[ty];
					if(geography){
						dy = dy / degree_per_kilometer_latitude;
						dx = dx / longitude_degree;
					}
					max_dist_sq = max(max_dist_sq, dx * dx + dy * dy);
				}
			}
		}
	}

	return sqrt(max_dist_sq);
}

box fast_pixel_box(const MyRaster &raster, int id, int dimx)
{
	int y = id / dimx;
	int x = id - y * dimx;
	double lowx = raster.mbr->low[0] + x * raster.get_step_x();
	double lowy = raster.mbr->low[1] + y * raster.get_step_y();
	return box(lowx, lowy, lowx + raster.get_step_x(), lowy + raster.get_step_y());
}

vector<pair<int, int>> build_child_ranges(const Hraster &parent, const Hraster &child, bool is_x)
{
	const int parent_dim = is_x ? parent.get_dimx() : parent.get_dimy();
	const int child_dim = is_x ? child.get_dimx() : child.get_dimy();
	const double parent_step = is_x ? parent.get_step_x() : parent.get_step_y();
	const double child_step = is_x ? child.get_step_x() : child.get_step_y();
	const double parent_low = is_x ? parent.mbr->low[0] : parent.mbr->low[1];
	const double child_low = is_x ? child.mbr->low[0] : child.mbr->low[1];

	const int child_span = max(1, static_cast<int>(lround(parent_step / child_step)));
	const int origin_offset = static_cast<int>(lround((parent_low - child_low) / child_step));

	vector<pair<int, int>> ranges(parent_dim);
	for (int i = 0; i < parent_dim; ++i)
	{
		int start = origin_offset + i * child_span;
		int end = start + child_span - 1;

		start = max(0, min(start, child_dim - 1));
		end = max(0, min(end, child_dim - 1));
		if (end < start)
		{
			end = start;
		}
		ranges[i] = {start, end};
	}
	return ranges;
}

Point interpolate_edge(Point *vertices, uint32_t edge_id, double param)
{
	Point d = vertices[edge_id + 1] - vertices[edge_id];
	return vertices[edge_id] + d * static_cast<float>(param);
}

int collinear_direction(const Point &a, const Point &b, double eps_sq)
{
	double cross = (double)a.x * b.y - (double)a.y * b.x;
	double len_a = (double)a.x * a.x + (double)a.y * a.y;
	double len_b = (double)b.x * b.x + (double)b.y * b.y;

	if (cross * cross > eps_sq * len_a * len_b)
		return 0;

	double dot = (double)a.x * b.x + (double)a.y * b.y;
	return dot > 0.0 ? 1 : -1;
}

bool same_direction(const Point &a, const Point &b)
{
	return (double)a.x * b.x + (double)a.y * b.y > 0.0;
}

void collect_intersections(Point *source_vertices, Point *target_vertices,
						   int source_start, int source_len,
						   int target_start, int target_len,
						   int source_num_vertices, int target_num_vertices,
						   vector<Intersection> &inters)
{
	const double eps_t = 1e-6;
	const double eps_sq = 1e-12;
	const int source_end = source_start + source_len;
	const int target_end = target_start + target_len;

	for (int i = source_start; i < source_end; i++)
	{
		Point source_a = source_vertices[i];
		Point source_b = source_vertices[i + 1];
		Point source_dir = source_b - source_a;
		double source_len_sq = (double)source_dir.x * source_dir.x + (double)source_dir.y * source_dir.y;
		double source_min_x = min((double)source_a.x, (double)source_b.x) - eps_t;
		double source_max_x = max((double)source_a.x, (double)source_b.x) + eps_t;
		double source_min_y = min((double)source_a.y, (double)source_b.y) - eps_t;
		double source_max_y = max((double)source_a.y, (double)source_b.y) + eps_t;

		for (int j = target_start; j < target_end; j++)
		{
			Point target_a = target_vertices[j];
			Point target_b = target_vertices[j + 1];
			if (source_max_x < min((double)target_a.x, (double)target_b.x) ||
				source_min_x > max((double)target_a.x, (double)target_b.x) ||
				source_max_y < min((double)target_a.y, (double)target_b.y) ||
				source_min_y > max((double)target_a.y, (double)target_b.y))
			{
				continue;
			}

			Point target_dir = target_b - target_a;
			double target_len_sq = (double)target_dir.x * target_dir.x + (double)target_dir.y * target_dir.y;
			double denom = (double)source_dir.x * target_dir.y - (double)source_dir.y * target_dir.x;
			if (denom * denom <= eps_sq * source_len_sq * target_len_sq)
				continue;

			Point delta = target_a - source_a;
			double t = ((double)delta.x * target_dir.y - (double)delta.y * target_dir.x) / denom;
			if (t < -eps_t || t > 1.0 + eps_t)
				continue;

			double u = ((double)delta.x * source_dir.y - (double)delta.y * source_dir.x) / denom;
			if (u < -eps_t || u > 1.0 + eps_t)
				continue;

			bool source_vertex = t <= eps_t || t >= 1.0 - eps_t;
			bool target_vertex = u <= eps_t || u >= 1.0 - eps_t;
			if (source_vertex && target_vertex)
			{
				int source_vertex_id = t < 0.5 ? i : i + 1;
				int target_vertex_id = u < 0.5 ? j : j + 1;
				Point source_in = source_vertex_id == 0 ? source_vertices[source_vertex_id] - source_vertices[source_num_vertices - 2] : source_vertices[source_vertex_id] - source_vertices[source_vertex_id - 1];
				Point source_out = source_vertex_id == source_num_vertices - 1 ? source_vertices[1] - source_vertices[source_vertex_id] : source_vertices[source_vertex_id + 1] - source_vertices[source_vertex_id];
				Point target_in = target_vertex_id == 0 ? target_vertices[target_vertex_id] - target_vertices[target_num_vertices - 2] : target_vertices[target_vertex_id] - target_vertices[target_vertex_id - 1];
				Point target_out = target_vertex_id == target_num_vertices - 1 ? target_vertices[1] - target_vertices[target_vertex_id] : target_vertices[target_vertex_id + 1] - target_vertices[target_vertex_id];

				bool same_dir_overlap = collinear_direction(source_in, target_in, eps_sq) == 1 &&
										collinear_direction(source_out, target_out, eps_sq) == 1;
				bool opp_dir_overlap = collinear_direction(source_in, target_out, eps_sq) == -1 &&
									   collinear_direction(source_out, target_in, eps_sq) == -1;
				if (same_dir_overlap || opp_dir_overlap)
					continue;
			}

			inters.push_back({0, static_cast<uint>(i), static_cast<uint>(j), t, u, OUT});
		}
	}
}

void sort_unique_source_intersections(vector<Intersection> &inters)
{
	sort(inters.begin(), inters.end(), [](const Intersection &a, const Intersection &b) {
		double a_pos = (double)a.edge_source_id + a.t;
		double b_pos = (double)b.edge_source_id + b.t;
		return a_pos < b_pos;
	});

	auto unique_end = unique(inters.begin(), inters.end(), [](const Intersection &a, const Intersection &b) {
		return fabs((double)a.edge_source_id - (double)b.edge_source_id + a.t - b.t) < eps;
	});
	inters.erase(unique_end, inters.end());

	sort(inters.begin(), inters.end(), [](const Intersection &a, const Intersection &b) {
		if (a.edge_source_id != b.edge_source_id)
			return a.edge_source_id < b.edge_source_id;
		return a.t < b.t;
	});
}

void sort_target_intersections(vector<Intersection> &inters)
{
	sort(inters.begin(), inters.end(), [](const Intersection &a, const Intersection &b) {
		if (a.edge_target_id != b.edge_target_id)
			return a.edge_target_id < b.edge_target_id;
		return a.u < b.u;
	});
}

void normalize_edge_param(int &edge_id, double &param, int num_edges, bool is_first)
{
	if (is_first && fabs(param - 1.0) < eps)
	{
		edge_id = (edge_id + 1) % num_edges;
		param = 0.0;
	}
	else if (!is_first && fabs(param) < eps)
	{
		edge_id = edge_id > 0 ? edge_id - 1 : num_edges - 1;
		param = 1.0;
	}
}

PartitionStatus classify_intersection_arc(const Intersection &inter1, const Intersection &inter2,
										  Point *primary_vertices, int primary_num_vertices,
										  Ideal *secondary, bool is_primary)
{
	uint32_t raw_edge1 = is_primary ? inter1.edge_source_id : inter1.edge_target_id;
	uint32_t raw_edge2 = is_primary ? inter2.edge_source_id : inter2.edge_target_id;
	double param1 = is_primary ? inter1.t : inter1.u;
	double param2 = is_primary ? inter2.t : inter2.u;
	Point p1 = interpolate_edge(primary_vertices, raw_edge1, param1);
	Point p2 = interpolate_edge(primary_vertices, raw_edge2, param2);

	int edge1 = raw_edge1;
	int edge2 = raw_edge2;
	int num_edges = primary_num_vertices - 1;
	normalize_edge_param(edge1, param1, num_edges, true);
	normalize_edge_param(edge2, param2, num_edges, false);

	Point sample;
	if (edge1 == edge2 && param1 < param2)
		sample = (p1 + p2) * 0.5f;
	else
		sample = (p1 + primary_vertices[edge1 + 1]) * 0.5f;

	double rem_x = remainder(sample.x - secondary->getMBB()->low[0], secondary->get_step_x());
	double rem_y = remainder(sample.y - secondary->getMBB()->low[1], secondary->get_step_y());
	int xoff = secondary->get_offset_x(sample.x);
	int yoff = secondary->get_offset_y(sample.y);
	int pix = secondary->get_id(xoff, yoff);
	PartitionStatus st = (fabs(rem_x) < 1e-9 || fabs(rem_y) < 1e-9) ? BORDER : secondary->show_status(pix);
	if (st != BORDER)
		return st;

	box bx = secondary->get_pixel_box(xoff, yoff);
	bool ret = false;
	for (uint32_t e = 0; e < secondary->get_num_sequences(pix); e++)
	{
		auto edges = secondary->get_edge_sequence(secondary->get_offset(pix) + e);
		auto pos = edges.first;
		for (int k = 0; k < edges.second; k++)
		{
			Point v1 = secondary->get_boundary()->p[pos + k];
			Point v2 = secondary->get_boundary()->p[pos + k + 1];
			if (sample == v1 || sample == v2)
				return BORDER;

			if ((v1.y >= sample.y) != (v2.y >= sample.y))
			{
				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				if (fabs(dy) > 1e-9)
				{
					const double int_x = dx * (sample.y - v1.y) / dy + v1.x;
					if (fabs(sample.x - int_x) < 1e-9)
						return BORDER;
					if (sample.x < int_x && int_x <= bx.high[0])
						ret = !ret;
				}
			}
			else if (v1.y == sample.y && v2.y == sample.y &&
					 (v1.x >= sample.x) != (v2.x >= sample.x))
			{
				return BORDER;
			}
		}
	}

	int right_line = xoff + 1;
	uint32_t i = secondary->get_vertical()->get_offset(right_line);
	uint32_t j = right_line < secondary->get_dimx()
					 ? secondary->get_vertical()->get_offset(right_line + 1)
					 : secondary->get_vertical()->get_num_crosses();
	while (i < j && secondary->get_vertical()->get_intersection_nodes(i) < sample.y)
	{
		ret = !ret;
		i++;
	}
	return ret ? IN : OUT;
}

double accumulate_area_for_order(vector<Intersection> &inters,
								 Point *primary_vertices, int primary_num_vertices,
								 Point *source_vertices, int source_num_vertices,
								 Point *target_vertices, int target_num_vertices,
								 bool is_primary)
{
	double area = 0.0;
	const int num_intersections = inters.size();
	const int num_edges = primary_num_vertices - 1;

	for (int i = 0; i < num_intersections; i++)
	{
		Intersection inter1 = inters[i];
		Intersection inter2 = (i + 1 >= num_intersections) ? inters[0] : inters[i + 1];
		bool wrapped = i + 1 >= num_intersections;

		int edge1 = is_primary ? inter1.edge_source_id : inter1.edge_target_id;
		int edge2 = is_primary ? inter2.edge_source_id : inter2.edge_target_id;
		double param1 = is_primary ? inter1.t : inter1.u;
		double param2 = is_primary ? inter2.t : inter2.u;
		Point p1 = interpolate_edge(primary_vertices, edge1, param1);
		Point p2 = interpolate_edge(primary_vertices, edge2, param2);

		normalize_edge_param(edge1, param1, num_edges, true);
		normalize_edge_param(edge2, param2, num_edges, false);

		if (inters[i].status == IN)
		{
			double a = 0.0;
			double b = 0.0;
			double last_x = p1.x;
			double last_y = p1.y;

			if (!wrapped || edge1 < edge2)
			{
				for (int ver_id = edge1 + 1; ver_id <= edge2; ver_id++)
				{
					a += last_x * primary_vertices[ver_id].y;
					b += last_y * primary_vertices[ver_id].x;
					last_x = primary_vertices[ver_id].x;
					last_y = primary_vertices[ver_id].y;
				}
			}
			else
			{
				for (int ver_id = edge1 + 1; ver_id < primary_num_vertices - 1; ver_id++)
				{
					a += last_x * primary_vertices[ver_id].y;
					b += last_y * primary_vertices[ver_id].x;
					last_x = primary_vertices[ver_id].x;
					last_y = primary_vertices[ver_id].y;
				}
				for (int ver_id = 0; ver_id <= edge2; ver_id++)
				{
					a += last_x * primary_vertices[ver_id].y;
					b += last_y * primary_vertices[ver_id].x;
					last_x = primary_vertices[ver_id].x;
					last_y = primary_vertices[ver_id].y;
				}
			}

			a += last_x * p2.y;
			b += last_y * p2.x;
			area += a - b;
		}
		else if (inters[i].status == BORDER && is_primary)
		{
			uint32_t source_vertex_id = inter1.t < 0.5 ? inter1.edge_source_id : inter1.edge_source_id + 1;
			uint32_t target_vertex_id = inter1.u < 0.5 ? inter1.edge_target_id : inter1.edge_target_id + 1;
			Point source_out = source_vertex_id == source_num_vertices - 1 ? source_vertices[1] - source_vertices[source_vertex_id] : source_vertices[source_vertex_id + 1] - source_vertices[source_vertex_id];
			Point target_out = target_vertex_id == target_num_vertices - 1 ? target_vertices[1] - target_vertices[target_vertex_id] : target_vertices[target_vertex_id + 1] - target_vertices[target_vertex_id];

			if (same_direction(source_out, target_out))
			{
				double a = 0.0;
				double b = 0.0;
				double last_x = p1.x;
				double last_y = p1.y;

				if (!wrapped || edge1 < edge2)
				{
					for (int ver_id = edge1 + 1; ver_id <= edge2; ver_id++)
					{
						a += last_x * primary_vertices[ver_id].y;
						b += last_y * primary_vertices[ver_id].x;
						last_x = primary_vertices[ver_id].x;
						last_y = primary_vertices[ver_id].y;
					}
				}
				else
				{
					for (int ver_id = edge1 + 1; ver_id < primary_num_vertices - 1; ver_id++)
					{
						a += last_x * primary_vertices[ver_id].y;
						b += last_y * primary_vertices[ver_id].x;
						last_x = primary_vertices[ver_id].x;
						last_y = primary_vertices[ver_id].y;
					}
					for (int ver_id = 0; ver_id <= edge2; ver_id++)
					{
						a += last_x * primary_vertices[ver_id].y;
						b += last_y * primary_vertices[ver_id].x;
						last_x = primary_vertices[ver_id].x;
						last_y = primary_vertices[ver_id].y;
					}
				}

				a += last_x * p2.y;
				b += last_y * p2.x;
				area += a - b;
			}
		}
	}

	return area;
}

} // namespace

Ideal::~Ideal()
{
	if (offset)
		delete[] offset;
	if (edge_sequences)
		delete[] edge_sequences;
	if (vertical)
		delete vertical;
	if (horizontal)
		delete horizontal;
	if (areas)
		delete[] areas;
	if (layer_offset)
		delete[] layer_offset;
	if (layer_info)
		delete[] layer_info;
	if (layers)
		delete[] layers;
}

void Ideal::add_edge(int idx, int start, int end)
{
	assert(end - start + 1 > 0);
	edge_sequences[idx] = make_pair(start, end - start + 1);
}

uint32_t Ideal::get_num_sequences(int id)
{
	if (show_status(id) != BORDER)
		return 0;
	return offset[id + 1] - offset[id];
}

void Ideal::init_edge_sequences(int num_edge_seqs)
{
	assert(num_edge_seqs >= 0);
	len_edge_sequences = num_edge_seqs;
	edge_sequences = new pair<uint32_t, uint32_t>[num_edge_seqs];
	assert(len_edge_sequences < 65536); // 2^16, to fit in uint16_t for edge id and count
}

void Ideal::process_pixels_null(int x, int y)
{
	offset[x * y] = len_edge_sequences;
	for (int i = x * y - 1; i >= 0; i--)
	{
		if (show_status(i) != BORDER)
		{
			offset[i] = offset[i + 1];
		}
	}
}

double Ideal::decodePixelArea(int id, bool isLow){
    uint8_t fullness = status[id];
	double pixelArea = get_pixel_area();
	if (fullness == 0)
    {
        return 0.0f;
    }
    else if (fullness == category_count - 1)
    {
        return pixelArea;
    }
    else
    {
        return (1.0 * fullness - isLow) / (category_count - 2) * pixelArea;
    }
}

uint8_t Ideal::encodePixelArea(double area){
	double pixelArea = get_pixel_area();
	double ratio = area / pixelArea;

	if (fabs(ratio - 1.0) < 1e-9)
	{
		// full
		return category_count - 1;
	}

	if (fabs(ratio) < 1e-9)
	{
		// empty
		return 0;
	}

	// int idx = static_cast<int>((ratio * (category_count - 2)) + 1);
	int idx = static_cast<int>(ceil(ratio * (category_count - 2)));

	if (idx >= category_count)
		idx = category_count - 1; // 防止越界

	assert(idx < 256);
	return idx;
}

// void Ideal::process_crosses(map<int, vector<cross_info>> edges_info)
// {
// 	int num_edge_seqs = 0;

// 	for (auto info : edges_info)
// 	{
// 		auto pix = info.first;
// 		auto crosses = info.second;

// 		if (crosses.size() == 0)
// 			return;

// 		if (crosses.size() % 2 == 1)
// 		{
// 			crosses.push_back(cross_info((cross_type)!crosses[crosses.size() - 1].type, crosses[crosses.size() - 1].edge_id));
// 		}

// 		int start = 0;
// 		int end = crosses.size() - 1;
// 		if (crosses[0].type == LEAVE)
// 		{
// 			assert(crosses[end].type == ENTER);
// 			num_edge_seqs += 2;
// 			start++;
// 			end--;
// 		}

// 		for (int i = start; i <= end; i++)
// 		{
// 			assert(crosses[i].type == ENTER);
// 			// special case, an ENTER has no pair LEAVE,
// 			// happens when one edge crosses the pair
// 			if (i == end || crosses[i + 1].type == ENTER)
// 			{
// 				num_edge_seqs++;
// 			}
// 			else
// 			{
// 				num_edge_seqs++;
// 				i++;
// 			}
// 		}
// 	}

// 	init_edge_sequences(num_edge_seqs);

// 	int idx = 0;
// 	int edge_count = 0;
// 	for (auto info : edges_info)
// 	{
// 		auto pix = info.first;
// 		auto crosses = info.second;

// 		if (crosses.size() == 0)
// 			return;

// 		if (crosses.size() % 2 == 1)
// 		{
// 			crosses.push_back(cross_info((cross_type)!crosses[crosses.size() - 1].type, crosses[crosses.size() - 1].edge_id));
// 		}

// 		assert(crosses.size() % 2 == 0);

// 		// Initialize based on crosses.size().
// 		int start = 0;
// 		int end = crosses.size() - 1;
// 		set_offset(pix, idx);

// 		if (crosses[0].type == LEAVE)
// 		{
// 			assert(crosses[end].type == ENTER);
// 			add_edge(idx++, 0, crosses[0].edge_id);
// 			add_edge(idx++, crosses[end].edge_id, boundary->num_vertices - 2);
// 			start++;
// 			end--;
// 		}

// 		for (int i = start; i <= end; i++)
// 		{
// 			assert(crosses[i].type == ENTER);
// 			// special case, an ENTER has no pair LEAVE,
// 			// happens when one edge crosses the pair
// 			if (i == end || crosses[i + 1].type == ENTER)
// 			{
// 				add_edge(idx++, crosses[i].edge_id, crosses[i].edge_id);
// 			}
// 			else
// 			{
// 				add_edge(idx++, crosses[i].edge_id, crosses[i + 1].edge_id);
// 				i++;
// 			}
// 		}
// 	}
// }

// 假设 edges_info 已经优化为 vector<vector<cross_info>>
// 如果上一步没改，这里参数类型依然是 map<int, vector<cross_info>>& (注意加引用!)
void Ideal::process_crosses(const vector<vector<cross_info>>& edges_info)
{
    // 预估总边数，避免频繁 realloc，这里只是一个启发式估算
    int estimated_seqs = edges_info.size() * 2; 
    // 注意：原代码的 init_edge_sequences 似乎是分配全局数组？
    // 如果是，我们需要先计算总数。但为了性能，建议改造 init_edge_sequences 支持动态添加，
    // 或者我们必须保留两次遍历，但要去掉昂贵的 map 拷贝。
    
    // 【方案一：如果底层存储支持动态添加 (推荐)】
    // init_edge_sequences(0); // 清空
    
    // 【方案二：必须预先分配 (保留原逻辑但优化性能)】
    // 既然原代码逻辑强依赖先分配，我们优化计算过程。
    
    int total_seqs = 0;
    
    // Pass 1: 快速计算需要的总空间
    for (int pix_id = 0; pix_id < edges_info.size(); ++pix_id)
    {
        const auto& crosses = edges_info[pix_id];
        if (crosses.empty()) continue;

        size_t size = crosses.size();
		assert(size % 2 == 0);
        // 奇数个事件补一个，变成偶数
        if (size % 2 != 0) size++;

        // 逻辑简化：
        // 正常情况下是成对的 ENTER -> LEAVE。
        // 原代码逻辑：
        // 1. 如果第一个是 LEAVE，说明跨越了首尾，这会产生 2 个序列片段。
        // 2. 剩下的中间部分，每对 ENTER...LEAVE 算 1 个，单独 ENTER 算 1 个。
        
        // 我们可以通过简单的数学推导简化这个循环：
        // 基本上每两个事件构成一个 edge_seq，除非断开。
        
        // 但为了保证逻辑完全一致，保留原有的判断结构，只做语法优化。
        
        bool first_is_leave = (crosses[0].type == LEAVE);
        if (first_is_leave) {
            total_seqs += 2; // 首尾各一段
        }
        
        int start = first_is_leave ? 1 : 0;
        int end = (first_is_leave) ? size - 2 : size - 1; // size 已经是偶数了
        
        // 剩下的事件数
        int remain = end - start + 1;
        // 原代码逻辑：如果成对 (ENTER, LEAVE) 则消耗2个产生1个seq；如果单独 (ENTER, ENTER) 则消耗1个产生1个seq。
        // 这实际上意味着：每个 seq 消耗 1 或 2 个事件。
        // 让我们仔细看原代码：
        // if (i == end || crosses[i+1].type == ENTER) -> 意味着当前是单独的 ENTER -> count++
        // else -> 当前是 ENTER, 下一个是 LEAVE -> count++, i++ (消耗两个)
        
        // 我们可以遍历来计算，但不需要拷贝 vector
        for (int i = start; i <= end; i++) {
            // 注意：这里需要小心访问越界，因为我们要模拟补齐后的效果
            bool is_last = (i == size - 1); 
            // 模拟补齐后的类型：如果是补齐的那个点，类型取反
            cross_type current_type = crosses[i].type; // 假设 i < crosses.size()
            
            // 下一个元素的类型
            cross_type next_type;
            if (i + 1 < crosses.size()) {
                next_type = crosses[i +  1].type;
            } else {
                // 这是补齐的那个虚拟元素
                next_type = (cross_type)!crosses.back().type;
            }

            if (is_last || next_type == ENTER) {
                total_seqs++;
            } else {
                total_seqs++;
                i++; // Skip the LEAVE pair
            }
        }
    }

    init_edge_sequences(total_seqs);

    // Pass 2: 实际填充
    int current_global_idx = 0;

    for (int pix_id = 0; pix_id < edges_info.size(); ++pix_id)
    {
        // 必须拷贝一份 crosses，因为可能要 push_back 修改它
        // 为了优化，只有在需要 push_back 时才拷贝，或者直接处理逻辑
        // 这里为了代码清晰，我们只引用，并在逻辑上处理“虚拟尾部”
        const auto& const_crosses = edges_info[pix_id];
        if (const_crosses.empty()) continue;

        // 设置当前像素的起始偏移量
        set_offset(pix_id, current_global_idx);

        bool needs_padding = (const_crosses.size() % 2 != 0);
        int effective_size = const_crosses.size() + (needs_padding ? 1 : 0);

        // 获取第 k 个元素的辅助 lambda，自动处理补齐逻辑
        auto get_cross = [&](int k) -> cross_info {
            if (k < const_crosses.size()) return const_crosses[k];
            // 补齐的元素：类型取反，ID复用最后一个
            return cross_info((cross_type)!const_crosses.back().type, const_crosses.back().edge_id);
        };

        int start = 0;
        int end = effective_size - 1;

        // 处理首尾跨越的情况 (First is LEAVE)
        if (get_cross(0).type == LEAVE)
        {
            // 首部：从 0 到 第一个 LEAVE 的边
            add_edge(current_global_idx++, 0, get_cross(0).edge_id);
            // 尾部：从 最后一个 ENTER 到 最后一个顶点
            add_edge(current_global_idx++, get_cross(end).edge_id, boundary->num_vertices - 2); // 注意这里原代码是 -2，需确认业务逻辑
            
            start++;
            end--;
        }

        // 处理中间部分
        for (int i = start; i <= end; i++)
        {
            cross_info curr = get_cross(i);
            assert(curr.type == ENTER); // 原代码有这个断言

            bool is_single = false;
            if (i == end) {
                is_single = true;
            } else {
                if (get_cross(i+1).type == ENTER) {
                    is_single = true;
                }
            }

            if (is_single)
            {
                // 单个 ENTER 事件，通常表示顶点就在像素内
                add_edge(current_global_idx++, curr.edge_id, curr.edge_id);
            }
            else
            {
                // ENTER -> LEAVE 对
                add_edge(current_global_idx++, curr.edge_id, get_cross(i+1).edge_id);
                i++; // 跳过下一个
            }
        }
	}
}

void Ideal::process_crosses_sparse(const vector<int> &pixel_ids, const vector<vector<cross_info>> &edges_info)
{
	vector<int> order;
	order.reserve(pixel_ids.size());
	for (int i = 0; i < pixel_ids.size(); ++i)
	{
		order.push_back(i);
	}
	sort(order.begin(), order.end(), [&](int lhs, int rhs) {
		return pixel_ids[lhs] < pixel_ids[rhs];
	});

	auto get_cross = [](const vector<cross_info> &crosses, int idx) -> cross_info {
		if (idx < crosses.size())
		{
			return crosses[idx];
		}
		return cross_info((cross_type)!crosses.back().type, crosses.back().edge_id);
	};

	auto emit_sequences = [&](const vector<cross_info> &crosses, auto emit) {
		if (crosses.empty())
		{
			return 0;
		}

		int emitted = 0;
		const int effective_size = crosses.size() + (crosses.size() % 2);
		int start = 0;
		int end = effective_size - 1;

		if (get_cross(crosses, 0).type == LEAVE)
		{
			emit(0, get_cross(crosses, 0).edge_id);
			emit(get_cross(crosses, end).edge_id, boundary->num_vertices - 2);
			emitted += 2;
			start++;
			end--;
		}

		for (int i = start; i <= end; ++i)
		{
			cross_info curr = get_cross(crosses, i);
			if (curr.type != ENTER)
			{
				continue;
			}

			int seq_end = curr.edge_id;
			if (i < end && get_cross(crosses, i + 1).type != ENTER)
			{
				seq_end = get_cross(crosses, i + 1).edge_id;
				i++;
			}
			emit(curr.edge_id, seq_end);
			emitted++;
		}

		return emitted;
	};

	int total_seqs = 0;
	for (int bucket : order)
	{
		total_seqs += emit_sequences(edges_info[bucket], [](int, int) {});
	}

	init_edge_sequences(total_seqs);

	int current_global_idx = 0;
	for (int bucket : order)
	{
		const auto &crosses = edges_info[bucket];
		if (crosses.empty())
			continue;

		set_offset(pixel_ids[bucket], current_global_idx);
		emit_sequences(crosses, [&](int start_edge, int end_edge) {
			add_edge(current_global_idx++, start_edge, end_edge);
		});
	}

	assert(current_global_idx == total_seqs);
}

// void Ideal::process_intersection(map<int, vector<double>> intersection_info, Direction direction)
// {
// 	int num_nodes = 0;
// 	for (auto i : intersection_info)
// 	{
// 		num_nodes += i.second.size();
// 	}
// 	if (direction == HORIZONTAL)
// 	{
// 		horizontal->init_intersection_node(num_nodes);
// 		horizontal->set_num_crosses(num_nodes);
// 		int idx = 0;
// 		for (auto info : intersection_info)
// 		{
// 			auto h = info.first;
// 			auto nodes = info.second;

// 			sort(nodes.begin(), nodes.end());

// 			horizontal->set_offset(h, idx);

// 			for (auto node : nodes)
// 			{
// 				horizontal->add_node(idx, node);
// 				idx++;
// 			}
// 		}
// 		horizontal->set_offset(dimy, idx);
// 	}
// 	else
// 	{
// 		vertical->init_intersection_node(num_nodes);
// 		vertical->set_num_crosses(num_nodes);
// 		vertical->set_offset(dimx + 1, num_nodes);

// 		int idx = 0;
// 		for (auto info : intersection_info)
// 		{
// 			auto h = info.first;
// 			auto nodes = info.second;

// 			sort(nodes.begin(), nodes.end());

// 			vertical->set_offset(h, idx);

// 			for (auto node : nodes)
// 			{
// 				vertical->add_node(idx, node);
// 				idx++;
// 			}
// 		}
// 		vertical->set_offset(dimx, idx);
// 	}
// }

// 假设 intersection_info 已经根据之前的建议优化为 vector<vector<double>>
// 并且为了支持原地排序，这里不加 const
void Ideal::process_intersection(vector<vector<double>>& intersection_info, Direction direction)
{
    // 1. [代码去重] 根据方向获取目标对象指针，避免写两遍完全一样的逻辑
    // 假设 horizontal 和 vertical 是相同类型的指针（例如 IntersectionData*）
    auto* target_data = (direction == HORIZONTAL) ? horizontal : vertical;
    // 确定遍历的维度上限 (水平对应 dimy，垂直对应 dimx)
    int max_dim = (direction == HORIZONTAL) ? dimy : dimx;

    // 2. [预计算] 统计总节点数和实际相交的网格线数量
    int num_nodes = 0;
    size_t num_intersected_lines = 0;
	for (const auto& line : intersection_info) {
        num_nodes += line.size();
        if (!line.empty())
        {
            num_intersected_lines++;
        }
    }

    target_data->init_intersection_node(num_nodes);
    target_data->set_num_crosses(num_nodes);
    target_data->set_num_intersected_lines(num_intersected_lines);

    // 保留原代码中对 VERTICAL 的特殊处理 (可能是为了设置末尾哨兵的边界)
    // if (direction == VERTICAL) {
    //	target_data->set_offset(max_dim + 1, num_nodes);
    // }

    int idx = 0;

    // 3. [遍历优化] 使用索引遍历，确保每一行(即使是空的)都会被处理
    // 原 Map 遍历会跳过空行，可能导致 offset 数组中间有未初始化的空洞
    int size = intersection_info.size();
    for (int i = 0; i < size; ++i)
    {
        // 4. [引用访问] 使用引用 &，避免 vector 拷贝
        auto& nodes = intersection_info[i];

        // 设置当前行的起始 offset
        target_data->set_offset(i, idx);

        if (nodes.empty()) continue;

        // 5. [原地排序] 直接在原数据上排序，零拷贝，速度最快
        // 扫描线算法通常要求交点必须从左到右(或从下到上)有序
        std::sort(nodes.begin(), nodes.end());

        for (double node : nodes)
        {
            target_data->add_node(idx, node);
            idx++;
        }
    }
}

void Ideal::init_pixels()
{
	assert(mbr);

	if (status)
	{
		delete[] status;
		status = nullptr;
	}
	if (areas)
	{
		delete[] areas;
		areas = nullptr;
	}
	if (offset)
	{
		delete[] offset;
		offset = nullptr;
	}
	if (edge_sequences)
	{
		delete[] edge_sequences;
		edge_sequences = nullptr;
		len_edge_sequences = 0;
	}
	if (horizontal)
	{
		delete horizontal;
		horizontal = nullptr;
	}
	if (vertical)
	{
		delete vertical;
		vertical = nullptr;
	}

	status = new uint8_t[status_size];
	areas = new double[dimx * dimy]();
	memset(status, 0, status_size * sizeof(uint8_t));
	offset = new uint32_t[dimx * dimy + 1]; // +1 here is to ensure that pointer[num_pixels] equals len_edge_sequences, so we don't need to make a special case for the last pointer.
	horizontal = new Grid_line(dimy);
	vertical = new Grid_line(dimx);
}

// void Ideal::evaluate_edges()
// {
// 	map<int, vector<double>> horizontal_intersect_info;
// 	map<int, vector<double>> vertical_intersect_info;
// 	map<int, vector<cross_info>> edges_info;

// 	// normalize
// 	assert(mbr);
// 	const double start_x = mbr->low[0];
// 	const double start_y = mbr->low[1];

// 	for (int i = 0; i < boundary->num_vertices - 1; i++)
// 	{
// 		double x1 = boundary->p[i].x;
// 		double y1 = boundary->p[i].y;
// 		double x2 = boundary->p[i + 1].x;
// 		double y2 = boundary->p[i + 1].y;

// 		int cur_startx = double_to_int((x1 - start_x) / step_x);
// 		int cur_endx = double_to_int((x2 - start_x) / step_x);
// 		int cur_starty = double_to_int((y1 - start_y) / step_y);
// 		int cur_endy = double_to_int((y2 - start_y) / step_y);

// 		if (cur_startx == dimx)
// 		{
// 			cur_startx--;
// 		}
// 		if (cur_endx == dimx)
// 		{
// 			cur_endx--;
// 		}

// 		int minx = min(cur_startx, cur_endx);
// 		int maxx = max(cur_startx, cur_endx);

// 		if (cur_starty == dimy)
// 		{
// 			cur_starty--;
// 		}
// 		if (cur_endy == dimy)
// 		{
// 			cur_endy--;
// 		}
// 		// todo should not happen for normal cases
// 		if (cur_startx >= dimx || cur_endx >= dimx || cur_starty >= dimy || cur_endy >= dimy)
// 		{
// 			cout << "xrange\t" << cur_startx << " " << cur_endx << endl;
// 			cout << "yrange\t" << cur_starty << " " << cur_endy << endl;
// 			printf("xrange_val\t%f %f\n", (x1 - start_x) / step_x, (x2 - start_x) / step_x);
// 			printf("yrange_val\t%f %f\n", (y1 - start_y) / step_y, (y2 - start_y) / step_y);
// 			assert(false);
// 		}
// 		assert(cur_startx < dimx);
// 		assert(cur_endx < dimx);
// 		assert(cur_starty < dimy);
// 		assert(cur_endy < dimy);

// 		set_status(get_id(cur_startx, cur_starty), BORDER);
// 		set_status(get_id(cur_endx, cur_endy), BORDER);

// 		// in the same pixel
// 		if (cur_startx == cur_endx && cur_starty == cur_endy)
// 		{
// 			continue;
// 		}

// 		if (y1 == y2)
// 		{
// 			// left to right
// 			if (cur_startx < cur_endx)
// 			{
// 				for (int x = cur_startx; x < cur_endx; x++)
// 				{
// 					vertical_intersect_info[x + 1].push_back(y1);
// 					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
// 					edges_info[get_id(x + 1, cur_starty)].push_back(cross_info(ENTER, i));
// 					set_status(get_id(x, cur_starty), BORDER);
// 					set_status(get_id(x + 1, cur_starty), BORDER);
// 				}
// 			}
// 			else
// 			{ // right to left
// 				for (int x = cur_startx; x > cur_endx; x--)
// 				{
// 					vertical_intersect_info[x].push_back(y1);
// 					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
// 					edges_info[get_id(x - 1, cur_starty)].push_back(cross_info(ENTER, i));
// 					set_status(get_id(x, cur_starty), BORDER);
// 					set_status(get_id(x - 1, cur_starty), BORDER);
// 				}
// 			}
// 		}
// 		else if (x1 == x2)
// 		{
// 			// bottom up
// 			if (cur_starty < cur_endy)
// 			{
// 				for (int y = cur_starty; y < cur_endy; y++)
// 				{
// 					horizontal_intersect_info[y + 1].push_back(x1);
// 					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
// 					edges_info[get_id(cur_startx, y + 1)].push_back(cross_info(ENTER, i));
// 					set_status(get_id(cur_startx, y), BORDER);
// 					set_status(get_id(cur_startx, y + 1), BORDER);
// 				}
// 			}
// 			else
// 			{ // border[bottom] down
// 				for (int y = cur_starty; y > cur_endy; y--)
// 				{
// 					horizontal_intersect_info[y].push_back(x1);
// 					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
// 					edges_info[get_id(cur_startx, y - 1)].push_back(cross_info(ENTER, i));
// 					set_status(get_id(cur_startx, y), BORDER);
// 					set_status(get_id(cur_startx, y - 1), BORDER);
// 				}
// 			}
// 		}
// 		else
// 		{
// 			// solve the line function
// 			double a = (y1 - y2) / (x1 - x2);
// 			double b = (x1 * y2 - x2 * y1) / (x1 - x2);

// 			int x = cur_startx;
// 			int y = cur_starty;
// 			while (x != cur_endx || y != cur_endy)
// 			{
// 				bool passed = false;
// 				double yval = 0;
// 				double xval = 0;
// 				int cur_x = 0;
// 				int cur_y = 0;
// 				// check horizontally
// 				if (x != cur_endx)
// 				{
// 					if (cur_startx < cur_endx)
// 					{
// 						xval = ((double)x + 1) * step_x + start_x;
// 					}
// 					else
// 					{
// 						xval = (double)x * step_x + start_x;
// 					}
// 					yval = xval * a + b;
// 					if(abs((yval - start_y) / step_y - round((yval - start_y) / step_y)) < 1e-9){
// 						if (cur_startx < cur_endx && cur_starty < cur_endy){
// 							vertical_intersect_info[x + 1].push_back(yval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x ++, y ++)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 						else if(cur_startx < cur_endx && cur_starty > cur_endy){
// 							vertical_intersect_info[x].push_back(yval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x ++, y --)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}else if(cur_startx > cur_endx && cur_starty < cur_endy){
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x --, y ++)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}else if(cur_startx > cur_endx && cur_starty > cur_endy){
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x --, y --)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 						continue;
// 					}
// 					cur_y = (yval - start_y) / step_y;
				
// 					// printf("y %f %d\n",(yval-start_y)/step_y,cur_y);
// 					if (cur_y > max(cur_endy, cur_starty))
// 					{
// 						cur_y = max(cur_endy, cur_starty);
// 					}
// 					if (cur_y < min(cur_endy, cur_starty))
// 					{
// 						cur_y = min(cur_endy, cur_starty);
// 					}
// 					if (cur_y == y)
// 					{
// 						passed = true;
// 						// left to right
// 						if (cur_startx < cur_endx)
// 						{
// 							vertical_intersect_info[x + 1].push_back(yval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x++, y)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 						else
// 						{ // right to left
// 							vertical_intersect_info[x].push_back(yval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x--, y)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 					}
// 				}
// 				// check vertically
// 				if (y != cur_endy)
// 				{
// 					if (cur_starty < cur_endy)
// 					{
// 						yval = (y + 1) * step_y + start_y;
// 					}
// 					else
// 					{
// 						yval = y * step_y + start_y;
// 					}
// 					xval = (yval - b) / a;
// 					cur_x = (xval - start_x) / step_x;
// 					// printf("x %f %d\n",(xval-start_x)/step_x,cur_x);
// 					if (cur_x > max(cur_endx, cur_startx))
// 					{
// 						cur_x = max(cur_endx, cur_startx);
// 					}
// 					if (cur_x < min(cur_endx, cur_startx))
// 					{
// 						cur_x = min(cur_endx, cur_startx);
// 					}
// 					if (cur_x == x)
// 					{
// 						passed = true;
// 						if (cur_starty < cur_endy)
// 						{ // bottom up
// 							horizontal_intersect_info[y + 1].push_back(xval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x, y++)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 						else
// 						{ // top down
// 							horizontal_intersect_info[y].push_back(xval);
// 							set_status(get_id(x, y), BORDER);
// 							edges_info[get_id(x, y--)].push_back(cross_info(LEAVE, i));
// 							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
// 							set_status(get_id(x, y), BORDER);
// 						}
// 					}
// 				}
// 				// for debugging, should never happen
// 				if (!passed)
// 				{
// 					boundary->print();
// 					cout << "dim\t" << dimx << " " << dimy << endl;
// 					cout << "step\t" << step_x << " " << step_y << endl;
// 					mbr->print();
// 					printf("POINT (%lf %lf)\n", x1, y1);
// 					printf("POINT (%lf %lf)\n", x2, y2);
// 					printf("check %.12lf %.12lf %.12lf\n", yval-start_y, step_y, fmod(yval - start_y + step_y, step_y));
// 					printf("val\t%.12f %.12f\n", (xval - start_x) / step_x, (yval - start_y) / step_y);
// 					cout << "curxy\t" << x << " " << y << endl;
// 					cout << "calxy\t" << cur_x << " " << cur_y << endl;
// 					cout << "xrange\t" << cur_startx << " " << cur_endx << endl;
// 					cout << "yrange\t" << cur_starty << " " << cur_endy << endl;
// 					printf("xrange_val\t%f %f\n", (x1 - start_x) / step_x, (x2 - start_x) / step_x);
// 					printf("yrange_val\t%f %f\n", (y1 - start_y) / step_y, (y2 - start_y) / step_y);
// 				}
// 				assert(passed);
// 			}
// 		}
// 	}

// 	// special case
// 	if (edges_info.size() == 0 && boundary->num_vertices > 0)
// 	{
// 		init_edge_sequences(1);
// 		set_offset(0, 0);
// 		add_edge(0, 0, boundary->num_vertices - 1);
// 	}
// 	else
// 	{
// 		process_crosses(edges_info);
// 	}

// 	process_intersection(horizontal_intersect_info, HORIZONTAL);
// 	process_intersection(vertical_intersect_info, VERTICAL);
// 	process_pixels_null(dimx, dimy);
// }

void Ideal::evaluate_edges()
{
	vector<vector<double>> horizontal_intersect_info(dimy + 1);
	vector<vector<double>> vertical_intersect_info(dimx + 1);
	vector<int> edge_bucket(dimx * dimy, -1);
	vector<int> edge_pixels;
	vector<vector<cross_info>> edges_info;

	assert(mbr);
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	auto add_cross = [&](int pixel_id, cross_type type, int edge_id) {
		assert(pixel_id >= 0 && pixel_id < dimx * dimy);
		set_status(pixel_id, BORDER);

		int bucket = edge_bucket[pixel_id];
		if (bucket < 0)
		{
			bucket = edges_info.size();
			edge_bucket[pixel_id] = bucket;
			edge_pixels.push_back(pixel_id);
			edges_info.emplace_back();
		}
		edges_info[bucket].emplace_back(type, edge_id);
	};

	for (int i = 0; i < boundary->num_vertices - 1; i++)
	{
		double x1 = boundary->p[i].x;
		double y1 = boundary->p[i].y;
		double x2 = boundary->p[i + 1].x;
		double y2 = boundary->p[i + 1].y;

		int start_idx_x = min(dimx - 1, double_to_int((x1 - start_x) / step_x));
		int end_idx_x = min(dimx - 1, double_to_int((x2 - start_x) / step_x));
		int start_idx_y = min(dimy - 1, double_to_int((y1 - start_y) / step_y));
		int end_idx_y = min(dimy - 1, double_to_int((y2 - start_y) / step_y));

		int cur_x = start_idx_x;
		int cur_y = start_idx_y;

		if (start_idx_x == end_idx_x && start_idx_y == end_idx_y)
		{
			continue;
		}

		int step_x_dir = (x2 > x1) ? 1 : ((x2 < x1) ? -1 : 0);
		int step_y_dir = (y2 > y1) ? 1 : ((y2 < y1) ? -1 : 0);

		if (step_y_dir == 0)
		{
			while (cur_x != end_idx_x)
			{
				int next_x = cur_x + step_x_dir;
				int intersect_idx = (step_x_dir > 0) ? cur_x + 1 : cur_x;
				vertical_intersect_info[intersect_idx].push_back(y1);

				int from_id = get_id(cur_x, cur_y);
				int to_id = get_id(next_x, cur_y);
				add_cross(from_id, LEAVE, i);
				add_cross(to_id, ENTER, i);

				cur_x = next_x;
			}
		}
		else if (step_x_dir == 0)
		{
			while (cur_y != end_idx_y)
			{
				int next_y = cur_y + step_y_dir;
				int intersect_idx = (step_y_dir > 0) ? cur_y + 1 : cur_y;
				horizontal_intersect_info[intersect_idx].push_back(x1);

				int from_id = get_id(cur_x, cur_y);
				int to_id = get_id(cur_x, next_y);
				add_cross(from_id, LEAVE, i);
				add_cross(to_id, ENTER, i);

				cur_y = next_y;
			}
		}
		else
		{
			double a = (y2 - y1) / (x2 - x1);
			double b = y1 - a * x1;

			while (cur_x != end_idx_x || cur_y != end_idx_y)
			{
				double next_x_boundary = (step_x_dir > 0) ? (cur_x + 1) * step_x + start_x : cur_x * step_x + start_x;
				double y_at_next_x = a * next_x_boundary + b;
				double next_y_boundary = (step_y_dir > 0) ? (cur_y + 1) * step_y + start_y : cur_y * step_y + start_y;
				double x_at_next_y = (next_y_boundary - b) / a;
				double cur_y_floor = cur_y * step_y + start_y;
				double cur_y_ceil = (cur_y + 1) * step_y + start_y;

				bool hit_corner = abs(y_at_next_x - (step_y_dir > 0 ? cur_y_ceil : cur_y_floor)) < 1e-9;
				bool move_x = hit_corner || (y_at_next_x >= cur_y_floor - 1e-9 && y_at_next_x <= cur_y_ceil + 1e-9);
				bool move_y = hit_corner || !move_x;

				if (move_x && cur_x == end_idx_x)
					move_x = false;
				if (move_y && cur_y == end_idx_y)
					move_y = false;
				if (!move_x && !move_y)
					break;

				int next_x = cur_x + (move_x ? step_x_dir : 0);
				int next_y = cur_y + (move_y ? step_y_dir : 0);

				if (move_x)
				{
					int edge_idx = (step_x_dir > 0) ? cur_x + 1 : cur_x;
					vertical_intersect_info[edge_idx].push_back(y_at_next_x);
				}
				if (move_y)
				{
					int edge_idx = (step_y_dir > 0) ? cur_y + 1 : cur_y;
					horizontal_intersect_info[edge_idx].push_back(x_at_next_y);
				}

				int from_id = get_id(cur_x, cur_y);
				int to_id = get_id(next_x, next_y);
				add_cross(from_id, LEAVE, i);
				add_cross(to_id, ENTER, i);

				cur_x = next_x;
				cur_y = next_y;
			}
		}
	}

	if (edges_info.empty())
	{
		if (boundary->num_vertices > 1)
		{
			int pixel_id = get_id(get_offset_x(boundary->p[0].x), get_offset_y(boundary->p[0].y));
			set_status(pixel_id, BORDER);
			init_edge_sequences(1);
			set_offset(pixel_id, 0);
			add_edge(0, 0, boundary->num_vertices - 2);
		}
		else
		{
			init_edge_sequences(0);
		}
	}
	else
	{
		process_crosses_sparse(edge_pixels, edges_info);
	}

	process_intersection(horizontal_intersect_info, HORIZONTAL);
	process_intersection(vertical_intersect_info, VERTICAL);
	process_pixels_null(dimx, dimy);
}

void Ideal::scanline_reandering()
{
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	for (int y = 1; y < dimy; y++)
	{
		bool isin = false;
		uint32_t i = horizontal->get_offset(y), j = horizontal->get_offset(y + 1);
		for (int x = 0; x < dimx; x++)
		{
			if (show_status(get_id(x, y)) != BORDER)
			{
				if (isin)
				{
					set_status(get_id(x, y), IN);
				}
				else
				{
					set_status(get_id(x, y), OUT);
				}
				continue;
			}
			int pass = 0;
			while (i < j && horizontal->get_intersection_nodes(i) <= start_x + step_x * (x + 1))
			{
				pass++;
				i++;
			}
			if (pass % 2 == 1)
				isin = !isin;
		}
	}
}

TempPolygon intersectionDualX(Ideal *pol, double &Xi, double &Xi1)
{
	double slope;
	double yi, yi1;
	TempPolygon clippedPolygon;

	// for each edge of the polygon
	//  ITERATES THE VERTICES WHICH MUST BE IN A COUNTER-CLOCKWISE DIRECTION!!!
	//  so that right of the line means inside, left means outside
	for (auto itV = 0; itV < pol->get_num_vertices() - 1; itV++)
	{
		Point pointA = pol->get_boundary()->p[itV];
		Point pointB = pol->get_boundary()->p[itV + 1];

		// check cases
		if (pointA.x == pointB.x)
		{
			// edge is vertical
			// set the lowest point as the intersection point
			yi = min(pointA.y, pointB.y);
			yi1 = min(pointA.y, pointB.y);
		}
		else
		{
			// solve for y
			slope = getSlope(pointA, pointB);
			// check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
			// find intersection points for both Xi and Xi1
			yi = slope * (Xi - pointB.x) + pointB.y;
			yi1 = slope * (Xi1 - pointB.x) + pointB.y;
		}

		// intersection points
		Point pi(Xi, yi);
		Point pi1(Xi1, yi1);

		// cout << "X CLIPPING " << endl;
		// POINT A
		if (isInsideVerticalDual(Xi, Xi1, pointA.x))
		{
			// printf("PointA ");
			// pointA.print();
			// printf("PointB ");
			// pointB.print();
			// printf("Intersection ");
			// pi.print();
			// pointA inside, clipping result
			clippedPolygon.addPoint(pointA);
			// cout << "point(" << pointA.x << " " << pointA.y << ") added" << endl;
		}

		// intersection points in PROPER ORDER
		if (pointA.x < pointB.x)
		{
			// first pi, then pi1
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
			{
				// intersection point pi is a clipping result
				clippedPolygon.addPoint(pi);
				// cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
			}

			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
			{
				// intersection point pi1 is a clipping result
				clippedPolygon.addPoint(pi1);
				// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
			}
		}
		else
		{
			// first pi1 then pi
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
			{
				// intersection point pi1 is a clipping result
				clippedPolygon.addPoint(pi1);
				// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
			}
			if (isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
			{
				// intersection point pi is a clipping result
				clippedPolygon.addPoint(pi);
				// cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
				// printf("point pi (%.12lf %.12lf) added\n", pi.x, pi.y);
			}
		}

		// POINT B
		if (isInsideVerticalDual(Xi, Xi1, pointB.x))
		{
			// pointB inside, clipping result
			clippedPolygon.addPoint(pointB);
			// cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
		}
	}
	// add the first point to the end
	//  so that the polygon "closes"
	if (clippedPolygon.vertices.size() != 0)
	{
		clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
	}

	return clippedPolygon;
}

vector<Point> intersectionDualY(TempPolygon &pol, double Yi, double Yi1)
{
	double slope;
	double xi, xi1;
	TempPolygon clippedPolygon;

	for (auto itV = pol.vertices.begin(); itV != pol.vertices.end() - 1; itV++)
	{
		Point pointA = *itV;
		Point pointB = *(itV + 1);

		// cout << "POINTA" << endl;
		// pointA.print();
		// pointB.print();
		// cout << "POINTB" << endl;

		bool parallels = false;
		// calculate the slope (if any)
		if (pointA.x == pointB.x)
		{
			// edge is vertical
			// the only possible point of intersection is y (Yi or Yi1)
			//  do nothing
			xi = pointA.x;
			xi1 = pointA.x;
		}
		else
		{
			slope = getSlope(pointA, pointB);
			// check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
			// this flag will skip unnecessary checks in this event
			if (pointA.y == pointB.y)
			{
				if (pointA.y == Yi)
				{
					xi = pointA.x;
					xi1 = -numeric_limits<double>::max();
				}
				else if (pointA.y == Yi1)
				{
					xi = -numeric_limits<double>::max();
					xi1 = pointA.x;
				}
				else
				{
					// THEY DO NOT INETERSECT, THEY ARE PARALLELS
					parallels = true;
				}
			}
			else
			{
				// solve for x
				xi = ((Yi - pointB.y) / slope) + pointB.x;
				xi1 = ((Yi1 - pointB.y) / slope) + pointB.x;
				// cout << "   solved for x" << endl;
			}
		}

		// POINT A
		if (isInsideHorizontalDual(Yi, Yi1, pointA.y))
		{
			// pointA inside, clipping result
			// clippedPolygon.addPoint(pointA);
			clippedPolygon.vertices.push_back(pointA);
			// cout << "point A (" << pointA.x << " " << pointA.y << ") added" << endl;
		}

		if (!parallels)
		{
			// intersection points
			Point pi(xi, Yi);
			Point pi1(xi1, Yi1);

			// intersection points in PROPER ORDER
			if (pointA.y < pointB.y)
			{
				// first pi, then pi1
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
				{
					// intersection point pi is a clipping result
					// clippedPolygon.addPoint(pi);
					clippedPolygon.vertices.push_back(pi);
					// cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
				}

				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
				{
					// intersection point pi1 is a clipping result
					// clippedPolygon.addPoint(pi1);
					clippedPolygon.vertices.push_back(pi1);
					// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
				}
			}
			else
			{
				// first pi1 then pi
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
				{
					// intersection point pi1 is a clipping result
					// clippedPolygon.addPoint(pi1);
					clippedPolygon.vertices.push_back(pi1);
					// cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
				}
				if (isInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && isInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
				{
					// intersection point pi is a clipping result
					// clippedPolygon.addPoint(pi);
					clippedPolygon.vertices.push_back(pi);
					// cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
				}
			}

			// POINT B
			// if (isInsideHorizontalDual(Yi, Yi1, pointB.y))
			// {
			// 	// pointB inside, clipping result
			// 	clippedPolygon.addPoint(pointB);
			// 	// cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
			// }
		}
	}
	// cout << "sort before" << endl;
	// for (auto x : clippedPolygon.vertices)
	//     x.print();
	// sort points
	// sort_by_polar_angle(clippedPolygon.vertices);

	// cout << "sort after" << endl;
	// for (auto x : clippedPolygon.vertices)
	//     x.print();

	//"close" the polygon (first and last points in order must be the same point)
	if (clippedPolygon.vertices.size() != 0 && clippedPolygon.vertices.front() != clippedPolygon.vertices.back())
	{
		clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
	}

	return clippedPolygon.vertices;
}

// void Ideal::calculate_fullness()
// {
// 	double Xi, Xi1, Yi, Yi1;
// 	double kx, ky;

// 	vector<Point> clippedPoints;
// 	vector<TempPolygon> subpolygonsAfterX;

// 	Xi = getMBB()->low[0];
// 	Xi1 = Xi + step_x;

// 	kx = Xi + dimx * step_x;

// 	TempPolygon tempPol;
// 	subpolygonsAfterX.reserve(dimx);

// 	int x = 0;
// 	while (Xi1 < kx + 1e-9)
// 	{
// 		tempPol = intersectionDualX(this, Xi, Xi1);

// 		// if(id == 0){
// 		// 	printf("----------------------------------------------------\n");
// 		// 	printf("tempPol%d: %lf %lf\n", x, tempPol.cellX, tempPol.cellY);
// 		// 	for (auto x : tempPol.vertices)
// 		// 		x.print();
// 		// 	printf("----------------------------------------------------\n");
// 		// }

// 		if (tempPol.vertices.size() > 0)
// 		{
// 			tempPol.cellX = x;
// 			subpolygonsAfterX.push_back(tempPol);
// 		}

// 		// move both vertical lines equally
// 		Xi += step_x;
// 		Xi1 = Xi + step_x;
// 		x++;
// 	}

// 	ky = getMBB()->low[1] + dimy * step_y;

// 	int type;

// 	auto it = subpolygonsAfterX.begin();
// 	while (it != subpolygonsAfterX.end())
// 	{
// 		// FOR NORMALIZED
// 		Yi = getMBB()->low[1];
// 		Yi1 = Yi + step_y;

// 		int y = 0;
// 		// sweep the y axis getting pairs of horizontal lines Yi & Yi+1
// 		while (Yi1 < ky + 1e-9)
// 		{
// 			// returns the subpolygon furtherly clipped in the y axis by Yi and Yi+1
// 			clippedPoints = intersectionDualY(*it, Yi, Yi1);

// 			// this helps ignore a large portion of the empty cells for a polygon
// 			// if (clippedPoints.size() > 2)
// 			// {
// 			// calculate its area and classify it

// 			double clippedArea = computePolygonArea(clippedPoints);
// 			type = encodePixelArea(clippedArea);

// 			// if(id == 0){
// 			// 	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");
// 			// 	for (auto point : clippedPoints)
// 			// 		point.print();
// 			// 	printf("x = %d y = %d area = %.16lf pixelArea = %.16lf type = %d\n", it->cellX, y, clippedArea, step_x * step_y, type);

// 			// 	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");
// 			// }
// 			int pix_id = get_id(it->cellX, y);
// 			assert(pix_id < dimx * dimy);
// 			if (status[pix_id] == BORDER)
// 			{
// 				status[pix_id] = max(1, min(type, category_count - 2));
// 				areas[pix_id] = clippedArea;
// 			}
// 			else if (status[pix_id] == IN)
// 			{
// 				status[pix_id] = category_count - 1;
// 				areas[pix_id] = get_pixel_area();
// 			}else{
// 				status[pix_id] = 0;
// 				areas[pix_id] = 0.0;
// 			}

// 			// move the horizontal lines equally to the next position
// 			Yi += step_y;
// 			Yi1 = Yi + step_y;
// 			y++;
// 		}
// 		it++;
// 	}
// }

// 辅助函数：将 val 限制在 [min_v, max_v]
inline double clamp(double val, double min_v, double max_v) {
    return max(min_v, min(val, max_v));
}

void Ideal::calculate_fullness(const vector<uint8_t> &edge_mask)
{
    // 1. 初始化缓冲区
    // delta_buf: 记录该像素内发生的垂直跨度 (Delta Y)
    // area_buf:  记录该像素内边产生的局部梯形面积
	int total_pixels = dimx * dimy;
	assert(edge_mask.size() == total_pixels);
    vector<double> delta_buf(total_pixels, 0.0);
    vector<double> area_buf(total_pixels, 0.0);

    // 坐标归一化参数
    double origin_x = mbr->low[0];
    double origin_y = mbr->low[1];
    double inv_step_x = 1.0 / step_x;
    double inv_step_y = 1.0 / step_y;
    double pixel_area_val = step_x * step_y;

    // 2. 遍历所有边，进行光栅化累加
    // 我们将坐标映射到网格空间：[0, dimx] x [0, dimy]
    int num_verts = boundary->num_vertices;
    for (int i = 0; i < num_verts - 1; ++i) {
        double p1x = (boundary->p[i].x - origin_x) * inv_step_x;
        double p1y = (boundary->p[i].y - origin_y) * inv_step_y;
        double p2x = (boundary->p[i+1].x - origin_x) * inv_step_x;
        double p2y = (boundary->p[i+1].y - origin_y) * inv_step_y;
        
        if (abs(p1y - p2y) < 1e-9) continue; // 忽略水平线

        // DDA 变量
        double dx = p2x - p1x;
        double dy = p2y - p1y;
        
        // 确定遍历方向步长
        int step_x_dir = (dx > 0) ? 1 : -1;
        int step_y_dir = (dy > 0) ? 1 : -1;
        
        // 起点所在的 Grid 坐标
        int x = floor(p1x);
        int y = floor(p1y);
        int end_x = floor(p2x);
        int end_y = floor(p2y);
        
        // 射线参数 t，表示走到下一个 grid line 需要多少 t
        // x = p1x + t * dx
        // t_delta_x: 走过一个完整的 grid cell 宽度所需的 t 增量
        double t_delta_x = abs(1.0 / dx);
        double t_delta_y = abs(1.0 / dy);
        
        // t_max_x: 到达下一个 x grid line 所需的 t
        double t_max_x, t_max_y;
        
        if (dx > 0) t_max_x = (floor(p1x) + 1.0 - p1x) / dx;
        else if (dx < 0) t_max_x = (floor(p1x) - p1x) / dx;
        else t_max_x = 1e9; // 垂直线

        if (dy > 0) t_max_y = (floor(p1y) + 1.0 - p1y) / dy;
        else        t_max_y = (floor(p1y) - p1y) / dy;
        
        // 当前处理的线段起点参数 t_prev (初始为0)
        double t_prev = 0.0;
        
        // 开始遍历直到 t >= 1.0
        while (t_prev < 1.0 - 1e-9) {
            // 限制索引在范围内
            int safe_x = max(0, min(dimx - 1, x));
            int safe_y = max(0, min(dimy - 1, y));
            int idx = get_id(safe_x, safe_y);

            // 决定下一步走到哪个格子，以及这一步的 t_next 是多少
            // 取 t_max_x 和 t_max_y 中较小的那个作为交点
            double t_next;
            int next_step = 0; // 1: x, 2: y

            if (t_max_x < t_max_y) {
                t_next = t_max_x;
                next_step = 1;
            } else {
                t_next = t_max_y;
                next_step = 2;
            }
            
            // 截断到 1.0
            if (t_next > 1.0) {
                t_next = 1.0;
            }

            // 计算该片段在当前格子内的入点和出点
            // P_enter = P1 + t_prev * D
            // P_exit  = P1 + t_next * D
            double enter_y = p1y + t_prev * dy;
            double exit_y  = p1y + t_next * dy;
            double enter_x = p1x + t_prev * dx;
            double exit_x  = p1x + t_next * dx;
            
            // 计算局部坐标 (相对于 grid cell 左下角)
            // 我们只需要 X 的局部坐标来计算面积，因为我们是对 Y 积分
            // Local Area Contribution = (avg_local_x) * (delta_y)
            double avg_x = (enter_x + exit_x) * 0.5;
            double local_x = avg_x - safe_x; // 归一化到 [0, 1] (如果是 cell 外则可能超出，但公式通用)
            double delta_y = exit_y - enter_y;
            
            delta_buf[idx] += delta_y;
			area_buf[idx] += (delta_y * (1.0 - local_x));

            // 推进到下一个格子
            t_prev = t_next;
            if (next_step == 1) {
                x += step_x_dir;
                t_max_x += t_delta_x;
            } else if (next_step == 2) {
                y += step_y_dir;
                t_max_y += t_delta_y;
            }
        }
    }
    
    // 3. 最终扫描：计算前缀和并生成结果
    for (int y = 0; y < dimy; ++y)
    {
        double current_accumulated_height = 0.0;
        for (int x = 0; x < dimx; ++x)
        {
            int idx = get_id(x, y);
            
            // 核心公式：
            // 覆盖率 = 局部梯形面积 + 左侧所有高度差造成的矩形面积
            // 注意：因为积分方向问题，这里的结果是有符号的
            // 通常顺时针/逆时针会导致正负不同，取绝对值即可
            
            double val = area_buf[idx] + current_accumulated_height;
            
            // 更新累加高度，供右边的像素使用
            current_accumulated_height += delta_buf[idx];
            
            // 计算实际物理面积
            double final_area = min(pixel_area_val, max(0.0, abs(val) * pixel_area_val));
            
            // 写入结果
            areas[idx] = final_area;
            
            uint8_t fullness = encodePixelArea(final_area);
			if (edge_mask[idx] == BORDER)
			{
				status[idx] = max<uint8_t>(1, min<uint8_t>(fullness, category_count - 2));
			}
			else if (fullness == 0 || fullness == category_count - 1)
			{
				status[idx] = fullness;
				areas[idx] = (fullness == 0) ? 0.0 : pixel_area_val;
			}
			else
			{
				assert(false); // Unexpected case
				// A partial non-edge pixel would have no edge sequence to refine;
				// snap it to a terminal state to keep status/offset consistent.
				if (final_area >= pixel_area_val * 0.5)
				{
					status[idx] = category_count - 1;
					areas[idx] = pixel_area_val;
				}
				else
				{
					status[idx] = 0;
					areas[idx] = 0.0;
				}
			}
        }
    }
}


void Ideal::rasterization()
{

	// 1. create space for the pixels
	init_pixels();

	// 2. edge crossing to identify BORDER pixels
	evaluate_edges();
	vector<uint8_t> edge_mask(status, status + dimx * dimy);
	// 3. determine the status of rest pixels with scanline rendering
	// scanline_reandering();

	// 4. determine the fullness of pixels
	calculate_fullness(edge_mask);
}

void Ideal::rasterization(int vpr)
{
	assert(vpr > 0);
	pthread_mutex_lock(&ideal_partition_lock);

	if (dimx <= 0 || dimy <= 0 || step_x <= 0.0 || step_y <= 0.0)
	{
		init_raster(max(1, boundary->num_vertices / vpr));
	}
	if (status_size == 0)
	{
		set_status_size();
	}
	if (use_hierachy)
	{
		assert(num_layers > 0 && layers);
	}

	rasterization();

	if (use_hierachy)
	{
		layers[num_layers - 1].attach_base_storage(status, areas);
		for (int i = num_layers - 2; i >= 0; i--)
		{
			merge_status(layers[i], layers[i + 1]);
	 		memcpy(status + layer_offset[i], layers[i].get_status(), layers[i].get_dimx() * layers[i].get_dimy() * sizeof(uint8_t));
		}
	}

	pthread_mutex_unlock(&ideal_partition_lock);
}

int Ideal::num_edges_covered(int id)
{
	int c = 0;
	for (int i = 0; i < get_num_sequences(id); i++)
	{
		auto r = edge_sequences[offset[id] + i];
		c += r.second;
	}
	return c;
}

int Ideal::get_num_border_edge()
{
	int num = 0;
	for (int i = 0; i < get_num_pixels(); i++)
	{
		if (show_status(i) == BORDER)
		{
			num += num_edges_covered(i);
		}
	}
	return num;
}

size_t Ideal::get_num_crosses()
{
	size_t num = 0;
	num = horizontal->get_num_crosses() + vertical->get_num_crosses();
	return num;
}

double Ideal::get_shape_complexity()
{
	// if (!horizontal || !vertical)
	// {
	// 	return 0.0;
	// }
	assert(horizontal && vertical);
	size_t num_crosses = horizontal->get_num_crosses() + vertical->get_num_crosses();
	size_t num_intersected_lines = horizontal->get_num_intersected_lines() + vertical->get_num_intersected_lines();
	if (num_intersected_lines == 0)
	{
		return 0.0;
	}
	return (double)num_crosses / num_intersected_lines;
}

int Ideal::count_intersection_nodes(Point &p)
{
	// here we assume the point inside one of the pixel
	int pix_id = get_pixel_id(p);
	assert(show_status(pix_id) == BORDER);
	int count = 0;
	int x = get_x(pix_id) + 1;
	uint32_t i = vertical->get_offset(x);
	uint32_t j = x < dimx ? vertical->get_offset(x + 1) : vertical->get_num_crosses();
	while (i < j && vertical->get_intersection_nodes(i) < p.y)
	{
		count++;
		i++;
	}
	return count;
}

// double Ideal::merge_area(box target, PartitionStatus &st)
// {
// 	st = OUT;
// 	int start_x = get_offset_x(target.low[0]);
// 	int start_y = get_offset_y(target.low[1]);
// 	auto high_to_end_offset = [](double high, double low, double step, int dim) {
// 		int offset = static_cast<int>(ceil((high - low) / step - 1e-9)) - 1;
// 		if (offset < 0)
// 			return 0;
// 		if (offset >= dim)
// 			return dim - 1;
// 		return offset;
// 	};
// 	int end_x = high_to_end_offset(target.high[0], mbr->low[0], step_x, dimx);
// 	int end_y = high_to_end_offset(target.high[1], mbr->low[1], step_y, dimy);

// 	assert(start_x >= 0 && start_y >= 0 && end_x < dimx && end_y < dimy);

// 	if (end_y < start_y)
// 		end_y = start_y;
// 	if (end_x < start_x)
// 		end_x = start_x;

// 	double clippedArea = 0.0;

// 	for (int i = start_x; i <= end_x; i++)
// 	{
// 		for (int j = start_y; j <= end_y; j++)
// 		{
			
// 			int id = get_id(i, j);
// 			if(show_status(id) == BORDER) st = BORDER;
// 			clippedArea += areas[id];
// 		}
// 	}
// 	if(st != BORDER) st = show_status(get_id(start_x, start_y));
// 	// printf("clippedArea = %.12lf\n", clippedArea);

// 	return clippedArea;
// }

void Ideal::merge_status(Hraster &parent, const Hraster &child)
{
	const auto x_ranges = build_child_ranges(parent, child, true);
	const auto y_ranges = build_child_ranges(parent, child, false);
	const int parent_dimx = parent.get_dimx();
	const int child_dimx = child.get_dimx();
	const double parent_pixel_area = parent.get_pixel_area();
	const double *child_areas = child.get_areas();

	for (int y = 0; y < parent.get_dimy(); ++y)
	{
		const auto [child_start_y, child_end_y] = y_ranges[y];
		const int parent_row = y * parent_dimx;
		for (int x = 0; x < parent_dimx; ++x)
		{
			const auto [child_start_x, child_end_x] = x_ranges[x];
			double merged_area = 0.0;

			for (int child_y = child_start_y; child_y <= child_end_y; ++child_y)
			{
				const int child_row = child_y * child_dimx;
				for (int child_x = child_start_x; child_x <= child_end_x; ++child_x)
				{
					merged_area += child_areas[child_row + child_x];
				}
			}

			const int parent_id = parent_row + x;
			parent.set_area(parent_id, merged_area);
			parent.set_status(parent_id, classifyPixel(merged_area, parent_pixel_area, category_count));
		}
	}
}

void Ideal::layering(int NLow)
{
	if (NLow < 1)
		NLow = 1;

	struct LayerParam {
		int dx, dy;
		double sx, sy;
		box mbr;
	};

	    std::vector<LayerParam> params;

	    int current_dx = get_dimx();
	    int current_dy = get_dimy();
	    double current_sx = get_step_x();
	    double current_sy = get_step_y();
	    box current_mbr = *getMBB();

	    params.push_back({current_dx, current_dy, current_sx, current_sy, current_mbr});

	    while (true) {
	        size_t current_size = static_cast<size_t>(current_dx) * current_dy;
        
        if (current_size <= static_cast<size_t>(NLow)) {
            break;
        }

	        current_dx = (current_dx + 1) / 2;
	        current_dy = (current_dy + 1) / 2;
	        current_sx *= 2.0;
	        current_sy *= 2.0;
	        current_mbr.high[0] = current_mbr.low[0] + current_dx * current_sx;
	        current_mbr.high[1] = current_mbr.low[1] + current_dy * current_sy;

	        params.push_back({current_dx, current_dy, current_sx, current_sy, current_mbr});
	    }

    num_layers = static_cast<int>(params.size());
    
    layers = new Hraster[num_layers];
    layer_offset = new uint32_t[num_layers];
    layer_info = new RasterInfo[num_layers];

    size_t current_offset_accum = 0; 
    
	    for (int i = 0; i < num_layers; ++i) {
	        int target_idx = num_layers - 1 - i;
	        
	        auto& p = params[i];
	        
	        bool is_base = (i == 0);
	        box *layer_mbr = is_base ? getMBB() : &p.mbr;
	        layers[target_idx].init(p.sx, p.sy, p.dx, p.dy, layer_mbr, is_base);
	        layers[target_idx].set_category_count(category_count);
	        layer_info[target_idx] = {*layers[target_idx].mbr, p.dx, p.dy, p.sx, p.sy};
	        layer_offset[target_idx] = static_cast<uint32_t>(current_offset_accum);
	        
	        current_offset_accum += (static_cast<size_t>(p.dx) * p.dy);
	    }
    
    status_size = current_offset_accum;
}

bool Ideal::contain(Point &p, query_context *ctx)
{

	// the MBB may not be checked for within query
	if (!mbr->contain(p))
	{
		return false;
	}

	// todo adjust the lower bound of pixel number when the raster model is usable
	int target = get_pixel_id(p);
	box bx = get_pixel_box(get_x(target), get_y(target));
	double bx_high = bx.high[0];
	if (show_status(target) == IN)
	{
		return true;
	}
	if (show_status(target) == OUT)
	{
		return false;
	}

	bool ret = false;

	// checking the intersection edges in the target pixel
	for (uint32_t e = 0; e < get_num_sequences(target); e++)
	{
		auto edges = get_edge_sequence(get_offset(target) + e);
		auto pos = edges.first;
		for (int k = 0; k < edges.second; k++)
		{
			int i = pos + k;
			int j = i + 1; // ATTENTION
			if (((boundary->p[i].y >= p.y) != (boundary->p[j].y >= p.y)))
			{
				double int_x = (boundary->p[j].x - boundary->p[i].x) * (p.y - boundary->p[i].y) / (boundary->p[j].y - boundary->p[i].y) + boundary->p[i].x;
				if (p.x <= int_x && int_x <= bx_high)
				{
					ret = !ret;
				}
			}
		}
	}
	// check the crossing nodes on the right bar
	// swap the state of ret if odd number of intersection
	// nodes encountered at the right side of the border
	struct timeval tstart = get_cur_time();
	int nc = count_intersection_nodes(p);
	if (nc % 2 == 1)
	{
		ret = !ret;
	}
	return ret;
}

bool Ideal::contain(Ideal *target, query_context *ctx, bool profile){
	if(!getMBB()->contain(*target->getMBB())){
		//log("mbb do not contain");
		return false;
	}

	vector<int> pxs = retrieve_pixels(target->getMBB());
	int etn = 0;
	int itn = 0;
	for(auto p : pxs){
		if(show_status(p) == OUT){
			etn++;
		}else if(show_status(p) == IN){
			itn++;
		}
	}
	if(etn == pxs.size()){
		return false;
	}
	if(itn == pxs.size()){
		return true;
	}

	vector<int> tpxs;

	for(auto p : pxs){
		box bx =  get_pixel_box(get_x(p), get_y(p));
		tpxs = target->retrieve_pixels(&bx);
		for(auto p2 : tpxs){
			// an external pixel of the container intersects an internal
			// pixel of the containee, which means the containment must be false
			if(show_status(p) == IN) continue;
			if(show_status(p) == OUT && target->show_status(p2) == IN){
				return false;
			}
			if (show_status(p) == OUT && target->show_status(p2) == BORDER){
				Point pix_border[5];
				pix_border[0].x = bx.low[0]; pix_border[0].y = bx.low[1];
				pix_border[1].x = bx.low[0]; pix_border[1].y = bx.high[1];
				pix_border[2].x = bx.high[0]; pix_border[2].y = bx.high[1];
				pix_border[3].x = bx.high[0]; pix_border[3].y = bx.low[1];
				pix_border[4].x = bx.low[0]; pix_border[4].y = bx.low[1];
				for (int e = 0; e < target->get_num_sequences(p2); e++){
					auto edges = target->get_edge_sequence(target->get_offset(p2) + e);
					auto pos = edges.first;
					auto size = edges.second;
					if (segment_intersect_batch(target->boundary->p + pos, pix_border, size, 4)){
						return false;
					}
				}
			}
			// evaluate the state
			if(show_status(p) == BORDER && target->show_status(p2) == BORDER){
				for(int i = 0; i < get_num_sequences(p); i ++){
					auto r = get_edge_sequence(get_offset(p) + i);
					for(int j = 0; j < target->get_num_sequences(p2); j ++){
						auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
						if(segment_intersect_batch(boundary->p+r.first, target->boundary->p+r2.first, r.second, r2.second)){
							return false;
						}
					}
				}
			}
		}
		tpxs.clear();
	}
	pxs.clear();

	// this is the last step for all the cases, when no intersection segment is identified
	// pick one point from the target and it must be contained by this polygon
	Point p(target->getx(0),target->gety(0));
	return contain(p, ctx);
}

PartitionStatus Ideal::segment_contain(Point &p)
{
	int target = get_pixel_id(p);

	box bx = get_pixel_box(get_x(target), get_y(target));
	double bx_high = bx.high[0];
	if (show_status(target) == IN)
	{
		return IN;
	}
	if (show_status(target) == OUT)
	{
		return OUT;
	}

	bool ret = false;

	// checking the intersection edges in the target pixel
	for (uint32_t e = 0; e < get_num_sequences(target); e++)
	{
		auto edges = get_edge_sequence(get_offset(target) + e);
		auto pos = edges.first;
		for (int k = 0; k < edges.second; k++)
		{
			Point v1 = boundary->p[pos + k];
			Point v2 = boundary->p[pos + k + 1];
			// if (abs(p.x - 133.967605) < 1e-9 && abs(p.y - 34.558846) < 1e-9)
			// {
			// 	printf("----------------------CHECK-----------------------------\n");
			// 	p.print();
			// 	v1.print();
			// 	v2.print();
			// 	printf("----------------------CHECK-----------------------------\n");
			// }

			if (p == v1 || p == v2)
			{
				// printf("OUTPUT1\n");
				// p.print();
				// v1.print();
				// v2.print();
				return BORDER;
			}

			if ((v1.y >= p.y) != (v2.y >= p.y))
			{

				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (abs(dy) > 1e-9)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if (fabs(p.x - int_x) < 1e-9)
					{
						// printf("OUTPUT2\n");
						// p.print();
						// v1.print();
						// v2.print();
						return BORDER;
					}
					if (p.x < int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
			else if (v1.y == p.y && v2.y == p.y && (v1.x >= p.x) != (v2.x >= p.x))
			{
				// printf("OUTPUT3\n");
				// p.print();
				// v1.print();
				// v2.print();
				return BORDER;
			}
		}
	}
	// check the crossing nodes on the right bar
	// swap the state of ret if odd number of intersection
	// nodes encountered at the right side of the border
	int nc = count_intersection_nodes(p);
	if (nc % 2 == 1)
	{
		ret = !ret;
	}
	if (ret)
	{
		return IN;
	}
	else
	{
		return OUT;
	}
}

bool Ideal::intersect(Ideal *target, query_context *ctx, bool approximation)
{
	auto start = std::chrono::high_resolution_clock::now();
	vector<tuple<double, int, int>> candidate_pairs;

	vector<int> pxs = retrieve_pixels(target->getMBB());
	for (auto pa : pxs)
	{
		if(show_status(pa) == OUT) continue;
		box bx = get_pixel_box(get_x(pa), get_y(pa));
		vector<int> tpxs = target->retrieve_pixels(&bx);
		for (auto pb : tpxs)
		{
			// evaluate the state
			if (target->show_status(pb) == OUT) continue;
			if(show_status(pa) == IN || target->show_status(pb) == IN) {
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> duration = end - start;
				ctx->raster_filter_time += duration.count();
				return true;
			}
			assert(show_status(pa) == BORDER && target->show_status(pb) == BORDER);
			auto s_fullness = get_fullness(pa), t_fullness = target->get_fullness(pb);
			auto s_p_low = decodePixelArea(pa, true);
			auto s_p_high = decodePixelArea(pa, false);
			auto t_p_low = target->decodePixelArea(pb, true);
			auto t_p_high = target->decodePixelArea(pb, false);
			if(s_p_low + t_p_low >= max(get_pixel_area(), target->get_pixel_area())){
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> duration = end - start;
				ctx->raster_filter_time += duration.count();
				return true;
			}
			if(!approximation){
				auto s_p_apx = (s_p_low + s_p_high) / 2;
				auto t_p_apx = (t_p_low + t_p_high) / 2;
				auto prob = (s_p_apx + t_p_apx) / max(get_pixel_area(), target->get_pixel_area());
				candidate_pairs.push_back({prob, pa, pb});
			}
		}
	}
	// for(int pair_id = 0; pair_id < candidate_pairs.size(); pair_id ++){
	// 	auto prob = get<0>(candidate_pairs[pair_id]);
	// 	auto pa = get<1>(candidate_pairs[pair_id]);
	// 	auto pb = get<2>(candidate_pairs[pair_id]);

	// 	if(prob > 1.0) continue;
	// 	auto pf = classifyPixel(prob, 20);
		
	// 	std::string filename = "class_" + std::to_string(pf) + ".txt";
	// 	std::ofstream outfile(filename, std::ios::app);

	// 	for (int i = 0; i < get_num_sequences(pa); i++)
	// 	{
	// 		auto r = get_edge_sequence(get_offset(pa) + i);
	// 		for (int j = 0; j < target->get_num_sequences(pb); j++)
	// 		{
	// 			auto r2 = target->get_edge_sequence(target->get_offset(pb) + j);
	// 			if (segment_intersect_batch(boundary->p + r.first, target->boundary->p + r2.first, r.second, r2.second))
	// 			{
	// 				outfile << std::fixed << std::setprecision(6) << 1 << endl;
	// 			}else{
	// 				outfile << std::fixed << std::setprecision(6) << 0 << endl;
	// 			}
	// 		}
	// 	}
	// }

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	ctx->raster_filter_time += duration.count();

	if(approximation){
		return false;
	}

	// // printf("%d\n", candidate_pairs.size());

	start = std::chrono::high_resolution_clock::now();

	sort(candidate_pairs.begin(), candidate_pairs.end(), [](const auto& a, const auto& b) {
		return a > b;
	});

	for(int pair_id = 0; pair_id < candidate_pairs.size(); pair_id ++){
		auto pa = get<1>(candidate_pairs[pair_id]);
		auto pb = get<2>(candidate_pairs[pair_id]);
		for (int i = 0; i < get_num_sequences(pa); i++)
		{
			auto r = get_edge_sequence(get_offset(pa) + i);
			for (int j = 0; j < target->get_num_sequences(pb); j++)
			{
				auto r2 = target->get_edge_sequence(target->get_offset(pb) + j);
				if (segment_intersect_batch(boundary->p + r.first, target->boundary->p + r2.first, r.second, r2.second))
				{
					end = std::chrono::high_resolution_clock::now();
					duration = end - start;
					ctx->refine_time += duration.count();
					return true;
				}
			}
		}
	}

	if(contain(target->get_boundary()->p[0], ctx)){
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		ctx->refine_time += duration.count();
		return true;
	}

	if(target->contain(boundary->p[0], ctx)){
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		ctx->refine_time += duration.count();
		return true;
	}

	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	ctx->refine_time += duration.count();

	return false;
}

inline int binary_search(vector<Segment> &sorted_array, int left, int right, Point target)
{
	while (left < right)
	{
		int mid = (left + right) >> 1;
		if (target <= sorted_array[mid].start)
			right = mid;
		else
			left = mid + 1;
	}
	if (sorted_array[left].start == target)
		return left;
	else
		return -1; // Not Found
}

void Ideal::intersection(Ideal *target, query_context *ctx)
{
	auto start = std::chrono::high_resolution_clock::now();
	query_context *output_ctx = ctx->global_ctx;
	size_t area_idx = ctx->target_id;
	if (output_ctx->areas && area_idx < output_ctx->num_pairs)
		output_ctx->areas[area_idx] = 0.0;

	vector<int> pxs = retrieve_pixels(target->getMBB());
	vector<Intersection> inters;

	for (auto p : pxs)
	{
		if (show_status(p) != BORDER)
			continue;

		box bx = get_pixel_box(get_x(p), get_y(p));
		vector<int> tpxs = target->retrieve_pixels(&bx);
		for (auto p2 : tpxs)
		{
			if (target->show_status(p2) != BORDER)
				continue;

			for (int i = 0; i < get_num_sequences(p); i++)
			{
				auto r = get_edge_sequence(get_offset(p) + i);
				for (int j = 0; j < target->get_num_sequences(p2); j++)
				{
					auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
					assert(r.second != 0 && r2.second != 0);
					collect_intersections(boundary->p, target->boundary->p,
										  r.first, r.second, r2.first, r2.second,
										  get_num_vertices(), target->get_num_vertices(),
										  inters);
				}
			}
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	ctx->raster_filter_time += duration.count();

	if (inters.empty())
		return;

	start = std::chrono::high_resolution_clock::now();

	sort_unique_source_intersections(inters);
	int num_inters = inters.size();
	if (num_inters == 0)
		return;

	for (int i = 0; i < num_inters; i++)
	{
		Intersection &inter = inters[i];
		Intersection &next = inters[(i + 1) % num_inters];
		inter.status = classify_intersection_arc(inter, next, boundary->p, get_num_vertices(), target, true);
	}

	double area = accumulate_area_for_order(inters, boundary->p, get_num_vertices(),
										   boundary->p, get_num_vertices(),
										   target->boundary->p, target->get_num_vertices(),
										   true);

	sort_target_intersections(inters);
	for (int i = 0; i < num_inters; i++)
	{
		Intersection &inter = inters[i];
		Intersection &next = inters[(i + 1) % num_inters];
		inter.status = classify_intersection_arc(inter, next, target->boundary->p, target->get_num_vertices(), this, false);
	}

	area += accumulate_area_for_order(inters, target->boundary->p, target->get_num_vertices(),
									  boundary->p, get_num_vertices(),
									  target->boundary->p, target->get_num_vertices(),
									  false);

	if (output_ctx->areas && area_idx < output_ctx->num_pairs)
		output_ctx->areas[area_idx] = area;

	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	ctx->refine_time += duration.count();
	return;
}

double Ideal::get_possible_min(Point &p, int center, int step, bool geography)
{
	int core_x_low = get_x(center);
	int core_x_high = get_x(center);
	int core_y_low = get_y(center);
	int core_y_high = get_y(center);

	vector<int> needprocess;

	int ymin = max(0, core_y_low - step);
	int ymax = min(dimy, core_y_high + step);

	double mindist = DBL_MAX;
	// left scan
	if (core_x_low - step >= 0)
	{
		double x = get_pixel_box(core_x_low - step, ymin).high[0];
		double y1 = get_pixel_box(core_x_low - step, ymin).low[1];
		double y2 = get_pixel_box(core_x_low - step, ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// right scan
	if (core_x_high + step <= get_dimx())
	{
		double x = get_pixel_box(core_x_high + step, ymin).low[0];
		double y1 = get_pixel_box(core_x_high + step, ymin).low[1];
		double y2 = get_pixel_box(core_x_high + step, ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0, core_x_low - step + (core_x_low - step >= 0));
	// skip the last if there is right scan
	int xmax = min(dimx, core_x_high + step - (core_x_high + step <= dimx));
	// bottom scan
	if (core_y_low - step >= 0)
	{
		double y = get_pixel_box(xmin, core_y_low - step).high[1];
		double x1 = get_pixel_box(xmin, core_y_low - step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low - step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// top scan
	if (core_y_high + step <= get_dimy())
	{
		double y = get_pixel_box(xmin, core_y_low + step).low[1];
		double x1 = get_pixel_box(xmin, core_y_low + step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low + step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::get_possible_min(box *t_mbr, int core_x_low, int core_y_low, int core_x_high, int core_y_high, int step, bool geography)
{

	vector<int> needprocess;

	int ymin = max(0, core_y_low - step);
	int ymax = min(dimy, core_y_high + step);

	double mindist = DBL_MAX;
	// left scan
	if (core_x_low - step >= 0)
	{
		double x = get_pixel_box(core_x_low - step, ymin).high[0];
		double y1 = get_pixel_box(core_x_low - step, ymin).low[1];
		double y2 = get_pixel_box(core_x_low - step, ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// right scan
	if (core_x_high + step <= get_dimx())
	{
		double x = get_pixel_box(core_x_high + step, ymin).low[0];
		double y1 = get_pixel_box(core_x_high + step, ymin).low[1];
		double y2 = get_pixel_box(core_x_high + step, ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0, core_x_low - step + (core_x_low - step >= 0));
	// skip the last if there is right scan
	int xmax = min(dimx, core_x_high + step - (core_x_high + step <= dimx));
	// bottom scan
	if (core_y_low - step >= 0)
	{
		double y = get_pixel_box(xmin, core_y_low - step).high[1];
		double x1 = get_pixel_box(xmin, core_y_low - step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low - step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// top scan
	if (core_y_high + step <= get_dimy())
	{
		double y = get_pixel_box(xmin, core_y_low + step).low[1];
		double x1 = get_pixel_box(xmin, core_y_low + step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low + step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::distance(Point &p, query_context *ctx, bool profile)
{
	// distance is 0 if contained by the polygon
	double mindist = getMBB()->max_distance(p, ctx->geography);

	bool contained = contain(p, ctx);
	if (contained)
	{
		return 0;
	}

	double mbrdist = mbr->distance(p, ctx->geography);

	// initialize the starting pixel
	int closest = get_closest_pixel(p);

	int step = 0;
	double step_size = get_step(ctx->geography);
	vector<int> needprocess;

	while (true)
	{
		if (step == 0)
		{
			needprocess.push_back(closest);
		}
		else
		{
			needprocess = expand_radius(closest, step);
		}
		// should never happen
		// all the boxes are scanned
		if (needprocess.size() == 0)
		{
			assert(false && "should not evaluated all boxes");
			return boundary->distance(p, ctx->geography);
		}
		for (auto cur : needprocess)
		{
			// printf("checking pixel %d %d %d\n",cur->id[0],cur->id[1],cur->status);
			if (show_status(cur) == BORDER)
			{
				box cur_box = get_pixel_box(get_x(cur), get_y(cur));
				// printf("BOX: lowx=%lf, lowy=%lf, highx=%lf, highy=%lf\n", cur_box.low[0], cur_box.low[1], cur_box.high[0], cur_box.high[1]);
				double mbr_dist = cur_box.distance(p, ctx->geography);
				// skip the pixels that is further than the current minimum
				if (mbr_dist >= mindist)
				{
					continue;
				}

				// the vector model need be checked.

				for (int i = 0; i < get_num_sequences(cur); i++)
				{
					auto rg = get_edge_sequence(get_offset(cur) + i);
					for (int j = 0; j < rg.second; j++)
					{
						auto r = rg.first + j;
						double dist = point_to_segment_distance(p, *get_point(r), *get_point(r + 1), ctx->geography);
						mindist = min(mindist, dist);
						if (ctx->within(mindist))
						{
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();

		// for within query, return if the current minimum is close enough
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = get_possible_min(p, closest, step, ctx->geography);
		// close enough
		if (mindist < minrasterdist)
		{
			break;
		}
	}
	// IDEAL return
	return mindist;
}

// get the distance from pixel pix to polygon target
double Ideal::distance(Ideal *target, int pix, query_context *ctx, bool profile)
{
	assert(show_status(pix) == BORDER);

	auto pix_x = get_x(pix);
	auto pix_y = get_y(pix);
	auto pix_box = get_pixel_box(pix_x, pix_y);
	double mindist = getMBB()->max_distance(pix_box, ctx->geography);
	double mbrdist = getMBB()->distance(pix_box, ctx->geography);
	int step = 0;

	// initialize the seed closest pixels
	vector<int> needprocess = target->get_closest_pixels(pix_box);
	assert(needprocess.size() > 0);
	unsigned short lowx = target->get_x(needprocess[0]);
	unsigned short highx = target->get_x(needprocess[0]);
	unsigned short lowy = target->get_y(needprocess[0]);
	unsigned short highy = target->get_y(needprocess[0]);
	for (auto p : needprocess)
	{
		lowx = min(lowx, (unsigned short)target->get_x(p));
		highx = max(highx, (unsigned short)target->get_x(p));
		lowy = min(lowy, (unsigned short)target->get_y(p));
		highy = max(highy, (unsigned short)target->get_y(p));
	}

	while (true)
	{
		// for later steps, expand the circle to involve more pixels
		if (step > 0)
		{
			needprocess = target->expand_radius(lowx, highx, lowy, highy, step);
		}

		// all the boxes are scanned (should never happen)
		if (needprocess.size() == 0)
		{
			return mindist;
		}

		for (auto cur : needprocess)
		{
			// note that there is no need to check the edges of
			// this pixel if it is too far from the target
			auto cur_x = target->get_x(cur);
			auto cur_y = target->get_y(cur);

			if (target->show_status(cur) == BORDER)
			{
				bool toofar = (target->get_pixel_box(cur_x, cur_y).distance(pix_box, ctx->geography) >= mindist);
				if (toofar)
				{
					continue;
				}
				// the vector model need be checked.
				for (int i = 0; i < get_num_sequences(pix); i++)
				{
					auto pix_er = get_edge_sequence(get_offset(pix) + i);
					for (int j = 0; j < target->get_num_sequences(cur); j++)
					{
						auto cur_er = target->get_edge_sequence(target->get_offset(cur) + j);
						if (cur_er.second < 2 || pix_er.second < 2)
							continue;
						double dist;
						if (ctx->is_within_query())
						{
							dist = segment_to_segment_within_batch(target->boundary->p + cur_er.first,
																   boundary->p + pix_er.first, cur_er.second, pix_er.second,
																   ctx->within_distance, ctx->geography);
						}
						else
						{
							dist = segment_sequence_distance(target->boundary->p + cur_er.first,
															 boundary->p + pix_er.first, cur_er.second, pix_er.second, ctx->geography);
						}
						mindist = min(dist, mindist);
						if (ctx->within(mindist))
						{
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = target->get_possible_min(&pix_box, lowx, lowy, highx, highy, step, ctx->geography);
		if (mindist < minrasterdist)
			break;
	}

	return mindist;
}

// TEMP 
bool IsInsideHorizontalDual(double Yi, double Yi1, double py)
{
    if (py < Yi || py > Yi1)
    {
        return false;
    }
    return true;
}

bool IsInsideVerticalDual(double Xi, double Xi1, double px)
{
    if (px < Xi || px > Xi1)
    {
        return false;
    }
    return true;
}

TempPolygon IntersectionDualX(MyPolygon &pol, double Xi, double Xi1)
{
    double slope;
    double yi, yi1;
    TempPolygon clippedPolygon;

    // for each edge of the polygon
    //  ITERATES THE VERTICES WHICH MUST BE IN A COUNTER-CLOCKWISE DIRECTION!!!
    //  so that right of the line means inside, left means outside
    for (auto itV = 0; itV < pol.get_num_vertices() - 1; itV++)
    {
        Point pointA = pol.get_boundary()->p[itV];
        Point pointB = pol.get_boundary()->p[itV + 1];

        // check cases
        if (pointA.x == pointB.x)
        {
            // edge is vertical
            // set the lowest point as the intersection point
            yi = min(pointA.y, pointB.y);
            yi1 = min(pointA.y, pointB.y);
        }
        else
        {
            // solve for y
            slope = getSlope(pointA, pointB);
            // check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
            // find intersection points for both Xi and Xi1
            yi = slope * (Xi - pointB.x) + pointB.y;
            yi1 = slope * (Xi1 - pointB.x) + pointB.y;
        }

        // intersection points
        Point pi(Xi, yi);
        Point pi1(Xi1, yi1);
        // cout << "X CLIPPING " << endl;
        // POINT A
        if (IsInsideVerticalDual(Xi, Xi1, pointA.x))
        {
            // pointA inside, clipping result
            clippedPolygon.addPoint(pointA);
            // cout << "point(" << pointA.x << " " << pointA.y << ") added" << endl;
        }

        // intersection points in PROPER ORDER
        if (pointA.x < pointB.x)
        {
            // first pi, then pi1
            if (IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
            {
                // intersection point pi is a clipping result
                clippedPolygon.addPoint(pi);
                // cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
            }

            if (IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
            {
                // intersection point pi1 is a clipping result
                clippedPolygon.addPoint(pi1);
                // cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
            }
        }
        else
        {
            // first pi1 then pi
            if (IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y) && IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x))
            {
                // intersection point pi1 is a clipping result
                clippedPolygon.addPoint(pi1);
                // cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added2" << endl;
            }
            if (IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y) && IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x))
            {
                // intersection point pi is a clipping result
                clippedPolygon.addPoint(pi);
                // cout << "point pi (" << pi.x << " " << pi.y << ") added2" << endl;
            }
        }

        // POINT B
        if (IsInsideVerticalDual(Xi, Xi1, pointB.x))
        {
            // pointB inside, clipping result
            clippedPolygon.addPoint(pointB);
            // cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
        }
    }
    // add the first point to the end
    //  so that the polygon "closes"
    if (clippedPolygon.vertices.size() != 0)
    {
        clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
    }

    return clippedPolygon;
}

vector<Point> IntersectionDualY(TempPolygon &pol, double Yi, double Yi1)
{
    double slope;
    double xi, xi1;
    TempPolygon clippedPolygon;

    for (auto itV = pol.vertices.begin(); itV != pol.vertices.end() - 1; itV++)
    {
        Point pointA = *itV;
        Point pointB = *(itV + 1);

        // cout << "POINTA" << endl;
        // pointA.print();
        // pointB.print();
        // cout << "POINTB" << endl;

        bool parallels = false;
        // calculate the slope (if any)
        if (pointA.x == pointB.x)
        {
            // edge is vertical
            // the only possible point of intersection is y (Yi or Yi1)
            //  do nothing
            xi = pointA.x;
            xi1 = pointA.x;
        }
        else
        {
            slope = getSlope(pointA, pointB);
            // check if AB is horizontal, if it is it only intersects if pointA.y == Yi or pointA.y == Yi1
            // this flag will skip unnecessary checks in this event
            if (pointA.y == pointB.y)
            {
                if (pointA.y == Yi)
                {
                    xi = pointA.x;
                    xi1 = -numeric_limits<double>::max();
                }
                else if (pointA.y == Yi1)
                {
                    xi = -numeric_limits<double>::max();
                    xi1 = pointA.x;
                }
                else
                {
                    // THEY DO NOT INETERSECT, THEY ARE PARALLELS
                    parallels = true;
                }
            }
            else
            {
                // solve for x
                xi = ((Yi - pointB.y) / slope) + pointB.x;
                xi1 = ((Yi1 - pointB.y) / slope) + pointB.x;
                // cout << "   solved for x" << endl;
            }
        }

        // POINT A
        if (IsInsideHorizontalDual(Yi, Yi1, pointA.y))
        {
            // pointA inside, clipping result
            // clippedPolygon.addPoint(pointA);
            clippedPolygon.vertices.push_back(pointA);
            // cout << "point A (" << pointA.x << " " << pointA.y << ") added" << endl;
        }

        if (!parallels)
        {
            // intersection points
            Point pi(xi, Yi);
            Point pi1(xi1, Yi1);

            // intersection points in PROPER ORDER
            if (pointA.y < pointB.y)
            {
                // first pi, then pi1
                if (IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
                {
                    // intersection point pi is a clipping result
                    // clippedPolygon.addPoint(pi);
                    clippedPolygon.vertices.push_back(pi);
                    // cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
                }

                if (IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
                {
                    // intersection point pi1 is a clipping result
                    // clippedPolygon.addPoint(pi1);
                    clippedPolygon.vertices.push_back(pi1);
                    // cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
                }
            }
            else
            {
                // first pi1 then pi
                if (IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi1.x) && IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi1.y))
                {
                    // intersection point pi1 is a clipping result
                    // clippedPolygon.addPoint(pi1);
                    clippedPolygon.vertices.push_back(pi1);
                    // cout << "point pi1 (" << pi1.x << " " << pi1.y << ") added" << endl;
                }
                if (IsInsideVerticalDual(min(pointA.x, pointB.x), max(pointA.x, pointB.x), pi.x) && IsInsideHorizontalDual(min(pointA.y, pointB.y), max(pointA.y, pointB.y), pi.y))
                {
                    // intersection point pi is a clipping result
                    // clippedPolygon.addPoint(pi);
                    clippedPolygon.vertices.push_back(pi);
                    // cout << "point pi (" << pi.x << " " << pi.y << ") added" << endl;
                }
            }

            // POINT B
            // if (IsInsideHorizontalDual(Yi, Yi1, pointB.y))
            // {
            //     // pointB inside, clipping result
            //     clippedPolygon.addPoint(pointB);
            //     // cout << "point B (" << pointB.x << " " << pointB.y << ") added" << endl;
            // }
        }
    }
    // cout << "sort before" << endl;
    // for (auto x : clippedPolygon.vertices)
    //     x.print();
    // sort points
    // sort_by_polar_angle(clippedPolygon.vertices);

    // cout << "sort after" << endl;
    // for (auto x : clippedPolygon.vertices)
    //     x.print();

    //"close" the polygon (first and last points in order must be the same point)
    if (clippedPolygon.vertices.size() != 0 && clippedPolygon.vertices.front() != clippedPolygon.vertices.back())
    {
        clippedPolygon.vertices.push_back(*clippedPolygon.vertices.begin());
    }

    return clippedPolygon.vertices;
}

void RasterizePolygon(MyPolygon *mappedPol, Ideal *ideal, vector<vector<Point>> &clippedPolygons)
{
    double Xi, Xi1, Yi, Yi1;
    double kx, ky;
    double step_x = ideal->get_step_x(), step_y = ideal->get_step_y();
    int dimx = ideal->get_dimx(), dimy = ideal->get_dimy();

    vector<Point> clippedPoints;
    vector<TempPolygon> subpolygonsAfterX;

    // initialize the x axis sweep lines
    Xi = ideal->getMBB()->low[0];
    Xi1 = Xi + step_x;

    // define the end x
    kx = Xi + dimx * step_x;
    // cout << Xi << " and " << Xi1 << "/" << kx << endl;

    TempPolygon tempPol;
    subpolygonsAfterX.reserve(dimx);

    int x = 0;
    // sweep the x axis getting pairs of vertical lines Xi & Xi+1
    while (Xi1 < kx + 1e-9)
    {
        // cout << Xi << " and " << Xi1 << "/" << kx << endl;
        // returns the sub polygon when pol is clipped by Xi and Xi1
        tempPol = IntersectionDualX(*mappedPol, Xi, Xi1);
        // add the sub polygon to the polygon's list
        //  we need to save it to further clip it in the y axis later
        // printf("----------------------------------------------------\n");
        // printf("tempPol%d\n", x);
        // for (auto x : tempPol.vertices)
        //     x.print();
        // printf("----------------------------------------------------\n");

        if (tempPol.vertices.size() > 0)
        {
            tempPol.cellX = x;
            subpolygonsAfterX.push_back(tempPol);
        }

        // move both vertical lines equally
        Xi += step_x;
        Xi1 = Xi + step_x;
        x++;
    }

    // define the end y
    ky = ideal->getMBB()->low[1] + dimy * step_y;

    int type;

    // iterate the subpolygons created by the x axis clipping instead of the original polygons
    auto it = subpolygonsAfterX.begin();
    while (it != subpolygonsAfterX.end())
    {
        // FOR NORMALIZED
        Yi = ideal->getMBB()->low[1];
        Yi1 = Yi + step_y;

        int y = 0;
        // sweep the y axis getting pairs of horizontal lines Yi & Yi+1
        while (Yi1 < ky + 1e-9)
        {
            // returns the subpolygon furtherly clipped in the y axis by Yi and Yi+1
            clippedPoints = IntersectionDualY(*it, Yi, Yi1);

            // this helps ignore a large portion of the empty cells for a polygon
            // if (clippedPoints.size() > 2)
            // {
            // calculate its area and classify it

            // double clippedArea = computePolygonArea(clippedPoints);

            // printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");
            // for (auto point : clippedPoints)
            //     point.print();

            // printf("x = %d y = %d area = %.12lf pixelArea = %.12lf type = %d\n", it->cellX, y, clippedArea, step_x * step_y, type);

            // printf("--------------------------------------------------------------------------------------------------------------------------------------------------------\n");

            // if (type != 0)
            // {
            // printf("CHECK = %d %d    %d\n", it->cellX, y, y * (dimx + 1) + it->cellX);
            clippedPolygons[y * dimx + it->cellX] = clippedPoints;
            // rasterizationCells.emplace_back(it->cellX, Yi, it->cellX + step_x, Yi1, type);
            // }
            // }

            // move the horizontal lines equally to the next position
            Yi += step_y;
            Yi1 = Yi + step_y;
            y++;
        }
        it++;
    }
    return;
}

// TEMP

bool Ideal::within(Ideal *target, query_context *ctx)
{
	using Clock = std::chrono::high_resolution_clock;
	auto total_start = Clock::now();
	auto refine_start = total_start;
	bool raster_done = false;
	auto elapsed_ms = [](Clock::time_point start, Clock::time_point end) {
		return std::chrono::duration<double, std::milli>(end - start).count();
	};
	auto finish_raster = [&]() {
		if(raster_done) return;
		auto now = Clock::now();
		ctx->raster_filter_time += elapsed_ms(total_start, now);
		refine_start = now;
		raster_done = true;
	};
	auto finish_within = [&](bool result) {
		auto now = Clock::now();
		if(raster_done){
			ctx->refine_time += elapsed_ms(refine_start, now);
		}else{
			ctx->raster_filter_time += elapsed_ms(total_start, now);
		}
		ctx->within_time += elapsed_ms(total_start, now);
		return result;
	};
	uint s_level = num_layers - 1;
	uint t_level = target->get_num_layers() - 1;
	auto get_retrieve_range = [](Hraster &raster, box target_box, int &start_x, int &end_x, int &start_y, int &end_y) {
		start_x = raster.get_offset_x(target_box.low[0] + 1e-6);
		start_y = raster.get_offset_y(target_box.low[1] + 1e-6);
		end_x = raster.get_offset_x(target_box.high[0] - 1e-6);
		end_y = raster.get_offset_y(target_box.high[1] - 1e-6);
	};

	auto init_start = Clock::now();
	queue<pair<int, int>> pairs;
	size_t source_top_pixels = layers[0].get_num_pixels();
	size_t target_top_pixels = target->layers[0].get_num_pixels();
	if(source_top_pixels == 1 && target_top_pixels == 1){
		pairs.push(make_pair(0, 0));
	}else{
		for(int i = 0; i < source_top_pixels; i ++){
			auto source_box = layers[0].get_pixel_box(layers[0].get_x(i), layers[0].get_y(i));
			auto expanded_source_box = source_box.expand(ctx->within_distance, true);
			int t_start_x = 0, t_end_x = 0, t_start_y = 0, t_end_y = 0;
			get_retrieve_range(target->layers[0], expanded_source_box, t_start_x, t_end_x, t_start_y, t_end_y);
			int target_dimx = target->layers[0].get_dimx();
			for(int x = t_start_x; x <= t_end_x; x ++){
				for(int y = t_start_y; y <= t_end_y; y ++){
					int j = y * target_dimx + x;
					auto target_box = target->layers[0].get_pixel_box(x, y);
					if(source_box.distance(target_box, true) <= ctx->within_distance){
						pairs.push(make_pair(i, j));
					}
				}
			}
		}
	}
	ctx->within_init_time += elapsed_ms(init_start, Clock::now());

	int i = 0, j = 0;
	int level = 0;
	while(true){
		// cout << "level: " << ++ level << endl;
		bool s_next_layer = false, t_next_layer = false;
		double s_step = layers[i].get_step_x(), t_step = target->get_layers()[j].get_step_x();

		if(i < s_level && (s_step >= t_step || j >= t_level)) {
			i ++;
			s_next_layer = true;
		}
		if(j < t_level && (s_step <= t_step || i >= s_level)) {
			j ++;
			t_next_layer = true;
		}

		int size = pairs.size();
		if(size == 0) break;
		size_t expected_pairs = size;
		if(s_next_layer) expected_pairs *= 4;
		if(t_next_layer) expected_pairs *= 4;
		vector<pair<int, int>> expanded_pairs;
		expanded_pairs.reserve(expected_pairs);
		auto &source_layer = layers[i];
		auto &target_layer = target->get_layers()[j];
		auto source_status = source_layer.get_status();
		auto target_status = target_layer.get_status();
		int source_dimx = source_layer.get_dimx();
		int target_dimx = target_layer.get_dimx();
		uint8_t in_status = ctx->category_count - 1;
		auto expand_start = Clock::now();
		for(int k = 0; k < size; k ++){
			auto pair = pairs.front();
			// printf("id = (%d %d)\n", pair.first, pair.second);
			pairs.pop();
			int s_pix_id = pair.first, t_pix_id = pair.second;
			int s_start_x = 0, s_end_x = 0, s_start_y = 0, s_end_y = 0;
			int t_start_x = 0, t_end_x = 0, t_start_y = 0, t_end_y = 0;

			if(s_next_layer){
				auto source_pixel_box = layers[i - 1].get_pixel_box(layers[i - 1].get_x(s_pix_id), layers[i - 1].get_y(s_pix_id));

				source_pixel_box.low[0] += 1e-6;
				source_pixel_box.low[1] += 1e-6;
				source_pixel_box.high[0] -= 1e-6;
				source_pixel_box.high[1] -= 1e-6;

				get_retrieve_range(source_layer, source_pixel_box, s_start_x, s_end_x, s_start_y, s_end_y);
			}else if(source_status[s_pix_id] == 0 || source_status[s_pix_id] == in_status){
				continue;
			}

			if(t_next_layer){
				auto target_pixel_box = target->get_layers()[j - 1].get_pixel_box(target->get_layers()[j - 1].get_x(t_pix_id), target->get_layers()[j - 1].get_y(t_pix_id));

				target_pixel_box.low[0] += 1e-6;
				target_pixel_box.low[1] += 1e-6;
				target_pixel_box.high[0] -= 1e-6;
				target_pixel_box.high[1] -= 1e-6;

				get_retrieve_range(target_layer, target_pixel_box, t_start_x, t_end_x, t_start_y, t_end_y);
			}else if(target_status[t_pix_id] == 0 || target_status[t_pix_id] == in_status){
				continue;
			}

			if(s_next_layer){
				for(int y1 = s_start_y; y1 <= s_end_y; y1 ++){
					int id1 = y1 * source_dimx + s_start_x;
					for(int x1 = s_start_x; x1 <= s_end_x; x1 ++, id1 ++){
						if(source_status[id1] == 0 || source_status[id1] == in_status) continue;
						if(t_next_layer){
							for(int y2 = t_start_y; y2 <= t_end_y; y2 ++){
								int id2 = y2 * target_dimx + t_start_x;
								for(int x2 = t_start_x; x2 <= t_end_x; x2 ++, id2 ++){
									if(target_status[id2] != 0 && target_status[id2] != in_status){
										expanded_pairs.emplace_back(id1, id2);
									}
								}
							}
						}else{
							expanded_pairs.emplace_back(id1, t_pix_id);
						}
					}
				}
			}else if(t_next_layer){
				for(int y2 = t_start_y; y2 <= t_end_y; y2 ++){
					int id2 = y2 * target_dimx + t_start_x;
					for(int x2 = t_start_x; x2 <= t_end_x; x2 ++, id2 ++){
						if(target_status[id2] != 0 && target_status[id2] != in_status){
							expanded_pairs.emplace_back(s_pix_id, id2);
						}
					}
				}
			}else{
				expanded_pairs.emplace_back(s_pix_id, t_pix_id);
			}
		}
		ctx->within_expand_time += elapsed_ms(expand_start, Clock::now());

		vector<tuple<double, int, int>> filtered_pairs;
		filtered_pairs.reserve(expanded_pairs.size());
		auto distance_start = Clock::now();
		double max_box_dist = 100000.0;
		int last_source_id = -1;
		box cached_source_box;
		for(auto pair : expanded_pairs){
			int id1 = pair.first, id2 = pair.second;
			if(id1 != last_source_id){
				cached_source_box = fast_pixel_box(source_layer, id1, source_dimx);
				last_source_id = id1;
			}
			auto box2 = fast_pixel_box(target_layer, id2, target_dimx);

			double max_distacne = fast_box_max_distance(cached_source_box, box2, true);
			if(max_distacne <= ctx->within_distance){
				ctx->within_distance_time += elapsed_ms(distance_start, Clock::now());
				return finish_within(true);
			}
			max_box_dist = min(max_box_dist, max_distacne);

			double min_distance = cached_source_box.distance(box2, true);
			if(min_distance > ctx->within_distance) continue;
			filtered_pairs.push_back({min_distance, id1, id2});
		}
		for(auto pair : filtered_pairs){
			if(get<0>(pair) < max_box_dist) {
				pairs.push(make_pair(get<1>(pair), get<2>(pair)));
			}
		}
		ctx->within_distance_time += elapsed_ms(distance_start, Clock::now());
		if(!s_next_layer && !t_next_layer) break;
	}

	vector<tuple<double, double, double, int, int>> candidate_pairs;
	auto candidate_start = Clock::now();
	int base_source_dimx = get_dimx();
	int base_target_dimx = target->get_dimx();
	while(!pairs.empty()){
		auto pair = pairs.front();
		pairs.pop();
		int id1 = pair.first, id2 = pair.second;
		auto s_fr = get_fullness(id1), t_fr = target->get_fullness(id2);
		auto s_p_apx = (decodePixelArea(id1, true) + decodePixelArea(id1, false)) / 2;
		auto t_p_apx = (target->decodePixelArea(id2, true) + target->decodePixelArea(id2, false)) / 2;
		auto pf = classifyPixel(get_areas(id1) + target->get_areas(id2), get_pixel_area() + target->get_pixel_area(), 20);

		auto box1 = fast_pixel_box(*this, id1, base_source_dimx);
		auto box2 = fast_pixel_box(*target, id2, base_target_dimx);
		double dist_low = box1.distance(box2, true);
		double dist_high = fast_box_max_distance(box1, box2, true);

		double mean = (1 - pf / 20.0) * 0.55;
		double dist_apx = dist_low + mean * (dist_high - dist_low);
		candidate_pairs.push_back({dist_apx, dist_low, dist_high, id1, id2});
	}
	ctx->within_candidate_time += elapsed_ms(candidate_start, Clock::now());

	finish_raster();

	sort(candidate_pairs.begin(), candidate_pairs.end());

	vector<double> suffix_min_dist(candidate_pairs.size() + 1, 100000.0);
	for(int i = (int)candidate_pairs.size() - 1; i >= 0; i --){
		suffix_min_dist[i] = min(get<1>(candidate_pairs[i]), suffix_min_dist[i + 1]);
	}

	double min_dist = 100000.0;
	for(int pair_id = 0; pair_id < candidate_pairs.size(); pair_id ++){
		double reference_dist = suffix_min_dist[pair_id + 1];
		auto id1 = get<3>(candidate_pairs[pair_id]);
		auto id2 = get<4>(candidate_pairs[pair_id]);
		uint32_t source_num_sequences = get_num_sequences(id1);
		uint32_t target_num_sequences = target->get_num_sequences(id2);
		size_t edge_count = 0;
		for(uint32_t i = 0; i < source_num_sequences; i ++){
			edge_count += get_edge_sequence(get_offset(id1) + i).second;
		}
		for(uint32_t j = 0; j < target_num_sequences; j ++){
			edge_count += target->get_edge_sequence(target->get_offset(id2) + j).second;
		}
		ctx->within_refine_pixel_pairs ++;
		ctx->within_refine_edges += edge_count;

		for(uint32_t i = 0; i < source_num_sequences; i ++){
			auto er1 = get_edge_sequence(get_offset(id1) + i);
			for(uint32_t j = 0; j < target_num_sequences; j ++){
				auto er2 = target->get_edge_sequence(target->get_offset(id2) + j);
				if(er1.second < 2 || er2.second < 2) continue;
				double dist = segment_to_segment_within_batch(target->boundary->p+er2.first,
									boundary->p+er1.first, er2.second, er1.second,
									ctx->within_distance, ctx->geography);

				min_dist = min(dist, min_dist);
				if(min_dist < ctx->within_distance){
					return finish_within(true);
				}
			}
		}
		if(min_dist >= ctx->within_distance && min_dist < reference_dist){
			return finish_within(false);
		}
	}

	return finish_within(false);
}

// bool Ideal::intersect_o(Ideal *target, query_context *ctx){

// 	if(!getMBB()->intersect(*target->getMBB())){
// 		return false;
// 	}

// 	vector<int> pxs = retrieve_pixels(target->getMBB());
// 	int etn = 0;
// 	int itn = 0;
// 	vector<int> bpxs;
// 	for(auto p : pxs){
// 		if(show_status(p) == OUT){
// 			etn++;
// 		}else if(show_status(p)==IN){
// 			itn++;
// 		}else{
// 			bpxs.push_back(p);
// 		}
// 	}
// 	//log("%d %d %d",etn,itn,pxs.size());
// 	if(etn == pxs.size()){
// 		return false;
// 	}
// 	if(itn == pxs.size()){
// 		return true;
// 	}

// 	vector<pair<int, int>> candidates;
// 	vector<int> bpxs2;
// 	for(auto p : bpxs){
// 		bpxs2 = target->retrieve_pixels(&get_pixel_box(get_x(p), get_y(p)));
// 		// cannot determine anything; e-i e-e e-b
// 		if(show_status(p) == OUT){
// 			continue;
// 		}
// 		for(auto p2 : bpxs2){

// 			// nothing specific; i-e b-e
// 			if(target->show_status(p2) == OUT){
// 				continue;
// 			}

// 			// must intersect; i-i
// 			if(show_status(p) == IN && target->show_status(p2) == IN){
// 				return true;
// 			}

// 			// b-b b-i i-b
// 			candidates.push_back(make_pair(p, p2));
// 		}
// 		bpxs2.clear();
// 	}


// 	for(auto pa : candidates){
// 		int p = pa.first;
// 		int p2 = pa.second;
// 		if(show_status(p) == BORDER && target->show_status(p2) == BORDER){
// 			for (int i = 0; i < get_num_sequences(p); i++){
// 				auto r = get_edge_sequence(get_offset(p) + i);
// 				for (int j = 0; j < target->get_num_sequences(p2); j++)
// 				{
// 					auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
// 					if (segment_intersect_batch(boundary->p + r.first, target->boundary->p + r2.first, r.second, r2.second))
// 					{
// 						return true;
// 					}
// 				}
// 			}
// 		} else if (show_status(p) == BORDER){
// 			return true;
// 		} else {
// 			return true;
// 		}
// 	}

// 	return segment_intersect_batch(this->boundary->p, target->boundary->p, this->boundary->num_vertices, target->boundary->num_vertices);
// }
