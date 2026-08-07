#include "../include/farm.h"

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

double fast_box_max_distance_sq(const box &source, double target_lowx, double target_lowy,
	double target_highx, double target_highy, double inv_lon, double inv_lat)
{
	double dx = max(fabs(source.low[0] - target_highx), fabs(source.high[0] - target_lowx)) * inv_lon;
	double dy = max(fabs(source.low[1] - target_highy), fabs(source.high[1] - target_lowy)) * inv_lat;
	return dx * dx + dy * dy;
}

double fast_box_min_distance_sq(const box &source, double target_lowx, double target_lowy,
	double target_highx, double target_highy, double inv_lon, double inv_lat)
{
	double dx = 0.0;
	double dy = 0.0;

	if(target_lowx > source.high[0]){
		dx = target_lowx - source.high[0];
	}else if(target_highx < source.low[0]){
		dx = source.low[0] - target_highx;
	}

	if(target_lowy > source.high[1]){
		dy = target_lowy - source.high[1];
	}else if(target_highy < source.low[1]){
		dy = source.low[1] - target_highy;
	}

	dx *= inv_lon;
	dy *= inv_lat;
	return dx * dx + dy * dy;
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

struct ChildPixels {
	int ids[4];
	int count = 0;
};

bool near_equal(double lhs, double rhs)
{
	const double scale = max(1.0, max(fabs(lhs), fabs(rhs)));
	return fabs(lhs - rhs) <= scale * 1e-9;
}

bool can_use_direct_child_pixels(const Hraster &parent, const Hraster &child)
{
	return parent.get_dimx() == (child.get_dimx() + 1) / 2
		&& parent.get_dimy() == (child.get_dimy() + 1) / 2
		&& near_equal(parent.get_step_x(), child.get_step_x() * 2.0)
		&& near_equal(parent.get_step_y(), child.get_step_y() * 2.0)
		&& near_equal(parent.mbr->low[0], child.mbr->low[0])
		&& near_equal(parent.mbr->low[1], child.mbr->low[1]);
}

ChildPixels direct_child_pixels(int parent_id, int parent_dimx, int child_dimx, int child_dimy)
{
	ChildPixels child_pixels;
	int parent_y = parent_id / parent_dimx;
	int parent_x = parent_id - parent_y * parent_dimx;
	int child_x = parent_x * 2;
	int child_y = parent_y * 2;
	int base = child_y * child_dimx + child_x;
	bool has_right = child_x + 1 < child_dimx;
	bool has_top = child_y + 1 < child_dimy;

	child_pixels.ids[child_pixels.count ++] = base;
	if(has_right){
		child_pixels.ids[child_pixels.count ++] = base + 1;
	}
	if(has_top){
		int top_base = base + child_dimx;
		child_pixels.ids[child_pixels.count ++] = top_base;
		if(has_right){
			child_pixels.ids[child_pixels.count ++] = top_base + 1;
		}
	}
	return child_pixels;
}

struct WithinCandidate {
	double dist_apx;
	double dist_low;
	double dist_high;
	int source_id;
	int target_id;

	bool operator<(const WithinCandidate &other) const
	{
		return tie(dist_apx, dist_low, dist_high, source_id, target_id)
			< tie(other.dist_apx, other.dist_low, other.dist_high, other.source_id, other.target_id);
	}
};

double calculate_within_probability(const WithinCandidate &candidate, double within_distance)
{
	if(within_distance < candidate.dist_low) return 0.0;
	if(within_distance >= candidate.dist_high) return 1.0;

	double span = candidate.dist_high - candidate.dist_low;
	if(span <= 0.0) return within_distance >= candidate.dist_low ? 1.0 : 0.0;

	double ratio = (within_distance - candidate.dist_low) / span;
	double mean = (candidate.dist_apx - candidate.dist_low) / span;
	ratio = min(1.0, max(0.0, ratio));
	mean = min(1.0, max(0.0, mean));

	double stddev = 0.3 * mean;
	if(stddev <= 0.0) return ratio >= mean ? 1.0 : 0.0;

	double probability = 0.5 * (1.0 + erf((ratio - mean) / (stddev * sqrt(2.0))));
	return min(1.0, max(0.0, probability));
}

bool approximate_within_candidates(const vector<WithinCandidate> &candidate_pairs, query_context *ctx)
{
	vector<double> probabilities;
	probabilities.reserve(candidate_pairs.size());
	for(const WithinCandidate &candidate : candidate_pairs){
		probabilities.push_back(calculate_within_probability(candidate, ctx->within_distance));
	}

	auto sort_timer = ctx->profiler.scoped(ProfileStage::WithinRefineCandidateSort);
	sort(probabilities.begin(), probabilities.end(), greater<double>());
	sort_timer.stop();

	double no_within_probability = 1.0;
	for(double probability : probabilities){
		no_within_probability *= 1.0 - probability;
		if(1.0 - no_within_probability >= ctx->approx_confidence){
			return true;
		}
	}
	return false;
}

struct WithinLayerPlan {
	bool source_next_layer = false;
	bool target_next_layer = false;
	size_t expected_pairs = 0;

	Hraster *source_layer = nullptr;
	Hraster *target_layer = nullptr;

	int source_dimx = 0;
	int source_dimy = 0;
	int target_dimx = 0;
	int target_dimy = 0;
	int source_parent_dimx = 0;
	int target_parent_dimx = 0;

	bool source_direct_child_pixels = false;
	bool target_direct_child_pixels = false;
	vector<pair<int, int>> source_child_x_ranges;
	vector<pair<int, int>> source_child_y_ranges;
	vector<pair<int, int>> target_child_x_ranges;
	vector<pair<int, int>> target_child_y_ranges;
};

void seed_within_initial_pairs(Farm *source, Farm *target, query_context *ctx,
	vector<pair<int, int>> &pairs)
{
	auto timer = ctx->profiler.scoped(ProfileStage::WithinRasterInit);
	Hraster *source_layers = source->get_layers();
	Hraster *target_layers = target->get_layers();
	Hraster &source_top_layer = source_layers[0];
	Hraster &target_top_layer = target_layers[0];
	size_t source_top_pixels = source_top_layer.get_num_pixels();
	size_t target_top_pixels = target_top_layer.get_num_pixels();
	// if(source_top_pixels == 1 && target_top_pixels == 1){
	// 	pairs.emplace(0, 0);
	// 	return;
	// }

	for(size_t source_id = 0; source_id < source_top_pixels; source_id ++){
		int i = static_cast<int>(source_id);
		auto source_box = source_top_layer.get_pixel_box(source_top_layer.get_x(i), source_top_layer.get_y(i));
		for(size_t target_id = 0; target_id < target_top_pixels; target_id ++){
			int j = static_cast<int>(target_id);
			auto target_box = target_top_layer.get_pixel_box(target_top_layer.get_x(j), target_top_layer.get_y(j));
			if(source_box.distance(target_box, ctx->geography) <= ctx->within_distance){
				pairs.emplace_back(i, j);
			}
		}
	}
}

WithinLayerPlan build_within_layer_plan(Farm *source, Farm *target, int &source_level, int &target_level,
	size_t current_pair_count)
{
	Hraster *source_layers = source->get_layers();
	Hraster *target_layers = target->get_layers();
	uint source_last_level = source->get_num_layers() - 1;
	uint target_last_level = target->get_num_layers() - 1;
	WithinLayerPlan plan;

	double source_step = source_layers[source_level].get_step_x();
	double target_step = target_layers[target_level].get_step_x();
	if(source_level < source_last_level && (source_step >= target_step || target_level >= target_last_level)) {
		source_level ++;
		plan.source_next_layer = true;
	}
	if(target_level < target_last_level && (source_step <= target_step || source_level >= source_last_level)) {
		target_level ++;
		plan.target_next_layer = true;
	}

	plan.expected_pairs = current_pair_count;
	if(plan.source_next_layer) plan.expected_pairs *= 4;
	if(plan.target_next_layer) plan.expected_pairs *= 4;

	plan.source_layer = &source_layers[source_level];
	plan.target_layer = &target_layers[target_level];
	plan.source_dimx = plan.source_layer->get_dimx();
	plan.source_dimy = plan.source_layer->get_dimy();
	plan.target_dimx = plan.target_layer->get_dimx();
	plan.target_dimy = plan.target_layer->get_dimy();

	if(plan.source_next_layer){
		auto &source_parent_layer = source_layers[source_level - 1];
		plan.source_parent_dimx = source_parent_layer.get_dimx();
		plan.source_direct_child_pixels = can_use_direct_child_pixels(source_parent_layer, *plan.source_layer);
		if(!plan.source_direct_child_pixels){
			plan.source_child_x_ranges = build_child_ranges(source_parent_layer, *plan.source_layer, true);
			plan.source_child_y_ranges = build_child_ranges(source_parent_layer, *plan.source_layer, false);
		}
	}
	if(plan.target_next_layer){
		auto &target_parent_layer = target_layers[target_level - 1];
		plan.target_parent_dimx = target_parent_layer.get_dimx();
		plan.target_direct_child_pixels = can_use_direct_child_pixels(target_parent_layer, *plan.target_layer);
		if(!plan.target_direct_child_pixels){
			plan.target_child_x_ranges = build_child_ranges(target_parent_layer, *plan.target_layer, true);
			plan.target_child_y_ranges = build_child_ranges(target_parent_layer, *plan.target_layer, false);
		}
	}

	return plan;
}

class WithinExpandContext {
public:
	WithinExpandContext(const WithinLayerPlan &plan, query_context *ctx, vector<pair<int, int>> &next_pairs)
		: plan(plan), next_pairs(next_pairs),
		  in_status(status_max_value(static_cast<uint8_t>(ctx->bitwidth))),
		  within_distance_sq((double)ctx->within_distance * ctx->within_distance),
		  geography(ctx->geography),
		  inv_lat(ctx->geography ? 1.0 / degree_per_kilometer_latitude : 1.0),
		  target_lowx(plan.target_layer->mbr->low[0]),
		  target_lowy(plan.target_layer->mbr->low[1]),
		  target_step_x(plan.target_layer->get_step_x()),
		  target_step_y(plan.target_layer->get_step_y())
	{
	}

	bool process_pair_with_target_coords(int source_id, int target_id, double box2_lowx, double box2_lowy,
		double box2_highx, double box2_highy)
	{
		if(source_id != last_source_id){
			cached_source_box = fast_pixel_box(*plan.source_layer, source_id, plan.source_dimx);
			cached_inv_lon = geography
				? 1.0 / degree_per_kilometer_longitude(cached_source_box.low[1])
				: 1.0;
			last_source_id = source_id;
		}

		double max_distance_sq = fast_box_max_distance_sq(cached_source_box, box2_lowx, box2_lowy,
			box2_highx, box2_highy, cached_inv_lon, inv_lat);
		if(max_distance_sq <= within_distance_sq){
			return true;
		}
		double min_distance_sq = fast_box_min_distance_sq(cached_source_box, box2_lowx, box2_lowy,
			box2_highx, box2_highy, cached_inv_lon, inv_lat);
		if(min_distance_sq <= within_distance_sq){
			next_pairs.emplace_back(source_id, target_id);
		}
		return false;
	}

	bool process_pair_by_id(int source_id, int target_id)
	{
		int target_y = target_id / plan.target_dimx;
		int target_x = target_id - target_y * plan.target_dimx;
		double box2_lowx = target_lowx + target_x * target_step_x;
		double box2_lowy = target_lowy + target_y * target_step_y;
		return process_pair_with_target_coords(source_id, target_id, box2_lowx, box2_lowy,
			box2_lowx + target_step_x, box2_lowy + target_step_y);
	}

	bool process_target_children(int source_id, const ChildPixels &target_children)
	{
		for(int target_child_idx = 0; target_child_idx < target_children.count; target_child_idx ++){
			int target_id = target_children.ids[target_child_idx];
			const uint8_t target_status = plan.target_layer->get_fullness(target_id);
			if(target_status != 0 && target_status != in_status){
				if(process_pair_by_id(source_id, target_id)){
					return true;
				}
			}
		}
		return false;
	}

	bool process_target_range(int source_id, int start_x, int end_x, int start_y, int end_y)
	{
		for(int y2 = start_y; y2 <= end_y; y2 ++){
			int target_id = y2 * plan.target_dimx + start_x;
			double box2_lowy = target_lowy + y2 * target_step_y;
			double box2_highy = box2_lowy + target_step_y;
			double box2_lowx = target_lowx + start_x * target_step_x;
			for(int x2 = start_x; x2 <= end_x; x2 ++, target_id ++, box2_lowx += target_step_x){
				const uint8_t target_status = plan.target_layer->get_fullness(target_id);
				if(target_status != 0 && target_status != in_status){
					if(process_pair_with_target_coords(source_id, target_id, box2_lowx, box2_lowy,
						box2_lowx + target_step_x, box2_highy)){
						return true;
					}
				}
			}
		}
		return false;
	}

	uint8_t get_in_status() const
	{
		return in_status;
	}

	private:
	const WithinLayerPlan &plan;
	vector<pair<int, int>> &next_pairs;
	uint8_t in_status;
	double within_distance_sq;
	bool geography;
	double inv_lat;
	int last_source_id = -1;
	box cached_source_box;
	double cached_inv_lon = 1.0;
	double target_lowx;
	double target_lowy;
	double target_step_x;
	double target_step_y;
};

bool process_target_for_source(const WithinLayerPlan &plan, WithinExpandContext &expand_ctx, int source_id,
	int target_pix_id, const ChildPixels &target_children, int t_start_x, int t_end_x, int t_start_y, int t_end_y)
{
	if(plan.target_next_layer){
		if(plan.target_direct_child_pixels){
			return expand_ctx.process_target_children(source_id, target_children);
		}
		return expand_ctx.process_target_range(source_id, t_start_x, t_end_x, t_start_y, t_end_y);
	}
	return expand_ctx.process_pair_by_id(source_id, target_pix_id);
}

bool expand_within_pairs(const WithinLayerPlan &plan, WithinExpandContext &expand_ctx,
	const vector<pair<int, int>> &pairs)
{
	uint8_t in_status = expand_ctx.get_in_status();
	for(const auto &pair : pairs){
		int source_pix_id = pair.first;
		int target_pix_id = pair.second;
		ChildPixels source_children, target_children;
		int s_start_x = 0, s_end_x = 0, s_start_y = 0, s_end_y = 0;
		int t_start_x = 0, t_end_x = 0, t_start_y = 0, t_end_y = 0;

		if(plan.source_next_layer){
			if(plan.source_direct_child_pixels){
				source_children = direct_child_pixels(source_pix_id, plan.source_parent_dimx,
					plan.source_dimx, plan.source_dimy);
			}else{
				int source_parent_y = source_pix_id / plan.source_parent_dimx;
				int source_parent_x = source_pix_id - source_parent_y * plan.source_parent_dimx;
				auto source_x_range = plan.source_child_x_ranges[source_parent_x];
				auto source_y_range = plan.source_child_y_ranges[source_parent_y];
				s_start_x = source_x_range.first;
				s_end_x = source_x_range.second;
				s_start_y = source_y_range.first;
				s_end_y = source_y_range.second;
			}
		}else if(plan.source_layer->get_fullness(source_pix_id) == 0 ||
			plan.source_layer->get_fullness(source_pix_id) == in_status){
			continue;
		}

		if(plan.target_next_layer){
			if(plan.target_direct_child_pixels){
				target_children = direct_child_pixels(target_pix_id, plan.target_parent_dimx,
					plan.target_dimx, plan.target_dimy);
			}else{
				int target_parent_y = target_pix_id / plan.target_parent_dimx;
				int target_parent_x = target_pix_id - target_parent_y * plan.target_parent_dimx;
				auto target_x_range = plan.target_child_x_ranges[target_parent_x];
				auto target_y_range = plan.target_child_y_ranges[target_parent_y];
				t_start_x = target_x_range.first;
				t_end_x = target_x_range.second;
				t_start_y = target_y_range.first;
				t_end_y = target_y_range.second;
			}
		}else if(plan.target_layer->get_fullness(target_pix_id) == 0 ||
			plan.target_layer->get_fullness(target_pix_id) == in_status){
			continue;
		}

		if(plan.source_next_layer){
			if(plan.source_direct_child_pixels){
				for(int source_child_idx = 0; source_child_idx < source_children.count; source_child_idx ++){
					int source_id = source_children.ids[source_child_idx];
					const uint8_t source_status = plan.source_layer->get_fullness(source_id);
					if(source_status == 0 || source_status == in_status) continue;
					if(process_target_for_source(plan, expand_ctx, source_id, target_pix_id, target_children,
						t_start_x, t_end_x, t_start_y, t_end_y)){
						return true;
					}
				}
			}else{
				for(int y1 = s_start_y; y1 <= s_end_y; y1 ++){
					int source_id = y1 * plan.source_dimx + s_start_x;
					for(int x1 = s_start_x; x1 <= s_end_x; x1 ++, source_id ++){
						const uint8_t source_status = plan.source_layer->get_fullness(source_id);
						if(source_status == 0 || source_status == in_status) continue;
						if(process_target_for_source(plan, expand_ctx, source_id, target_pix_id, target_children,
							t_start_x, t_end_x, t_start_y, t_end_y)){
							return true;
						}
					}
				}
			}
		}else if(plan.target_next_layer){
			if(plan.target_direct_child_pixels){
				if(expand_ctx.process_target_children(source_pix_id, target_children)) return true;
			}else{
				if(expand_ctx.process_target_range(source_pix_id, t_start_x, t_end_x, t_start_y, t_end_y)) return true;
			}
		}else if(expand_ctx.process_pair_by_id(source_pix_id, target_pix_id)){
			return true;
		}
	}
	return false;
}

void build_within_refine_candidates(Farm *source, Farm *target, query_context *ctx,
	const vector<pair<int, int>> &pairs, vector<WithinCandidate> &candidate_pairs)
{
	auto timer = ctx->profiler.scoped(ProfileStage::WithinRefineCandidateBuild);
	int base_source_dimx = source->get_dimx();
	int base_target_dimx = target->get_dimx();
	for(const auto &pair : pairs){
		int source_id = pair.first;
		int target_id = pair.second;
		auto pf = classifyPixel(source->get_areas(source_id) + target->get_areas(target_id),
			source->get_pixel_area() + target->get_pixel_area(),
			static_cast<uint8_t>(ctx->bitwidth));

		auto source_box = fast_pixel_box(*source, source_id, base_source_dimx);
		auto target_box = fast_pixel_box(*target, target_id, base_target_dimx);
		double dist_low = source_box.distance(target_box, ctx->geography);
		double dist_high = fast_box_max_distance(source_box, target_box, ctx->geography);
		double mean = (1 - pf / static_cast<double>(
			status_category_count(static_cast<uint8_t>(ctx->bitwidth)))) * 0.55;
		double dist_apx = dist_low + mean * (dist_high - dist_low);
		candidate_pairs.push_back({dist_apx, dist_low, dist_high, source_id, target_id});
	}
}

vector<double> build_within_suffix_min_dist(query_context *ctx, const vector<WithinCandidate> &candidate_pairs)
{
	auto timer = ctx->profiler.scoped(ProfileStage::WithinRefineSuffix);
	vector<double> suffix_min_dist(candidate_pairs.size() + 1,
		std::numeric_limits<double>::infinity());
	for(size_t i = candidate_pairs.size(); i-- > 0;){
		suffix_min_dist[i] = min(candidate_pairs[i].dist_low, suffix_min_dist[i + 1]);
	}
	return suffix_min_dist;
}

bool refine_within_candidates(Farm *source, Farm *target, query_context *ctx,
	const vector<WithinCandidate> &candidate_pairs, const vector<double> &suffix_min_dist)
{
	double min_dist = std::numeric_limits<double>::infinity();
	for(size_t pair_id = 0; pair_id < candidate_pairs.size(); pair_id ++){
		const WithinCandidate &candidate = candidate_pairs[pair_id];
		double reference_dist = suffix_min_dist[pair_id + 1];
		uint32_t source_num_sequences = source->get_num_sequences(candidate.source_id);
		uint32_t target_num_sequences = target->get_num_sequences(candidate.target_id);
		uint32_t source_sequence_offset = source->get_offset(candidate.source_id);
		uint32_t target_sequence_offset = target->get_offset(candidate.target_id);
#if FARM_PROFILE_QUERY
		{
			auto timer = ctx->profiler.scoped(ProfileStage::WithinRefinePrepare);
			ctx->profiler.add_count(ProfileCount::WithinRefinePixels, 2);
			for(uint32_t i = 0; i < source_num_sequences; i ++){
				ctx->profiler.add_count(ProfileCount::WithinRefineEdges,
					source->get_edge_sequence(source_sequence_offset + i).second);
			}
			for(uint32_t j = 0; j < target_num_sequences; j ++){
				ctx->profiler.add_count(ProfileCount::WithinRefineEdges,
					target->get_edge_sequence(target_sequence_offset + j).second);
			}
		}
#endif

		auto exact_timer = ctx->profiler.scoped(ProfileStage::WithinRefineExact);
		for(uint32_t i = 0; i < source_num_sequences; i ++){
			auto source_range = source->get_edge_sequence(source_sequence_offset + i);
			for(uint32_t j = 0; j < target_num_sequences; j ++){
				auto target_range = target->get_edge_sequence(target_sequence_offset + j);
				if(source_range.second < 2 || target_range.second < 2) continue;
				double dist = segment_to_segment_within_batch(target->get_boundary()->p + target_range.first,
					source->get_boundary()->p + source_range.first, target_range.second, source_range.second,
					ctx->within_distance, ctx->geography);

				min_dist = min(dist, min_dist);
				if(min_dist <= ctx->within_distance){
					exact_timer.stop();
					return true;
				}
			}
		}
		exact_timer.stop();
		if(min_dist > ctx->within_distance && min_dist < reference_dist){
			return false;
		}
	}
	return false;
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

double clamp_intersection_param(double value)
{
	if(value < 0.0) return 0.0;
	if(value > 1.0) return 1.0;
	return value;
}

double signed_double_area(Point *vertices, int num_vertices)
{
	double area = 0.0;
	for(int i = 0; i < num_vertices - 1; i ++){
		area += (double)vertices[i].x * vertices[i + 1].y
			- (double)vertices[i].y * vertices[i + 1].x;
	}
	return area;
}

double positive_double_area(Point *vertices, int num_vertices)
{
	return fabs(signed_double_area(vertices, num_vertices));
}

bool point_on_segment(const Point &p, const Point &a, const Point &b)
{
	const double cross = (double)(p.y - a.y) * (b.x - a.x) - (double)(p.x - a.x) * (b.y - a.y);
	if(fabs(cross) > 1e-9){
		return false;
	}
	const double dot = (double)(p.x - a.x) * (b.x - a.x) + (double)(p.y - a.y) * (b.y - a.y);
	if(dot < -1e-9){
		return false;
	}
	const double len_sq = (double)(b.x - a.x) * (b.x - a.x) + (double)(b.y - a.y) * (b.y - a.y);
	return dot <= len_sq + 1e-9;
}

PartitionStatus point_in_ring(const Point &p, Point *vertices, int num_vertices)
{
	bool inside = false;
	for(int i = 0; i < num_vertices - 1; i ++){
		const Point &a = vertices[i];
		const Point &b = vertices[i + 1];
		if(point_on_segment(p, a, b)){
			return BORDER;
		}
		if((a.y > p.y) != (b.y > p.y)){
			double int_x = (double)(b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
			if(int_x > p.x){
				inside = !inside;
			}
		}
	}
	return inside ? IN : OUT;
}

bool ring_covers_ring(Farm *container, Farm *subject)
{
	if(!container->getMBB()->contain(*subject->getMBB())){
		return false;
	}

	bool has_inside_point = false;
	Point *container_vertices = container->get_boundary()->p;
	Point *subject_vertices = subject->get_boundary()->p;
	int container_num_vertices = container->get_num_vertices();
	int subject_num_vertices = subject->get_num_vertices();

	for(int i = 0; i < subject_num_vertices - 1; i ++){
		PartitionStatus status = point_in_ring(subject_vertices[i], container_vertices, container_num_vertices);
		if(status == OUT){
			return false;
		}
		if(status == IN){
			has_inside_point = true;
		}

		Point mid((subject_vertices[i].x + subject_vertices[i + 1].x) * 0.5f,
			(subject_vertices[i].y + subject_vertices[i + 1].y) * 0.5f);
		status = point_in_ring(mid, container_vertices, container_num_vertices);
		if(status == OUT){
			return false;
		}
		if(status == IN){
			has_inside_point = true;
		}
	}

	return has_inside_point || near_equal(positive_double_area(container_vertices, container_num_vertices),
		positive_double_area(subject_vertices, subject_num_vertices));
}

double no_crossing_intersection_area(Farm *source, Farm *target)
{
	if(ring_covers_ring(source, target)){
		return positive_double_area(target->get_boundary()->p, target->get_num_vertices());
	}
	if(ring_covers_ring(target, source)){
		return positive_double_area(source->get_boundary()->p, source->get_num_vertices());
	}
	return 0.0;
}

struct IntersectionAreaBounds {
	double low = 0.0;
	double high = 0.0;
};

IntersectionAreaBounds restrict_area_to_overlap(double area_low, double area_high,
	double pixel_area, double overlap_area)
{
	IntersectionAreaBounds bounds;
	bounds.low = max(0.0, area_low - (pixel_area - overlap_area));
	bounds.high = min(overlap_area, area_high);
	bounds.low = min(overlap_area, max(0.0, bounds.low));
	bounds.high = min(overlap_area, max(0.0, bounds.high));
	bounds.low = min(bounds.low, bounds.high);
	return bounds;
}

double approximate_intersection_area(Farm *source, Farm *target, const vector<int> &source_pixels)
{
	double total_low = 0.0;
	double total_high = 0.0;
	double source_pixel_area = source->get_pixel_area();
	double target_pixel_area = target->get_pixel_area();

	for(int source_id : source_pixels){
		if(source->show_status(source_id) == OUT) continue;

		box source_box = source->get_pixel_box(source->get_x(source_id), source->get_y(source_id));
		vector<int> target_pixels = target->retrieve_pixels(&source_box);
		double source_low = source->decodePixelArea(source_id, true);
		double source_high = source->decodePixelArea(source_id, false);

		for(int target_id : target_pixels){
			if(target->show_status(target_id) == OUT) continue;

			box target_box = target->get_pixel_box(target->get_x(target_id), target->get_y(target_id));
			double overlap_width = min(source_box.high[0], target_box.high[0])
				- max(source_box.low[0], target_box.low[0]);
			double overlap_height = min(source_box.high[1], target_box.high[1])
				- max(source_box.low[1], target_box.low[1]);
			if(overlap_width <= 0.0 || overlap_height <= 0.0) continue;

			double overlap_area = overlap_width * overlap_height;
			IntersectionAreaBounds source_bounds = restrict_area_to_overlap(
				source_low, source_high, source_pixel_area, overlap_area);
			IntersectionAreaBounds target_bounds = restrict_area_to_overlap(
				target->decodePixelArea(target_id, true),
				target->decodePixelArea(target_id, false),
				target_pixel_area, overlap_area);

			double intersection_low = max(0.0,
				source_bounds.low + target_bounds.low - overlap_area);
			double intersection_high = min(source_bounds.high, target_bounds.high);
			intersection_low = min(intersection_high, max(0.0, intersection_low));
			total_low += intersection_low;
			total_high += intersection_high;
		}
	}

	return (total_low + total_high) * 0.5;
}

void write_intersection_area(query_context *output_ctx, size_t area_idx, double area)
{
	if(output_ctx->areas && area_idx < output_ctx->num_pairs){
		output_ctx->areas[area_idx] = fabs(area);
	}
}

void append_intersection(vector<Intersection> &inters, int source_edge, int target_edge, double t, double u)
{
	inters.push_back({0, static_cast<uint>(source_edge), static_cast<uint>(target_edge),
		clamp_intersection_param(t), clamp_intersection_param(u), OUT});
}

void append_overlap_intersection(vector<Intersection> &inters, int pair_id, int source_edge, int target_edge,
	double t, double u)
{
	inters.push_back({pair_id, static_cast<uint>(source_edge), static_cast<uint>(target_edge),
		clamp_intersection_param(t), clamp_intersection_param(u), OUT});
}

struct EdgeGeometry {
	double ax;
	double ay;
	double dx;
	double dy;
	double len_sq;
	double min_x;
	double max_x;
	double min_y;
	double max_y;
};

vector<EdgeGeometry> build_edge_geometries(Point *vertices, int num_vertices)
{
	vector<EdgeGeometry> edges;
	int num_edges = num_vertices - 1;
	edges.reserve(num_edges);
	for(int i = 0; i < num_edges; i ++){
		const Point &a = vertices[i];
		const Point &b = vertices[i + 1];
		float dx = b.x - a.x;
		float dy = b.y - a.y;
		edges.push_back({
			(double)a.x,
			(double)a.y,
			(double)dx,
			(double)dy,
			(double)dx * dx + (double)dy * dy,
			min((double)a.x, (double)b.x),
			max((double)a.x, (double)b.x),
			min((double)a.y, (double)b.y),
			max((double)a.y, (double)b.y)
		});
	}
	return edges;
}

bool collect_collinear_overlap(const EdgeGeometry &source_edge_geom, int source_edge,
	const EdgeGeometry &target_edge_geom, int target_edge, vector<Intersection> &inters)
{
	const double eps_t = 1e-6;
	bool use_x = fabs(source_edge_geom.dx) >= fabs(source_edge_geom.dy);
	double source_axis = use_x ? source_edge_geom.dx : source_edge_geom.dy;
	double target_axis = use_x ? target_edge_geom.dx : target_edge_geom.dy;
	if(fabs(source_axis) < eps_t || fabs(target_axis) < eps_t){
		return false;
	}

	double source_start_axis = use_x ? source_edge_geom.ax : source_edge_geom.ay;
	double target_start_axis = use_x ? target_edge_geom.ax : target_edge_geom.ay;
	double target_end_axis = target_start_axis + target_axis;
	double t0 = (target_start_axis - source_start_axis) / source_axis;
	double t1 = (target_end_axis - source_start_axis) / source_axis;
	double overlap_low = max(0.0, min(t0, t1));
	double overlap_high = min(1.0, max(t0, t1));
	if(overlap_high - overlap_low <= eps_t){
		return false;
	}

	auto target_param_at_source_param = [&](double t) {
		double point_axis = source_start_axis + source_axis * t;
		return (point_axis - target_start_axis) / target_axis;
	};

	int overlap_pair_id = static_cast<int>(inters.size()) + 1;
	append_overlap_intersection(inters, overlap_pair_id, source_edge, target_edge, overlap_low,
		target_param_at_source_param(overlap_low));
	append_overlap_intersection(inters, overlap_pair_id, source_edge, target_edge, overlap_high,
		target_param_at_source_param(overlap_high));
	return true;
}

bool collect_intersections(Point *source_vertices, Point *target_vertices,
						   const vector<EdgeGeometry> &source_edges,
						   const vector<EdgeGeometry> &target_edges,
						   int source_start, int source_len,
						   int target_start, int target_len,
						   int source_num_vertices, int target_num_vertices,
						   vector<Intersection> &inters)
{
	const double eps_t = 1e-6;
	const double eps_sq = 1e-12;
	const int source_end = source_start + source_len;
	const int target_end = target_start + target_len;
	bool has_collinear_overlap = false;

	for (int i = source_start; i < source_end; i++)
	{
		const EdgeGeometry &source_edge_geom = source_edges[i];
		double source_min_x = source_edge_geom.min_x - eps_t;
		double source_max_x = source_edge_geom.max_x + eps_t;
		double source_min_y = source_edge_geom.min_y - eps_t;
		double source_max_y = source_edge_geom.max_y + eps_t;

		for (int j = target_start; j < target_end; j++)
		{
			const EdgeGeometry &target_edge_geom = target_edges[j];
			if (source_max_x < target_edge_geom.min_x ||
				source_min_x > target_edge_geom.max_x ||
				source_max_y < target_edge_geom.min_y ||
				source_min_y > target_edge_geom.max_y)
			{
				continue;
			}

			double denom = source_edge_geom.dx * target_edge_geom.dy - source_edge_geom.dy * target_edge_geom.dx;
			double delta_x = target_edge_geom.ax - source_edge_geom.ax;
			double delta_y = target_edge_geom.ay - source_edge_geom.ay;
			if (denom * denom <= eps_sq * source_edge_geom.len_sq * target_edge_geom.len_sq)
			{
				double cross = delta_x * source_edge_geom.dy - delta_y * source_edge_geom.dx;
				if(cross * cross <= eps_sq * source_edge_geom.len_sq * max(1.0, delta_x * delta_x + delta_y * delta_y)){
					has_collinear_overlap |= collect_collinear_overlap(source_edge_geom, i, target_edge_geom, j, inters);
				}
				continue;
			}

			double t = (delta_x * target_edge_geom.dy - delta_y * target_edge_geom.dx) / denom;
			if (t < -eps_t || t > 1.0 + eps_t)
				continue;

			double u = (delta_x * source_edge_geom.dy - delta_y * source_edge_geom.dx) / denom;
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

			append_intersection(inters, i, j, t, u);
		}
	}
	return has_collinear_overlap;
}

struct EdgeSequencePairKey {
	uint32_t source_start;
	uint32_t source_len;
	uint32_t target_start;
	uint32_t target_len;
};

size_t collect_edge_sequence_pairs(Farm *source, Farm *target, const vector<int> &source_pixels,
	vector<EdgeSequencePairKey> &sequence_pairs)
{
	sequence_pairs.reserve(source_pixels.size() * 4);
	size_t candidate_pairs = 0;

	for(auto p : source_pixels)
	{
		if(source->show_status(p) != BORDER)
			continue;

		box bx = source->get_pixel_box(source->get_x(p), source->get_y(p));
		vector<int> target_pixels = target->retrieve_pixels(&bx);
		uint32_t source_offset = source->get_offset(p);
		uint32_t source_num_sequences = source->get_num_sequences(p);
		for(auto p2 : target_pixels)
		{
			if(target->show_status(p2) != BORDER)
				continue;

			uint32_t target_offset = target->get_offset(p2);
			uint32_t target_num_sequences = target->get_num_sequences(p2);
			for(uint32_t i = 0; i < source_num_sequences; i ++)
			{
				auto source_range = source->get_edge_sequence(source_offset + i);
				for(uint32_t j = 0; j < target_num_sequences; j ++)
				{
					auto target_range = target->get_edge_sequence(target_offset + j);
					assert(source_range.second != 0 && target_range.second != 0);
					candidate_pairs ++;
					sequence_pairs.push_back({
						source_range.first,
						source_range.second,
						target_range.first,
						target_range.second
					});
				}
			}
		}
	}

	return candidate_pairs;
}

bool collect_intersections_from_sequence_pairs(Farm *source, Farm *target,
	const vector<EdgeGeometry> &source_edges, const vector<EdgeGeometry> &target_edges,
	const vector<EdgeSequencePairKey> &sequence_pairs, vector<Intersection> &inters)
{
	bool has_collinear_overlap = false;
	Point *source_vertices = source->get_boundary()->p;
	Point *target_vertices = target->get_boundary()->p;
	int source_num_vertices = source->get_num_vertices();
	int target_num_vertices = target->get_num_vertices();

	for(const auto &key : sequence_pairs){
		has_collinear_overlap |= collect_intersections(source_vertices, target_vertices,
			source_edges, target_edges,
			key.source_start, key.source_len, key.target_start, key.target_len,
			source_num_vertices, target_num_vertices, inters);
	}
	return has_collinear_overlap;
}

double ring_position(uint32_t edge_id, double param, int num_edges)
{
	double pos = (double)edge_id + param;
	if(fabs(pos - num_edges) < eps){
		return 0.0;
	}
	return pos;
}

struct IntersectionSortKey {
	double primary_pos;
	double secondary_pos;
	uint32_t index;
	uint8_t overlap_priority;
};

double sort_position(uint32_t edge_id, double param, int num_edges)
{
	if(edge_id >= (uint32_t)num_edges){
		return 0.0;
	}
	if(edge_id == (uint32_t)(num_edges - 1) && param > 1.0 - eps){
		return 0.0;
	}
	if(param > 1.0 - eps){
		return (double)edge_id + 1.0;
	}
	if(param < eps){
		return (double)edge_id;
	}
	return (double)edge_id + param;
}

int compare_sort_position(double lhs, double rhs)
{
	if(fabs(lhs - rhs) >= eps){
		return lhs < rhs ? -1 : 1;
	}
	return 0;
}

void sort_unique_source_intersections(vector<Intersection> &inters, int source_num_vertices, int target_num_vertices)
{
	const int source_num_edges = source_num_vertices - 1;
	const int target_num_edges = target_num_vertices - 1;
	assert(inters.size() <= 0xffffffffULL);
	vector<IntersectionSortKey> keys(inters.size());
	for(size_t i = 0; i < inters.size(); i ++){
		keys[i] = {
			sort_position(inters[i].edge_source_id, inters[i].t, source_num_edges),
			sort_position(inters[i].edge_target_id, inters[i].u, target_num_edges),
			static_cast<uint32_t>(i),
			static_cast<uint8_t>(inters[i].pair_id > 0 ? 1 : 0)
		};
	}

	sort(keys.begin(), keys.end(), [](const IntersectionSortKey &a, const IntersectionSortKey &b) {
		int primary_order = compare_sort_position(a.primary_pos, b.primary_pos);
		if(primary_order != 0)
			return primary_order < 0;
		int secondary_order = compare_sort_position(a.secondary_pos, b.secondary_pos);
		if(secondary_order != 0)
			return secondary_order < 0;
		if(a.overlap_priority != b.overlap_priority)
			return a.overlap_priority > b.overlap_priority;
		return a.index < b.index;
	});

	bool first = true;
	size_t unique_count = 0;
	double last_primary_pos = 0.0;
	double last_secondary_pos = 0.0;
	for(const auto &key : keys){
		if(first || compare_sort_position(key.primary_pos, last_primary_pos) != 0 ||
			compare_sort_position(key.secondary_pos, last_secondary_pos) != 0){
			unique_count ++;
			last_primary_pos = key.primary_pos;
			last_secondary_pos = key.secondary_pos;
			first = false;
		}
	}

	vector<Intersection> sorted;
	sorted.reserve(unique_count);
	first = true;
	last_primary_pos = 0.0;
	last_secondary_pos = 0.0;
	for(const auto &key : keys){
		if(first || compare_sort_position(key.primary_pos, last_primary_pos) != 0 ||
			compare_sort_position(key.secondary_pos, last_secondary_pos) != 0){
			sorted.push_back(inters[key.index]);
			last_primary_pos = key.primary_pos;
			last_secondary_pos = key.secondary_pos;
			first = false;
		}
	}
	inters.swap(sorted);
}

void sort_target_intersections(vector<Intersection> &inters, int source_num_vertices, int target_num_vertices)
{
	const int source_num_edges = source_num_vertices - 1;
	const int target_num_edges = target_num_vertices - 1;
	assert(inters.size() <= 0xffffffffULL);
	vector<IntersectionSortKey> keys(inters.size());
	for(size_t i = 0; i < inters.size(); i ++){
		keys[i] = {
			sort_position(inters[i].edge_target_id, inters[i].u, target_num_edges),
			sort_position(inters[i].edge_source_id, inters[i].t, source_num_edges),
			static_cast<uint32_t>(i),
			static_cast<uint8_t>(inters[i].pair_id > 0 ? 1 : 0)
		};
	}

	sort(keys.begin(), keys.end(), [](const IntersectionSortKey &a, const IntersectionSortKey &b) {
		int primary_order = compare_sort_position(a.primary_pos, b.primary_pos);
		if(primary_order != 0)
			return primary_order < 0;
		int secondary_order = compare_sort_position(a.secondary_pos, b.secondary_pos);
		if(secondary_order != 0)
			return secondary_order < 0;
		if(a.overlap_priority != b.overlap_priority)
			return a.overlap_priority > b.overlap_priority;
		return a.index < b.index;
	});

	vector<Intersection> sorted;
	sorted.reserve(keys.size());
	for(const auto &key : keys){
		sorted.push_back(inters[key.index]);
	}
	inters.swap(sorted);
}

double point_length_sq(const Point &p)
{
	return (double)p.x * p.x + (double)p.y * p.y;
}

PartitionStatus classify_shared_boundary_arc(const Point &sample, const Point &primary_dir,
	Farm *secondary, bool is_primary)
{
	if(point_length_sq(primary_dir) <= 1e-18){
		return OUT;
	}

	int xoff = secondary->get_offset_x(sample.x);
	int yoff = secondary->get_offset_y(sample.y);
	int start_x = max(0, xoff - 1);
	int end_x = min(secondary->get_dimx() - 1, xoff + 1);
	int start_y = max(0, yoff - 1);
	int end_y = min(secondary->get_dimy() - 1, yoff + 1);
	for(int yy = start_y; yy <= end_y; yy ++)
	{
		for(int xx = start_x; xx <= end_x; xx ++)
		{
			int pix = secondary->get_id(xx, yy);
			if(secondary->show_status(pix) != BORDER){
				continue;
			}
			for (uint32_t e = 0; e < secondary->get_num_sequences(pix); e++)
			{
				auto edges = secondary->get_edge_sequence(secondary->get_offset(pix) + e);
				auto pos = edges.first;
				for (int k = 0; k < edges.second; k++)
				{
					Point v1 = secondary->get_boundary()->p[pos + k];
					Point v2 = secondary->get_boundary()->p[pos + k + 1];
					if(!point_on_segment(sample, v1, v2)){
						continue;
					}

					Point secondary_dir = v2 - v1;
					if(collinear_direction(primary_dir, secondary_dir, 1e-12) == 0){
						continue;
					}
					if(!is_primary){
						return OUT;
					}
					return same_direction(primary_dir, secondary_dir) ? IN : OUT;
				}
			}
		}
	}
	return BORDER;
}

int collinear_direction(double ax, double ay, double bx, double by, double eps_sq)
{
	double cross = ax * by - ay * bx;
	double len_a = ax * ax + ay * ay;
	double len_b = bx * bx + by * by;

	if(cross * cross > eps_sq * len_a * len_b)
		return 0;

	double dot = ax * bx + ay * by;
	return dot > 0.0 ? 1 : -1;
}

PartitionStatus classify_shared_boundary_arc_direct(const Intersection &inter1, const Intersection &inter2,
	double sample_x, double sample_y, double primary_dx, double primary_dy,
	const vector<EdgeGeometry> &source_edges, const vector<EdgeGeometry> &target_edges,
	bool is_primary)
{
	double primary_len_sq = primary_dx * primary_dx + primary_dy * primary_dy;
	if(primary_len_sq <= 1e-18){
		return OUT;
	}

	const vector<EdgeGeometry> &secondary_edges = is_primary ? target_edges : source_edges;
	int secondary_num_edges = secondary_edges.size();
	int candidate_edges[6];
	if(is_primary){
		candidate_edges[0] = inter1.edge_target_id;
		candidate_edges[1] = inter2.edge_target_id;
	}else{
		candidate_edges[0] = inter1.edge_source_id;
		candidate_edges[1] = inter2.edge_source_id;
	}
	candidate_edges[2] = candidate_edges[0] == 0 ? secondary_num_edges - 1 : candidate_edges[0] - 1;
	candidate_edges[3] = candidate_edges[0] + 1 >= secondary_num_edges ? 0 : candidate_edges[0] + 1;
	candidate_edges[4] = candidate_edges[1] == 0 ? secondary_num_edges - 1 : candidate_edges[1] - 1;
	candidate_edges[5] = candidate_edges[1] + 1 >= secondary_num_edges ? 0 : candidate_edges[1] + 1;

	for(int i = 0; i < 6; i ++){
		int edge_id = candidate_edges[i];
		bool duplicate = false;
		for(int j = 0; j < i; j ++){
			if(candidate_edges[j] == edge_id){
				duplicate = true;
				break;
			}
		}
		if(duplicate) continue;

		const EdgeGeometry &secondary = secondary_edges[edge_id];
		if(collinear_direction(primary_dx, primary_dy, secondary.dx, secondary.dy, 1e-12) == 0){
			continue;
		}
		if(sample_x < secondary.min_x - 1e-9 || sample_x > secondary.max_x + 1e-9 ||
			sample_y < secondary.min_y - 1e-9 || sample_y > secondary.max_y + 1e-9){
			continue;
		}
		if(!is_primary){
			return OUT;
		}
		return primary_dx * secondary.dx + primary_dy * secondary.dy > 0.0 ? IN : OUT;
	}
	return BORDER;
}

PartitionStatus classify_overlap_pair_arc(const Intersection &inter1, const Intersection &inter2,
	Point *source_vertices, Point *target_vertices, bool is_primary)
{
	if(inter1.pair_id <= 0 || inter1.pair_id != inter2.pair_id){
		return BORDER;
	}
	if(!is_primary){
		return OUT;
	}
	if(inter1.edge_source_id != inter2.edge_source_id || inter2.t <= inter1.t + eps){
		return BORDER;
	}

	Point source_p1 = interpolate_edge(source_vertices, inter1.edge_source_id, inter1.t);
	Point source_p2 = interpolate_edge(source_vertices, inter2.edge_source_id, inter2.t);
	Point target_dir = target_vertices[inter1.edge_target_id + 1] - target_vertices[inter1.edge_target_id];
	return same_direction(source_p2 - source_p1, target_dir) ? IN : OUT;
}

struct ClassifyArcStats {
	size_t raster_direct = 0;
	size_t shared_boundary = 0;
	size_t shared_boundary_direct = 0;
	size_t shared_boundary_scan = 0;
	size_t border_refine = 0;
	size_t overlap_direct = 0;
};

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
											  Point *source_vertices, Point *target_vertices,
										  const vector<EdgeGeometry> &source_edges,
										  const vector<EdgeGeometry> &target_edges,
										  Farm *secondary, bool is_primary, bool has_collinear_overlap,
										  ClassifyArcStats *stats = nullptr)
{
	if(has_collinear_overlap){
		PartitionStatus overlap_status = classify_overlap_pair_arc(inter1, inter2, source_vertices, target_vertices, is_primary);
		if(overlap_status != BORDER){
			if(stats) stats->overlap_direct ++;
			return overlap_status;
		}
	}

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
	Point primary_dir;
	if (edge1 == edge2 && param1 < param2)
	{
		sample = (p1 + p2) * 0.5f;
		primary_dir = p2 - p1;
	}
	else
	{
		sample = (p1 + primary_vertices[edge1 + 1]) * 0.5f;
		primary_dir = primary_vertices[edge1 + 1] - p1;
	}

	if(has_collinear_overlap){
		PartitionStatus shared_status = classify_shared_boundary_arc_direct(inter1, inter2,
			sample.x, sample.y, primary_dir.x, primary_dir.y, source_edges, target_edges, is_primary);
		if(shared_status != BORDER){
			if(stats){
				stats->shared_boundary ++;
				stats->shared_boundary_direct ++;
			}
			return shared_status;
		}
	}

	double rem_x = remainder(sample.x - secondary->getMBB()->low[0], secondary->get_step_x());
	double rem_y = remainder(sample.y - secondary->getMBB()->low[1], secondary->get_step_y());
	int xoff = secondary->get_offset_x(sample.x);
	int yoff = secondary->get_offset_y(sample.y);
	int pix = secondary->get_id(xoff, yoff);
	PartitionStatus st = (fabs(rem_x) < 1e-9 || fabs(rem_y) < 1e-9) ? BORDER : secondary->show_status(pix);
	if (st != BORDER){
		if(stats) stats->raster_direct ++;
		return st;
	}
	if(has_collinear_overlap){
		PartitionStatus shared_status = classify_shared_boundary_arc(sample, primary_dir, secondary, is_primary);
		if(shared_status != BORDER){
			if(stats){
				stats->shared_boundary ++;
				stats->shared_boundary_scan ++;
			}
			return shared_status;
		}
	}

	if(stats) stats->border_refine ++;
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

bool is_proper_crossing_intersection(const Intersection &inter)
{
	return inter.pair_id == 0
		&& inter.t > eps && inter.t < 1.0 - eps
		&& inter.u > eps && inter.u < 1.0 - eps;
}

PartitionStatus classify_proper_crossing_arc(const Intersection &inter,
	Point *source_vertices, Point *target_vertices, bool is_primary)
{
	Point source_dir = source_vertices[inter.edge_source_id + 1] - source_vertices[inter.edge_source_id];
	Point target_dir = target_vertices[inter.edge_target_id + 1] - target_vertices[inter.edge_target_id];
	double cross = (double)target_dir.x * source_dir.y - (double)target_dir.y * source_dir.x;
	if(is_primary){
		return cross > 0.0 ? IN : OUT;
	}
	return cross < 0.0 ? IN : OUT;
}

bool is_overlap_arc(const Intersection &inter1, const Intersection &inter2)
{
	return inter1.pair_id > 0 && inter1.pair_id == inter2.pair_id;
}

void classify_intersection_arcs(vector<Intersection> &inters,
	Point *primary_vertices, int primary_num_vertices,
	Point *source_vertices, Point *target_vertices,
	const vector<EdgeGeometry> &source_edges,
	const vector<EdgeGeometry> &target_edges,
	Farm *secondary, bool is_primary, bool has_collinear_overlap, query_context *ctx)
{
	const int num_intersections = inters.size();
	if(num_intersections == 0){
		return;
	}

	size_t proper_count = 0;
	size_t overlap_count = 0;
	size_t fallback_count = 0;
	ClassifyArcStats stats;
	for(int i = 0; i < num_intersections; i ++){
		Intersection &inter = inters[i];
		Intersection &next = inters[(i + 1) % num_intersections];

		if(is_proper_crossing_intersection(inter)){
			proper_count ++;
			inter.status = classify_proper_crossing_arc(inter, source_vertices, target_vertices, is_primary);
		}else if(has_collinear_overlap && is_overlap_arc(inter, next)){
			overlap_count ++;
			inter.status = classify_intersection_arc(inter, next, primary_vertices, primary_num_vertices,
				source_vertices, target_vertices,
				source_edges, target_edges,
				secondary, is_primary, has_collinear_overlap, &stats);
		}else{
			fallback_count ++;
			inter.status = classify_intersection_arc(inter, next, primary_vertices, primary_num_vertices,
				source_vertices, target_vertices,
				source_edges, target_edges,
				secondary, is_primary, has_collinear_overlap, &stats);
		}
	}

	if(is_primary){
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceProper, proper_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceOverlap, overlap_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceFallback, fallback_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceRasterDirect, stats.raster_direct);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceSharedBoundary, stats.shared_boundary);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceSharedBoundaryDirect, stats.shared_boundary_direct);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceSharedBoundaryScan, stats.shared_boundary_scan);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifySourceBorderRefine, stats.border_refine);
	}else{
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetProper, proper_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetOverlap, overlap_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetFallback, fallback_count);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetRasterDirect, stats.raster_direct);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetSharedBoundary, stats.shared_boundary);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetSharedBoundaryDirect, stats.shared_boundary_direct);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetSharedBoundaryScan, stats.shared_boundary_scan);
		ctx->profiler.add_count(ProfileCount::IntersectionClassifyTargetBorderRefine, stats.border_refine);
	}
}

double accumulate_arc_area(Point *vertices, int num_vertices, int edge1, double param1,
	int edge2, double param2, bool wrapped)
{
	const int num_edges = num_vertices - 1;
	double start_pos = ring_position(edge1, param1, num_edges);
	double end_pos = ring_position(edge2, param2, num_edges);
	if(wrapped || end_pos <= start_pos + eps){
		end_pos += num_edges;
	}
	if(end_pos <= start_pos + eps){
		return 0.0;
	}

	Point p1 = interpolate_edge(vertices, edge1, param1);
	Point p2 = interpolate_edge(vertices, edge2, param2);
	double a = 0.0;
	double b = 0.0;
	double last_x = p1.x;
	double last_y = p1.y;
	int first_vertex = (int)floor(start_pos + eps) + 1;
	int last_vertex = (int)floor(end_pos + eps);
	for(int pos = first_vertex; pos <= last_vertex; pos ++){
		if(pos > end_pos + eps){
			break;
		}
		int vertex_id = pos % num_edges;
		a += last_x * vertices[vertex_id].y;
		b += last_y * vertices[vertex_id].x;
		last_x = vertices[vertex_id].x;
		last_y = vertices[vertex_id].y;
	}

	a += last_x * p2.y;
	b += last_y * p2.x;
	return a - b;
}

double accumulate_area_for_order(vector<Intersection> &inters,
								 Point *primary_vertices, int primary_num_vertices,
								 bool is_primary)
{
	double area = 0.0;
	const int num_intersections = inters.size();

	for (int i = 0; i < num_intersections; i++)
	{
		Intersection inter1 = inters[i];
		Intersection inter2 = (i + 1 >= num_intersections) ? inters[0] : inters[i + 1];
		bool wrapped = i + 1 >= num_intersections;

		int edge1 = is_primary ? inter1.edge_source_id : inter1.edge_target_id;
		int edge2 = is_primary ? inter2.edge_source_id : inter2.edge_target_id;
		double param1 = is_primary ? inter1.t : inter1.u;
		double param2 = is_primary ? inter2.t : inter2.u;

		if (inters[i].status == IN)
		{
			area += accumulate_arc_area(primary_vertices, primary_num_vertices, edge1, param1,
				edge2, param2, wrapped);
		}
	}

	return area;
}

} // namespace

Farm::~Farm()
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

void Farm::add_edge(int idx, int start, int end)
{
	assert(end - start + 1 > 0);
	edge_sequences[idx] = make_pair(start, end - start + 1);
}

uint32_t Farm::get_num_sequences(int id)
{
	if (show_status(id) != BORDER)
		return 0;
	return offset[id + 1] - offset[id];
}

void Farm::init_edge_sequences(int num_edge_seqs)
{
	assert(num_edge_seqs >= 0);
	len_edge_sequences = num_edge_seqs;
	edge_sequences = new pair<uint32_t, uint32_t>[num_edge_seqs];
	assert(len_edge_sequences < 65536); // 2^16, to fit in uint16_t for edge id and count
}

void Farm::process_pixels_null(int x, int y)
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

double Farm::decodePixelArea(int id, bool isLow){
    uint8_t fullness = get_fullness(id);
	double pixelArea = get_pixel_area();
	const int category_count = get_category_count();
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

uint8_t Farm::encodePixelArea(double area){
	return encodePixelArea(area, get_pixel_area());
}

uint8_t Farm::encodePixelArea(double area, double pixel_area_val){
	const int category_count = get_category_count();
	double ratio = area / pixel_area_val;

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
		idx = category_count - 1;

	assert(idx < 256);
	return idx;
}

void Farm::process_crosses_sparse(const vector<int> &pixel_ids, const vector<vector<cross_info>> &edges_info)
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

void Farm::process_intersection(vector<vector<double>>& intersection_info, Direction direction)
{
	    auto* target_data = (direction == HORIZONTAL) ? horizontal : vertical;

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

    int idx = 0;

    int size = intersection_info.size();
    for (int i = 0; i < size; ++i)
    {
        auto& nodes = intersection_info[i];

        target_data->set_offset(i, idx);

        if (nodes.empty()) continue;

        std::sort(nodes.begin(), nodes.end());

        for (double node : nodes)
        {
            target_data->add_node(idx, node);
            idx++;
        }
    }
}

void Farm::init_pixels()
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

void Farm::evaluate_edges()
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

void Farm::calculate_fullness()
{
	int total_pixels = dimx * dimy;
    vector<double> delta_buf(total_pixels, 0.0);
    vector<double> area_buf(total_pixels, 0.0);

    double origin_x = mbr->low[0];
    double origin_y = mbr->low[1];
    double inv_step_x = 1.0 / step_x;
    double inv_step_y = 1.0 / step_y;
    double pixel_area_val = step_x * step_y;

    int num_verts = boundary->num_vertices;
    for (int i = 0; i < num_verts - 1; ++i) {
        double p1x = (boundary->p[i].x - origin_x) * inv_step_x;
        double p1y = (boundary->p[i].y - origin_y) * inv_step_y;
        double p2x = (boundary->p[i+1].x - origin_x) * inv_step_x;
        double p2y = (boundary->p[i+1].y - origin_y) * inv_step_y;
        
	        if (abs(p1y - p2y) < 1e-9) continue;

        double dx = p2x - p1x;
        double dy = p2y - p1y;
        
        int step_x_dir = (dx > 0) ? 1 : -1;
        int step_y_dir = (dy > 0) ? 1 : -1;
        
	        int x = floor(p1x);
	        int y = floor(p1y);

		const double INF_T = 1e30;
		double t_delta_x = (abs(dx) < 1e-12) ? INF_T : abs(1.0 / dx);
		double t_delta_y = (abs(dy) < 1e-12) ? INF_T : abs(1.0 / dy);
        
        double t_max_x, t_max_y;
        
        if (dx > 1e-12) t_max_x = (floor(p1x) + 1.0 - p1x) / dx;
        else if (dx < -1e-12) t_max_x = (floor(p1x) - p1x) / dx;
        else t_max_x = INF_T;

        if (dy > 1e-12) t_max_y = (floor(p1y) + 1.0 - p1y) / dy;
        else if (dy < -1e-12) t_max_y = (floor(p1y) - p1y) / dy;
        else t_max_y = INF_T;
        
        double t_prev = 0.0;
        
        while (t_prev < 1.0 - 1e-9) {
            int safe_x = max(0, min(dimx - 1, x));
            int safe_y = max(0, min(dimy - 1, y));
            int idx = get_id(safe_x, safe_y);

            double t_next;
            int next_step = 0; // 1: x, 2: y

            if (t_max_x < t_max_y) {
                t_next = t_max_x;
                next_step = 1;
            } else {
                t_next = t_max_y;
                next_step = 2;
            }
            
            if (t_next > 1.0) {
                t_next = 1.0;
            }

            double enter_y = p1y + t_prev * dy;
            double exit_y  = p1y + t_next * dy;
            double enter_x = p1x + t_prev * dx;
            double exit_x  = p1x + t_next * dx;
            
            double avg_x = (enter_x + exit_x) * 0.5;
            double local_x = avg_x - safe_x;
            double delta_y = exit_y - enter_y;
            
            delta_buf[idx] += delta_y;
			area_buf[idx] += (delta_y * (1.0 - local_x));

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
    
    for (int y = 0; y < dimy; ++y)
    {
        double current_accumulated_height = 0.0;
        for (int x = 0; x < dimx; ++x)
        {
            int idx = get_id(x, y);
            
            double val = area_buf[idx] + current_accumulated_height;
            
            current_accumulated_height += delta_buf[idx];
            
            double final_area = min(pixel_area_val, max(0.0, abs(val) * pixel_area_val));
            
			bool was_border = get_fullness(idx) == BORDER;

            areas[idx] = final_area;
            
            uint8_t fullness = encodePixelArea(final_area, pixel_area_val);
			if (was_border)
			{
				set_status(idx, max<uint8_t>(1, min<uint8_t>(
					fullness, static_cast<uint8_t>(get_category_count() - 2))));
			}
			else if (fullness == 0 || fullness == status_max_value(bitwidth))
			{
				set_status(idx, fullness);
				areas[idx] = (fullness == 0) ? 0.0 : pixel_area_val;
			}
			else
			{
				assert(false); // Unexpected case
				// A partial non-edge pixel would have no edge sequence to refine;
				// snap it to a terminal state to keep status/offset consistent.
				if (final_area >= pixel_area_val * 0.5)
				{
					set_status(idx, status_max_value(bitwidth));
					areas[idx] = pixel_area_val;
				}
				else
				{
					set_status(idx, 0);
					areas[idx] = 0.0;
				}
			}
        }
    }
}


void Farm::rasterization()
{

	// 1. create space for the pixels
	init_pixels();

	// 2. edge crossing to identify BORDER pixels
	evaluate_edges();

	// 3. determine the fullness of pixels
	calculate_fullness();
}

void Farm::rasterization(int vpr)
{
	assert(vpr > 0);
	pthread_mutex_lock(&farm_partition_lock);

	assert(dimx > 0 && dimy > 0 && step_x > 0.0 && step_y > 0.0);
	assert(status_size > 0);
	if (use_hierarchy)
	{
		assert(num_layers > 0 && layers);
	}

	rasterization();

	if (use_hierarchy)
	{
		layers[num_layers - 1].attach_base_storage(status, areas);
			for (int i = num_layers - 2; i >= 0; i--)
			{
				merge_status(layers[i], layers[i + 1]);
				const size_t layer_status_bytes = packed_status_bytes(
					layers[i].get_num_pixels(), bitwidth);
				memcpy(status + layer_offset[i], layers[i].get_status(), layer_status_bytes);
			}
	}

	pthread_mutex_unlock(&farm_partition_lock);
}

int Farm::num_edges_covered(int id)
{
	int c = 0;
	for (int i = 0; i < get_num_sequences(id); i++)
	{
		auto r = edge_sequences[offset[id] + i];
		c += r.second;
	}
	return c;
}

int Farm::count_intersection_nodes(Point &p)
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

// double Farm::merge_area(box target, PartitionStatus &st)
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

void Farm::merge_status(Hraster &parent, const Hraster &child)
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
			parent.set_status(parent_id, classifyPixel(merged_area, parent_pixel_area, bitwidth));
		}
	}
}

void Farm::layering(int NLow)
{
	if (NLow < 1)
		NLow = 1;

	struct LayerParam {
		int dx, dy;
		double sx, sy;
		box mbr;
	};

	vector<LayerParam> params;

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
		auto &p = params[i];
		bool is_base = (i == 0);
		box *layer_mbr = is_base ? getMBB() : &p.mbr;

		layers[target_idx].set_bitwidth(bitwidth);
		layers[target_idx].init(p.sx, p.sy, p.dx, p.dy, layer_mbr, is_base);
		layer_info[target_idx] = {*layers[target_idx].mbr, p.dx, p.dy, p.sx, p.sy};
		layer_offset[target_idx] = static_cast<uint32_t>(current_offset_accum);
		current_offset_accum += packed_status_bytes(
			static_cast<size_t>(p.dx) * p.dy, bitwidth);
	}

	status_size = current_offset_accum;
}

bool Farm::contain(Point &p, query_context *ctx)
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
	int nc = count_intersection_nodes(p);
	if (nc % 2 == 1)
	{
		ret = !ret;
	}
	return ret;
}

bool Farm::border_pixel_intersects_box(int pixel_id, box &overlap, query_context *ctx)
{
	Point overlap_border[5];
	overlap_border[0] = Point(overlap.low[0], overlap.low[1]);
	overlap_border[1] = Point(overlap.low[0], overlap.high[1]);
	overlap_border[2] = Point(overlap.high[0], overlap.high[1]);
	overlap_border[3] = Point(overlap.high[0], overlap.low[1]);
	overlap_border[4] = Point(overlap.low[0], overlap.low[1]);

	for (uint32_t e = 0; e < get_num_sequences(pixel_id); e++)
	{
		auto edges = get_edge_sequence(get_offset(pixel_id) + e);
		auto pos = edges.first;
		auto size = edges.second;
		if (segment_intersect_batch(boundary->p + pos, overlap_border, size, 4))
		{
			return true;
		}
	}

	Point center((overlap.low[0] + overlap.high[0]) * 0.5, (overlap.low[1] + overlap.high[1]) * 0.5);
	return contain(center, ctx);
}

bool Farm::intersect(Farm *target, query_context *ctx, bool approximation)
{
	QueryProfileCall profile(ctx->profiler, ProfileStage::IntersectTotal, ProfileStage::IntersectRaster);
	vector<tuple<double, int, int>> candidate_pairs;

	auto candidate_build_timer = ctx->profiler.scoped(ProfileStage::IntersectCandidateBuild);
	vector<int> pxs = retrieve_pixels(target->getMBB());
	for (auto pa : pxs)
	{
		auto source_status = show_status(pa);
		if(source_status == OUT) continue;
		box bx = get_pixel_box(get_x(pa), get_y(pa));
		vector<int> tpxs = target->retrieve_pixels(&bx);
		for (auto pb : tpxs)
		{
			// evaluate the state
			auto target_status = target->show_status(pb);
			if (target_status == OUT) continue;
			if(source_status == IN) {
				candidate_build_timer.stop();
				return profile.finish(true);
			}
			if(source_status == BORDER && target_status == IN){
				box target_box = target->get_pixel_box(target->get_x(pb), target->get_y(pb));
				if(border_pixel_intersects_box(pa, target_box, ctx)){
					candidate_build_timer.stop();
					return profile.finish(true);
				}
				continue;
			}
			assert(source_status == BORDER && target_status == BORDER);
			auto s_p_low = decodePixelArea(pa, true);
			auto s_p_high = decodePixelArea(pa, false);
			auto t_p_low = target->decodePixelArea(pb, true);
			auto t_p_high = target->decodePixelArea(pb, false);
			auto pixel_area = max(get_pixel_area(), target->get_pixel_area());
			if(s_p_low + t_p_low >= pixel_area){
				candidate_build_timer.stop();
				return profile.finish(true);
			}
			auto s_p_apx = (s_p_low + s_p_high) / 2;
			auto t_p_apx = (t_p_low + t_p_high) / 2;
			auto prob = (s_p_apx + t_p_apx) / pixel_area;
			prob = min(1.0, max(0.0, prob));
			candidate_pairs.push_back({prob, pa, pb});
		}
	}

	candidate_build_timer.stop();

	auto sort_timer = ctx->profiler.scoped(ProfileStage::IntersectCandidateSort);
	sort(candidate_pairs.begin(), candidate_pairs.end(), [](const auto& a, const auto& b) {
		return a > b;
	});
	sort_timer.stop();

	if(approximation){
		double no_intersect_prob = 1.0;
		for(int pair_id = 0; pair_id < candidate_pairs.size(); pair_id ++){
			auto prob = get<0>(candidate_pairs[pair_id]);
			no_intersect_prob *= (1.0 - prob);
			if(1.0 - no_intersect_prob >= ctx->approx_confidence){
				return profile.finish(true);
			}
		}
		return profile.finish(false);
	}

	profile.finish_phase(ProfileStage::IntersectRefine);

	auto exact_timer = ctx->profiler.scoped(ProfileStage::IntersectExactEdges);
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
					exact_timer.stop();
					return profile.finish(true);
				}
			}
		}
	}
	exact_timer.stop();

	auto contain_timer = ctx->profiler.scoped(ProfileStage::IntersectContainCheck);
	if(contain(target->get_boundary()->p[0], ctx)){
		contain_timer.stop();
		return profile.finish(true);
	}

	if(target->contain(boundary->p[0], ctx)){
		contain_timer.stop();
		return profile.finish(true);
	}
	contain_timer.stop();

	return profile.finish(false);
}

void Farm::intersection(Farm *target, query_context *ctx, bool approximation)
{
	QueryProfileCall profile(ctx->profiler, ProfileStage::IntersectionTotal, ProfileStage::IntersectionRaster);
	query_context *output_ctx = ctx->global_ctx;
	size_t area_idx = ctx->target_id;
	if (output_ctx->areas && area_idx < output_ctx->num_pairs)
		output_ctx->areas[area_idx] = 0.0;

	vector<int> pxs = retrieve_pixels(target->getMBB());
	if(approximation){
		double area = approximate_intersection_area(this, target, pxs);
		write_intersection_area(output_ctx, area_idx, 2.0 * area);
		profile.finish();
		return;
	}

	vector<Intersection> inters;
	vector<EdgeGeometry> source_edges_for_classify;
	vector<EdgeGeometry> target_edges_for_classify;
	bool has_collinear_overlap = false;

	{
		auto collect_timer = ctx->profiler.scoped(ProfileStage::IntersectionCollectEdges);
		source_edges_for_classify = build_edge_geometries(boundary->p, get_num_vertices());
		target_edges_for_classify = build_edge_geometries(target->boundary->p, target->get_num_vertices());
		vector<EdgeSequencePairKey> sequence_pairs;
		size_t edge_sequence_pairs = collect_edge_sequence_pairs(this, target, pxs, sequence_pairs);
		has_collinear_overlap = collect_intersections_from_sequence_pairs(this, target,
			source_edges_for_classify, target_edges_for_classify, sequence_pairs, inters);
		ctx->profiler.add_count(ProfileCount::IntersectionEdgeSequencePairs, edge_sequence_pairs);
	}

	if (inters.empty())
	{
		write_intersection_area(output_ctx, area_idx, no_crossing_intersection_area(this, target));
		profile.finish();
		return;
	}

	profile.finish_phase(ProfileStage::IntersectionRefine);
	ctx->profiler.add_count(ProfileCount::IntersectionRawIntersections, inters.size());

	{
		auto sort_timer = ctx->profiler.scoped(ProfileStage::IntersectionSortSource);
		sort_unique_source_intersections(inters, get_num_vertices(), target->get_num_vertices());
	}
	int num_inters = inters.size();
	ctx->profiler.add_count(ProfileCount::IntersectionUniqueIntersections, num_inters);
	if (num_inters == 0){
		write_intersection_area(output_ctx, area_idx, no_crossing_intersection_area(this, target));
		profile.finish();
		return;
	}

	{
		auto classify_timer = ctx->profiler.scoped(ProfileStage::IntersectionClassifySource);
		classify_intersection_arcs(inters, boundary->p, get_num_vertices(),
			boundary->p, target->boundary->p,
			source_edges_for_classify, target_edges_for_classify,
			target, true, has_collinear_overlap, ctx);
	}

	double area = 0.0;
	{
		auto area_timer = ctx->profiler.scoped(ProfileStage::IntersectionAreaSource);
		area = accumulate_area_for_order(inters, boundary->p, get_num_vertices(),
										 true);
	}

	{
		auto sort_timer = ctx->profiler.scoped(ProfileStage::IntersectionSortTarget);
		sort_target_intersections(inters, get_num_vertices(), target->get_num_vertices());
	}
	{
		auto classify_timer = ctx->profiler.scoped(ProfileStage::IntersectionClassifyTarget);
		classify_intersection_arcs(inters, target->boundary->p, target->get_num_vertices(),
			boundary->p, target->boundary->p,
			source_edges_for_classify, target_edges_for_classify,
			this, false, has_collinear_overlap, ctx);
	}

	{
		auto area_timer = ctx->profiler.scoped(ProfileStage::IntersectionAreaTarget);
		area += accumulate_area_for_order(inters, target->boundary->p, target->get_num_vertices(),
										  false);
	}

	write_intersection_area(output_ctx, area_idx, area);

	profile.finish();
	return;
}

bool Farm::within(Farm *target, query_context *ctx, bool approximation)
{
	QueryProfileCall profile(ctx->profiler, ProfileStage::WithinTotal, ProfileStage::WithinRaster);

	vector<pair<int, int>> current_pairs;
	vector<pair<int, int>> next_pairs;
	seed_within_initial_pairs(this, target, ctx, current_pairs);

	int source_level = 0;
	int target_level = 0;
	while(true){
		auto layer_setup_timer = ctx->profiler.scoped(ProfileStage::WithinRasterLayerSetup);
		size_t pair_count = current_pairs.size();
		if(pair_count == 0){
			layer_setup_timer.stop();
			break;
		}
		WithinLayerPlan plan = build_within_layer_plan(this, target, source_level, target_level, pair_count);
		layer_setup_timer.stop();

		{
			auto buffer_timer = ctx->profiler.scoped(ProfileStage::WithinRasterBuffer);
			next_pairs.clear();
			next_pairs.reserve(plan.expected_pairs);
		}

		auto expand_distance_timer = ctx->profiler.scoped(ProfileStage::WithinRasterExpandDistance);
		WithinExpandContext expand_ctx(plan, ctx, next_pairs);
		bool found = expand_within_pairs(plan, expand_ctx, current_pairs);
		expand_distance_timer.stop();
		if(found){
			return profile.finish(true);
		}
		current_pairs.swap(next_pairs);
		if(!plan.source_next_layer && !plan.target_next_layer){
			break;
		}
	}

	if(!approximation){
		profile.finish_phase(ProfileStage::WithinRefine);
	}

	vector<WithinCandidate> candidate_pairs;
	build_within_refine_candidates(this, target, ctx, current_pairs, candidate_pairs);
	if(approximation){
		return profile.finish(approximate_within_candidates(candidate_pairs, ctx));
	}

	{
		auto timer = ctx->profiler.scoped(ProfileStage::WithinRefineCandidateSort);
		sort(candidate_pairs.begin(), candidate_pairs.end());
	}

	vector<double> suffix_min_dist = build_within_suffix_min_dist(ctx, candidate_pairs);
	return profile.finish(refine_within_candidates(this, target, ctx, candidate_pairs, suffix_min_dist));
}
