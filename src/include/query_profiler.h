#ifndef SRC_INCLUDE_QUERY_PROFILER_H_
#define SRC_INCLUDE_QUERY_PROFILER_H_

#include <array>
#include <chrono>
#include <cstddef>

#include "util.h"

#ifndef FARM_PROFILE_QUERY
#define FARM_PROFILE_QUERY 0
#endif

enum class ProfileStage {
	WithinTotal,
	WithinRaster,
	WithinRefine,
	WithinRasterInit,
	WithinRasterLayerSetup,
	WithinRasterBuffer,
	WithinRasterExpandDistance,
	WithinRefineCandidateBuild,
	WithinRefineCandidateSort,
	WithinRefineSuffix,
	WithinRefinePrepare,
	WithinRefineExact,
	IntersectTotal,
	IntersectRaster,
	IntersectRefine,
	IntersectCandidateBuild,
	IntersectCandidateSort,
	IntersectExactEdges,
	IntersectContainCheck,
	IntersectionTotal,
	IntersectionRaster,
	IntersectionRefine,
	IntersectionCollectEdges,
	IntersectionSortSource,
	IntersectionClassifySource,
	IntersectionAreaSource,
	IntersectionSortTarget,
	IntersectionClassifyTarget,
	IntersectionAreaTarget,
	Count
};

enum class ProfileCount {
	WithinRefinePixels,
	WithinRefineEdges,
	IntersectionEdgeSequencePairs,
	IntersectionRawIntersections,
	IntersectionUniqueIntersections,
	IntersectionClassifySourceProper,
	IntersectionClassifySourceOverlap,
	IntersectionClassifySourceFallback,
	IntersectionClassifySourceRasterDirect,
	IntersectionClassifySourceSharedBoundary,
	IntersectionClassifySourceSharedBoundaryDirect,
	IntersectionClassifySourceSharedBoundaryScan,
	IntersectionClassifySourceBorderRefine,
	IntersectionClassifyTargetProper,
	IntersectionClassifyTargetOverlap,
	IntersectionClassifyTargetFallback,
	IntersectionClassifyTargetRasterDirect,
	IntersectionClassifyTargetSharedBoundary,
	IntersectionClassifyTargetSharedBoundaryDirect,
	IntersectionClassifyTargetSharedBoundaryScan,
	IntersectionClassifyTargetBorderRefine,
	Count
};

class QueryProfiler {
public:
	class ScopedTimer {
	public:
#if FARM_PROFILE_QUERY
		ScopedTimer(QueryProfiler *profiler, ProfileStage stage)
			: profiler(profiler), stage(stage), start(Clock::now()), active(true)
		{
		}

		ScopedTimer(const ScopedTimer&) = delete;
		ScopedTimer& operator=(const ScopedTimer&) = delete;

		ScopedTimer(ScopedTimer &&other)
			: profiler(other.profiler), stage(other.stage), start(other.start), active(other.active)
		{
			other.active = false;
		}

		~ScopedTimer()
		{
			stop();
		}

		void stop()
		{
			if(active){
				profiler->add_time(stage, elapsed_ms(start, Clock::now()));
				active = false;
			}
		}

	private:
		using Clock = std::chrono::high_resolution_clock;

		static double elapsed_ms(Clock::time_point start, Clock::time_point end)
		{
			return std::chrono::duration<double, std::milli>(end - start).count();
		}

		QueryProfiler *profiler;
		ProfileStage stage;
		Clock::time_point start;
		bool active;
#else
		ScopedTimer(QueryProfiler *, ProfileStage) {}
		ScopedTimer(const ScopedTimer&) = delete;
		ScopedTimer& operator=(const ScopedTimer&) = delete;
		ScopedTimer(ScopedTimer&&) {}
		void stop() {}
#endif
	};

	QueryProfiler()
	{
		reset();
	}

	void reset()
	{
		stage_time.fill(0.0);
		counts.fill(0);
	}

	void merge(const QueryProfiler &other)
	{
		for(size_t i = 0; i < stage_time.size(); i ++){
			stage_time[i] += other.stage_time[i];
		}
		for(size_t i = 0; i < counts.size(); i ++){
			counts[i] += other.counts[i];
		}
	}

	void add_time(ProfileStage stage, double ms)
	{
#if FARM_PROFILE_QUERY
		stage_time[(size_t)stage] += ms;
#else
		(void)stage;
		(void)ms;
#endif
	}

	void add_count(ProfileCount counter, size_t value)
	{
#if FARM_PROFILE_QUERY
		counts[(size_t)counter] += value;
#else
		(void)counter;
		(void)value;
#endif
	}

	ScopedTimer scoped(ProfileStage stage)
	{
		return ScopedTimer(this, stage);
	}

	double time(ProfileStage stage) const
	{
		return stage_time[(size_t)stage];
	}

	size_t count(ProfileCount counter) const
	{
		return counts[(size_t)counter];
	}

	void print() const
	{
#if FARM_PROFILE_QUERY
		print_within();
		print_intersect();
		print_intersection();
#endif
	}

private:
	std::array<double, (size_t)ProfileStage::Count> stage_time;
	std::array<size_t, (size_t)ProfileCount::Count> counts;

#if FARM_PROFILE_QUERY
	static void print_time(const char *name, double value)
	{
		log("%s:\t%.7f", name, value);
	}

	void print_within() const
	{
		double total = time(ProfileStage::WithinTotal);
		if(total <= 0.0) return;

		double raster = time(ProfileStage::WithinRaster);
		double refine = time(ProfileStage::WithinRefine);

		print_time("within-time-ms", total);
		print_time("within-raster-time-ms", raster);
		print_time("within-refine-time-ms", refine);
		print_time("within-raster-init-time-ms", time(ProfileStage::WithinRasterInit));
		print_time("within-raster-layer-setup-time-ms", time(ProfileStage::WithinRasterLayerSetup));
		print_time("within-raster-buffer-time-ms", time(ProfileStage::WithinRasterBuffer));
		print_time("within-raster-expand-distance-time-ms", time(ProfileStage::WithinRasterExpandDistance));
		print_time("within-refine-candidate-build-time-ms", time(ProfileStage::WithinRefineCandidateBuild));
		print_time("within-refine-candidate-sort-time-ms", time(ProfileStage::WithinRefineCandidateSort));
		print_time("within-refine-suffix-time-ms", time(ProfileStage::WithinRefineSuffix));
		print_time("within-refine-prepare-time-ms", time(ProfileStage::WithinRefinePrepare));
		print_time("within-refine-exact-time-ms", time(ProfileStage::WithinRefineExact));

		log("within-raster-ratio:\t%.7f", raster / total);
		log("within-refine-ratio:\t%.7f", refine / total);

		size_t pixels = count(ProfileCount::WithinRefinePixels);
		size_t edges = count(ProfileCount::WithinRefineEdges);
		if(pixels > 0){
			log("within-refine-pixels:\t%zu", pixels);
			log("within-refine-edges:\t%zu", edges);
			log("within-refine-edges-per-pixel:\t%.7f", (double)edges / pixels);
		}
	}

	void print_intersect() const
	{
		double total = time(ProfileStage::IntersectTotal);
		if(total <= 0.0) return;

		double raster = time(ProfileStage::IntersectRaster);
		double refine = time(ProfileStage::IntersectRefine);

		print_time("intersect-time-ms", total);
		print_time("intersect-raster-time-ms", raster);
		print_time("intersect-refine-time-ms", refine);
		print_time("intersect-candidate-build-time-ms", time(ProfileStage::IntersectCandidateBuild));
		print_time("intersect-candidate-sort-time-ms", time(ProfileStage::IntersectCandidateSort));
		print_time("intersect-exact-edges-time-ms", time(ProfileStage::IntersectExactEdges));
		print_time("intersect-contain-check-time-ms", time(ProfileStage::IntersectContainCheck));
		log("intersect-raster-ratio:\t%.7f", raster / total);
		log("intersect-refine-ratio:\t%.7f", refine / total);
	}

	void print_intersection() const
	{
		double total = time(ProfileStage::IntersectionTotal);
		if(total <= 0.0) return;

		double raster = time(ProfileStage::IntersectionRaster);
		double refine = time(ProfileStage::IntersectionRefine);

		print_time("intersection-time-ms", total);
		print_time("intersection-raster-time-ms", raster);
		print_time("intersection-refine-time-ms", refine);
		print_time("intersection-collect-edges-time-ms", time(ProfileStage::IntersectionCollectEdges));
		print_time("intersection-sort-source-time-ms", time(ProfileStage::IntersectionSortSource));
		print_time("intersection-classify-source-time-ms", time(ProfileStage::IntersectionClassifySource));
		print_time("intersection-area-source-time-ms", time(ProfileStage::IntersectionAreaSource));
		print_time("intersection-sort-target-time-ms", time(ProfileStage::IntersectionSortTarget));
		print_time("intersection-classify-target-time-ms", time(ProfileStage::IntersectionClassifyTarget));
		print_time("intersection-area-target-time-ms", time(ProfileStage::IntersectionAreaTarget));
		log("intersection-raster-ratio:\t%.7f", raster / total);
		log("intersection-refine-ratio:\t%.7f", refine / total);
		log("intersection-edge-seq-pairs:\t%zu", count(ProfileCount::IntersectionEdgeSequencePairs));
		log("intersection-raw-intersections:\t%zu", count(ProfileCount::IntersectionRawIntersections));
		log("intersection-unique-intersections:\t%zu", count(ProfileCount::IntersectionUniqueIntersections));
		log("intersection-classify-source-proper:\t%zu", count(ProfileCount::IntersectionClassifySourceProper));
		log("intersection-classify-source-overlap:\t%zu", count(ProfileCount::IntersectionClassifySourceOverlap));
		log("intersection-classify-source-fallback:\t%zu", count(ProfileCount::IntersectionClassifySourceFallback));
		log("intersection-classify-source-raster-direct:\t%zu", count(ProfileCount::IntersectionClassifySourceRasterDirect));
		log("intersection-classify-source-shared-boundary:\t%zu", count(ProfileCount::IntersectionClassifySourceSharedBoundary));
		log("intersection-classify-source-shared-boundary-direct:\t%zu", count(ProfileCount::IntersectionClassifySourceSharedBoundaryDirect));
		log("intersection-classify-source-shared-boundary-scan:\t%zu", count(ProfileCount::IntersectionClassifySourceSharedBoundaryScan));
		log("intersection-classify-source-border-refine:\t%zu", count(ProfileCount::IntersectionClassifySourceBorderRefine));
		log("intersection-classify-target-proper:\t%zu", count(ProfileCount::IntersectionClassifyTargetProper));
		log("intersection-classify-target-overlap:\t%zu", count(ProfileCount::IntersectionClassifyTargetOverlap));
		log("intersection-classify-target-fallback:\t%zu", count(ProfileCount::IntersectionClassifyTargetFallback));
		log("intersection-classify-target-raster-direct:\t%zu", count(ProfileCount::IntersectionClassifyTargetRasterDirect));
		log("intersection-classify-target-shared-boundary:\t%zu", count(ProfileCount::IntersectionClassifyTargetSharedBoundary));
		log("intersection-classify-target-shared-boundary-direct:\t%zu", count(ProfileCount::IntersectionClassifyTargetSharedBoundaryDirect));
		log("intersection-classify-target-shared-boundary-scan:\t%zu", count(ProfileCount::IntersectionClassifyTargetSharedBoundaryScan));
		log("intersection-classify-target-border-refine:\t%zu", count(ProfileCount::IntersectionClassifyTargetBorderRefine));
	}
#endif
};

class QueryProfileCall {
public:
#if FARM_PROFILE_QUERY
	QueryProfileCall(QueryProfiler &profiler, ProfileStage total_stage, ProfileStage first_phase)
		: profiler(&profiler), total_stage(total_stage), current_phase(first_phase),
		  total_start(Clock::now()), phase_start(total_start), finished(false)
	{
	}

	void finish_phase(ProfileStage phase)
	{
		if(finished) return;
		auto now = Clock::now();
		profiler->add_time(current_phase, elapsed_ms(phase_start, now));
		current_phase = phase;
		phase_start = now;
	}

	template<typename T>
	T finish(T result)
	{
		finish();
		return result;
	}

	void finish()
	{
		if(finished) return;
		auto now = Clock::now();
		profiler->add_time(current_phase, elapsed_ms(phase_start, now));
		profiler->add_time(total_stage, elapsed_ms(total_start, now));
		finished = true;
	}

	~QueryProfileCall()
	{
		finish();
	}

private:
	using Clock = std::chrono::high_resolution_clock;

	static double elapsed_ms(Clock::time_point start, Clock::time_point end)
	{
		return std::chrono::duration<double, std::milli>(end - start).count();
	}

	QueryProfiler *profiler;
	ProfileStage total_stage;
	ProfileStage current_phase;
	Clock::time_point total_start;
	Clock::time_point phase_start;
	bool finished;
#else
	QueryProfileCall(QueryProfiler &, ProfileStage, ProfileStage) {}
	void finish_phase(ProfileStage) {}
	template<typename T>
	T finish(T result) { return result; }
	void finish() {}
#endif
};

#endif
