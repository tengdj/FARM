#include "../include/Ideal.h"
#include "cuda_util.h"
#include <thrust/device_vector.h>
#include <rtspatial/spatial_index.cuh>

#include <optix_function_table_definition.h>

__global__ void PrintResults(pair<uint32_t, uint32_t>* results, uint size){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size){
        printf("%u %u\n", results[x].first, results[x].second);
    }
}

void indexFilter(query_context *gctx){
    int limit_box = std::numeric_limits<int>::max();
    int limit_query = std::numeric_limits<int>::max();
    vector<box> boxes;
    for (auto polygon : gctx->source_ideals)
    {
        boxes.push_back(*polygon->getMBB());
    }
	thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> d_boxes;
    CopyBoxes(boxes, d_boxes);

    rtspatial::SpatialIndex<coord_t, 2> index;
    rtspatial::Config config;
    rtspatial::Stream stream;
    rtspatial::Stopwatch sw;

    config.ptx_root = "/home/qmh/IDEAL/src/index/ptx";
    config.prefer_fast_build_query = false;
    config.max_geometries = d_boxes.size();

    index.Init(config);
    sw.start();
    index.Insert(
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>>(d_boxes),
        stream.cuda_stream());
    stream.Sync();
    sw.stop();

    double t_load = sw.ms(), t_query;
    size_t n_results;
    rtspatial::Queue<thrust::pair<uint32_t, uint32_t>> results;
    rtspatial::SharedValue<rtspatial::Queue<thrust::pair<uint32_t, uint32_t>>::device_t> d_results;

    thrust::device_vector<rtspatial::Point<coord_t, 2>> d_queries;
    results.Init(std::max(
        1ul, (size_t)(boxes.size() * gctx->target_num)));
    d_results.set(stream.cuda_stream(), results.DeviceObject());

	CopyPoints(gctx->points, gctx->target_num, d_queries);
    std::cout << "Loaded point queries " << gctx->target_num << std::endl;

    sw.start();
    index.Query(rtspatial::Predicate::kContains, rtspatial::ArrayView<rtspatial::Point<coord_t, 2>>(d_queries),
                d_results.data(), stream.cuda_stream());
    n_results = results.size(stream.cuda_stream());
    sw.stop();

    t_query = sw.ms();
    std::cout << "RT, load " << t_load << " ms, query " << t_query
              << " ms, results: " << n_results << std::endl;

    auto d_result_ptr = results.data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_candidate_pairs, n_results * sizeof(pair<uint32_t, uint32_t>)));
    gctx->num_pairs = n_results;
    cudaMemcpy(gctx->d_candidate_pairs, d_result_ptr, n_results * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToDevice);

    // int grid_size_x = (n_results + 256 - 1) / 256;
	// dim3 block_size(256, 1, 1);
	// dim3 grid_size(grid_size_x, 1, 1);
    // PrintResults<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->num_pairs);
    // cudaDeviceSynchronize();
    
}