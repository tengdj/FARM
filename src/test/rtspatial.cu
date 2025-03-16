/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#include "../include/Ideal.h"
#include <chrono>
#include <thrust/device_vector.h>
#include "../cuda/cuda_util.h"
#include <rtspatial/spatial_index.cuh>

#include <optix_function_table_definition.h>

// some shared parameters

int main(int argc, char **argv)
{
    query_context global_ctx;
    global_ctx = get_parameters(argc, argv);
    global_ctx.query_type = QueryType::contain;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
    
    int limit_box = std::numeric_limits<int>::max();
    int limit_query = std::numeric_limits<int>::max();
    vector<box> boxes;
    for(auto polygon : global_ctx.source_ideals) {
        boxes.push_back(*polygon->getMBB());
    }
    thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_boxes;
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
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > >(d_boxes),
        stream.cuda_stream());
    stream.Sync();
    sw.stop();

    double t_load = sw.ms(), t_query;
    size_t n_results;
    rtspatial::Queue<thrust::pair<uint32_t, uint32_t> > results;
    rtspatial::SharedValue<rtspatial::Queue<thrust::pair<uint32_t, uint32_t> >::device_t> d_results;

    global_ctx.load_points();
    // Point* queries = load_point_wkt(global_ctx.target_path.c_str(), global_ctx.target_num, &global_ctx);
    thrust::device_vector<rtspatial::Point<coord_t, 2> > d_queries;

    results.Init(std::max(
        1ul, (size_t) (boxes.size() * global_ctx.target_num)));
    d_results.set(stream.cuda_stream(), results.DeviceObject());

    CopyPoints(global_ctx.points, global_ctx.target_num, d_queries);
    std::cout << "Loaded point queries " << global_ctx.target_num << std::endl;

    sw.start();
    index.Query(rtspatial::Predicate::kContains, rtspatial::ArrayView<rtspatial::Point<coord_t, 2> >(d_queries),
                d_results.data(), stream.cuda_stream());
    n_results = results.size(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();

    std::cout << "RT, load " << t_load << " ms, query " << t_query
            << " ms, results: " << n_results << std::endl;

    return 0;
}