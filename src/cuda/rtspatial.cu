#include "../include/Ideal.h"
#include "cuda_util.h"
#include <thrust/device_vector.h>
#include <rtspatial/spatial_index.cuh>

#include <optix_function_table_definition.h>

rtspatial::SpatialIndex<coord_t, 2> rt_index;
size_t box_size;
rtspatial::Queue<thrust::pair<uint32_t, uint32_t>> *results = nullptr;

__global__ void PrintResults(pair<uint32_t, uint32_t>* results, uint size){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size){
        printf("%u %u\n", results[x].first, results[x].second);
    }
}

__global__ void convertPairsToIdealPairs(
    const thrust::pair<uint32_t, uint32_t>* inputPairs,
    IdealPair* outputIdealPairs,
    int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        outputIdealPairs[idx].source = inputPairs[idx].first;
        outputIdealPairs[idx].target = inputPairs[idx].second;
        outputIdealPairs[idx].pair_id = idx;
    }
}

void indexBuild(query_context *gctx){
    vector<box> boxes;
    for (auto polygon : gctx->source_ideals)
    {
        boxes.push_back(*polygon->getMBB());
    }
    box_size = boxes.size();
    thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> d_boxes;
    CopyBoxes(boxes, d_boxes);

    rtspatial::Config config;
    rtspatial::Stream stream;
    rtspatial::Stopwatch sw;

    config.ptx_root = "/home/qmh/IDEAL/src/index/ptx";
    config.prefer_fast_build_query = false;
    config.max_geometries = d_boxes.size();

    rt_index.Init(config);
    sw.start();
    rt_index.Insert(
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>>(d_boxes),
        stream.cuda_stream());
    stream.Sync();
    sw.stop();

    double t_load = sw.ms();
    std::cout << "RT, load " << t_load << " ms" << std::endl;

    results = new rtspatial::Queue<thrust::pair<uint32_t, uint32_t>>();
    results->Init(std::max(1ul, (size_t) (gctx->batch_size * box_size * 0.004)));
}

void indexQuery(query_context *gctx){
    rtspatial::Stream stream;
    rtspatial::Stopwatch sw;

    double t_query;
    size_t n_results;
    
    rtspatial::SharedValue<rtspatial::Queue<thrust::pair<uint32_t, uint32_t>>::device_t> d_results;

    if(gctx->query_type == QueryType::contain){
        thrust::device_vector<rtspatial::Point<coord_t, 2>> d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());

        CopyPoints(gctx->points, gctx->target_num, d_queries);
        std::cout << "Loaded point queries " << gctx->target_num << std::endl;

        sw.start();
        rt_index.Query(rtspatial::Predicate::kContains, rtspatial::ArrayView<rtspatial::Point<coord_t, 2>>(d_queries),
                    d_results.data(), stream.cuda_stream());
                
    }else if(gctx->query_type == QueryType::contain_polygon){
        vector<box> queries;
        for(auto polygon : gctx->target_ideals)
        {
            queries.push_back(*polygon->getMBB());
        }

        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());
        
        CopyBoxes(queries, d_queries);
        std::cout << "Loaded box queries " << queries.size() << std::endl;
        
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        sw.start();
        rt_index.Query(rtspatial::Predicate::kContains, v_queries, d_results.data(),
                    stream.cuda_stream());

    }else if(gctx->query_type == QueryType::within){
        vector<box> queries;
        for (int i = gctx->index; i < gctx->index_end; i ++)
        {
            double shiftx = degree_per_kilometer_longitude(gctx->points[i].y) * gctx->within_distance;
            double shifty = degree_per_kilometer_latitude * gctx->within_distance;
            box bx(gctx->points[i].x-shiftx, gctx->points[i].y-shifty, gctx->points[i].x+shiftx, gctx->points[i].y+shifty);
            queries.push_back(bx);
        }
        
        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());
        
        CopyBoxes(queries, d_queries);
        std::cout << "Loaded box queries " << queries.size() << std::endl;

        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        // sw.start();
        rt_index.Query(rtspatial::Predicate::kIntersects, v_queries, d_results.data(),
                   stream.cuda_stream());

    }else{
        vector<box> queries;
        for (auto polygon : gctx->target_ideals)
        {
            queries.push_back(polygon->getMBB()->expand(gctx->within_distance, true));
        }

        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());

        CopyBoxes(queries, d_queries);
        std::cout << "Loaded box queries " << queries.size() << std::endl;

        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        sw.start();
        rt_index.Query(rtspatial::Predicate::kIntersects, v_queries, d_results.data(),
                    stream.cuda_stream());

    }

    n_results = results->size(stream.cuda_stream());
    // sw.stop();
    t_query = sw.ms();
    std::cout << "RT, query " << t_query
              << " ms, results: " << n_results << std::endl;

    auto d_result_ptr = results->data();
    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_candidate_pairs, n_results * sizeof(pair<uint32_t, uint32_t>)));
    gctx->num_pairs = n_results;
    cudaMemcpy(gctx->d_candidate_pairs, d_result_ptr, n_results * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToDevice);

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n_results + threadsPerBlock - 1) / threadsPerBlock;
    // convertPairsToIdealPairs<<<blocksPerGrid, threadsPerBlock>>>(
    //     d_result_ptr, 
    //     gctx->d_candidate_pairs,
    //     n_results);


    // int grid_size_x = (n_results + 256 - 1) / 256;
	// dim3 block_size(256, 1, 1);
	// dim3 grid_size(grid_size_x, 1, 1);
    // PrintResults<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->num_pairs);
    // cudaDeviceSynchronize();

    // printf("n_results = %d\n", n_results);

    // pair<uint32_t, uint32_t>* h_candidate_pairs = new pair<uint32_t, uint32_t>[n_results];
    // cudaMemcpy(h_candidate_pairs, gctx->d_candidate_pairs, n_results * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 100; i ++){
    //     printf("pair%d\n", i);
    //     int source = h_candidate_pairs[i].first;
    //     int target = h_candidate_pairs[i].second;
    //     gctx->points[target].print();
    //     gctx->source_ideals[source]->MyPolygon::print();
    //     // gctx->target_ideals[target]->MyPolygon::print();
    // }
    
}

void indexDestroy(query_context *gctx){
    CUDA_SAFE_CALL(cudaFree(gctx->d_candidate_pairs));
    if(gctx->index_end - gctx->index < gctx->batch_size)
    {
        delete results;
        results = nullptr;
    }
}