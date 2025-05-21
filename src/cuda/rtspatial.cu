#include "../include/Ideal.h"
#include "cuda_util.h"
#include <thrust/device_vector.h>
#include <rtspatial/spatial_index.cuh>

#include <optix_function_table_definition.h>

rtspatial::SpatialIndex<coord_t, 2> *rt_index;
size_t *offset;
rtspatial::Queue<thrust::pair<uint32_t, uint32_t>> *results = nullptr;

__global__ void PrintResults(pair<uint32_t, uint32_t>* results, uint size){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < size){
        printf("%u %u\n", results[x].first, results[x].second);
    }
}

__global__ void PairsCopy(
    thrust::pair<uint32_t, uint32_t> *inputPairs,
    pair<uint32_t, uint32_t> *outputPairs,
    size_t numElements, size_t offset, size_t target_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        outputPairs[idx + offset].first = inputPairs[idx].first;
        outputPairs[idx + offset].second = inputPairs[idx].second + target_start;
        // 如果target是polygon那么还需要再加source ideal的数量
    }
}

void indexBuild(query_context *gctx){
    rt_index = new rtspatial::SpatialIndex<coord_t, 2>;
    offset = new size_t(0);
    results = new rtspatial::Queue<thrust::pair<uint32_t, uint32_t>>();

    vector<box> boxes;
    for (auto polygon : gctx->source_ideals)
    {
        boxes.push_back(*polygon->getMBB());
    }
    thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> d_boxes;
    CopyBoxes(boxes, d_boxes);

    rtspatial::Config config;
    rtspatial::Stream stream;
    rtspatial::Stopwatch sw;

    config.ptx_root = "/home/qmh/IDEAL/src/index/ptx";
    config.prefer_fast_build_query = false;
    config.max_geometries = d_boxes.size();

    rt_index->Init(config);
    sw.start();
    rt_index->Insert(
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>>(d_boxes),
        stream.cuda_stream());
    stream.Sync();
    sw.stop();

    double t_load = sw.ms();
    std::cout << "RT, load " << t_load << " ms" << std::endl;

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_candidate_pairs, gctx->target_num * boxes.size() * gctx->load_factor * sizeof(pair<uint32_t, uint32_t>)));
    results->Init(std::max(1ul, (size_t) (gctx->target_num * boxes.size() * gctx->load_factor)));
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

        CopyPoints(gctx->points + gctx->index, gctx->index_end - gctx->index, d_queries);

        rt_index->Query(rtspatial::Predicate::kContains, rtspatial::ArrayView<rtspatial::Point<coord_t, 2>>(d_queries),
                    d_results.data(), stream.cuda_stream());
                
    }else if(gctx->query_type == QueryType::contain_polygon){
        vector<box> queries;
        for (int i = gctx->index; i < gctx->index_end; i ++)
        {
            queries.push_back(gctx->target_ideals[i]->getMBB());
        }

        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());
        
        CopyBoxes(queries, d_queries);
        
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        rt_index->Query(rtspatial::Predicate::kContains, v_queries, d_results.data(),
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

        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        rt_index->Query(rtspatial::Predicate::kIntersects, v_queries, d_results.data(),
                   stream.cuda_stream());

    }else if(gctx->query_type == QueryType::within_polygon){
        vector<box> queries;
        for (int i = gctx->index; i < gctx->index_end; i ++)
        {
            queries.push_back(gctx->target_ideals[i]->getMBB()->expand(gctx->within_distance, true));
        }

        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());

        CopyBoxes(queries, d_queries);

        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        rt_index->Query(rtspatial::Predicate::kIntersects, v_queries, d_results.data(),
                    stream.cuda_stream());

    }else{
        vector<box> queries;
        for (int i = gctx->index; i < gctx->index_end; i ++)
        {
            queries.push_back(gctx->target_ideals[i]->getMBB());
        }
        printf("querysize = %d\n", queries.size());

        thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > d_queries;
        results->Clear();
        d_results.set(stream.cuda_stream(), results->DeviceObject());
        
        CopyBoxes(queries, d_queries);
        
        rtspatial::ArrayView<rtspatial::Envelope<rtspatial::Point<coord_t, 2> > > v_queries(d_queries);
        
        rt_index->Query(rtspatial::Predicate::kIntersects, v_queries, d_results.data(),
                    stream.cuda_stream());

    }

    n_results = results->size(stream.cuda_stream());
    auto d_result_ptr = results->data();

    gctx->num_pairs += n_results;
    // cudaMemcpy(gctx->d_candidate_pairs + *offset, d_result_ptr, n_results * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToDevice);
 
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_results + threadsPerBlock - 1) / threadsPerBlock;
    if(gctx->query_type == QueryType::contain || gctx->query_type == QueryType::within)
        PairsCopy<<<blocksPerGrid, threadsPerBlock>>>(d_result_ptr, gctx->d_candidate_pairs, n_results, *offset, gctx->index);
    else{
        PairsCopy<<<blocksPerGrid, threadsPerBlock>>>(d_result_ptr, gctx->d_candidate_pairs, n_results, *offset, gctx->index + gctx->source_ideals.size());
    }

    cudaDeviceSynchronize();
    check_execution("PairsCopy");
    // int grid_size_x = (n_results + 256 - 1) / 256;
	// dim3 block_size(256, 1, 1);
	// dim3 grid_size(grid_size_x, 1, 1);
    // PrintResults<<<grid_size, block_size>>>(gctx->d_candidate_pairs, gctx->num_pairs);
    // cudaDeviceSynchronize();

    *offset += n_results;
    
}

void indexDestroy(query_context *gctx){
    delete results;
    results = nullptr;
    delete rt_index;
    rt_index = nullptr;

    printf("num_pairs = %u\n", *offset);

    // compress d_candidate_pairs
    pair<uint32_t, uint32_t> *new_ptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&new_ptr, (*offset) * sizeof(pair<uint32_t, uint32_t>)));
    CUDA_SAFE_CALL(cudaMemcpy(new_ptr, gctx->d_candidate_pairs, (*offset) * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaFree(gctx->d_candidate_pairs));
    gctx->d_candidate_pairs = new_ptr;

    // pair<uint32_t, uint32_t>* h_candidate_pairs = new pair<uint32_t, uint32_t>[*offset];
    // cudaMemcpy(h_candidate_pairs, gctx->d_candidate_pairs, *offset * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < *offset; i ++){
    //     printf("pair%d\n", i);
    //     int source = h_candidate_pairs[i].first;
    //     int target = h_candidate_pairs[i].second;
    //     // printf("source = %d target = %d\n", source, target);
    //     // gctx->points[target].print();
    //     gctx->source_ideals[source]->MyPolygon::print();
    //     gctx->target_ideals[target - gctx->source_ideals.size()]->MyPolygon::print();
    // }

    delete offset;
    offset = nullptr;

    gctx->index = 0;
    gctx->index_end = 0;
    
    gctx->h_candidate_pairs = new pair<uint32_t, uint32_t>[gctx->num_pairs];
	cudaMemcpy(gctx->h_candidate_pairs, gctx->d_candidate_pairs, gctx->num_pairs * sizeof(pair<uint32_t, uint32_t>), cudaMemcpyDeviceToHost);
    return;
}