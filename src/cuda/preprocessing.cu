#include "cuda_util.h"
#include "../include/farm.h"
#include "util.h"

void cuda_create_buffer(query_context *gctx)
{
    constexpr size_t longitude_table_size =
        sizeof(degree_per_kilometer_longitude_arr) / sizeof(degree_per_kilometer_longitude_arr[0]);
    size_t source_count = gctx->source_objects.size();
    size_t num_polygons = source_count + gctx->target_objects.size();
    
    size_t num_status = 0;
    size_t num_offset = 0;
    size_t num_edge_sequences = 0;
    size_t num_vertices = 0;
    size_t num_gridline_offset = 0;
    size_t num_gridline_nodes = 0;
    size_t num_layers = 0;

    for (size_t i = 0; i < gctx->target_objects.size(); i++)
    {
        gctx->target_objects[i]->id = source_count + i;
    }

    auto count_objects = [&](Farm *obj) {
        num_status += obj->get_status_size();
        num_offset += obj->get_num_pixels() + 1;
        num_edge_sequences += obj->get_len_edge_sequences();
        num_vertices += obj->get_num_vertices();
        num_gridline_offset += obj->get_vertical()->get_num_grid_lines();
        num_gridline_nodes += obj->get_vertical()->get_num_crosses();
        if (gctx->use_hierarchy)
        {
            num_layers += obj->get_num_layers();
        }
    };

    if (!gctx->referred_objects.empty())
    {
        for (Farm *obj : gctx->referred_objects)
        {
            count_objects(obj);
        }
    }
    else
    {
        for (Farm *obj : gctx->source_objects)
        {
            count_objects(obj);
        }
        for (Farm *obj : gctx->target_objects)
        {
            count_objects(obj);
        }
    }

    gctx->num_polygons = num_polygons;
    gctx->num_status = num_status;
    gctx->num_offset = num_offset;
    gctx->num_edge_sequences = num_edge_sequences;
    gctx->num_vertices = num_vertices;
    gctx->num_gridline_offset = num_gridline_offset;
    gctx->num_gridline_nodes = num_gridline_nodes;
    if(gctx->use_hierarchy){
        gctx->num_layers = num_layers;
    }

    log("CPU Memory Allocation (Pinned):\n");

    alloc_host_pinned(&gctx->h_farm_offset,     num_polygons + 1,    "farm offset");
    memset(gctx->h_farm_offset, 0, sizeof(FarmOffset) * (num_polygons + 1));
    alloc_host_pinned(&gctx->h_info,            num_polygons,        "raster info");

    if(gctx->use_hierarchy){
        alloc_host_pinned(&gctx->h_layer_info,   num_layers,          "layer info");
        alloc_host_pinned(&gctx->h_layer_offset, num_layers,          "layer offset");
    }
        
    alloc_host_pinned(&gctx->h_status,           num_status,          "status");
    alloc_host_pinned(&gctx->h_offset,           num_offset,          "offset");
    alloc_host_pinned(&gctx->h_edge_sequences,   num_edge_sequences,  "edge sequences");
    alloc_host_pinned(&gctx->h_vertices,         num_vertices,        "vertices");
    alloc_host_pinned(&gctx->h_gridline_offset,  num_gridline_offset, "grid line offset");
    alloc_host_pinned(&gctx->h_gridline_nodes,   num_gridline_nodes,  "grid line nodes");

    alloc_host_pinned(&gctx->h_degree_degree_per_kilometer_latitude, 1, "degree per kilometer (latitude)");
    alloc_host_pinned(&gctx->h_degree_per_kilometer_longitude_arr, longitude_table_size, "degree per kilometer (longitude)");

    log("GPU Memory Allocation:\n");

    alloc_device(&gctx->d_farm_offset, num_polygons + 1, "farm offset");
    alloc_device(&gctx->d_info, num_polygons, "raster info");

    if (gctx->use_hierarchy){
        alloc_device(&gctx->d_layer_info, num_layers, "layer info");
        alloc_device(&gctx->d_layer_offset, num_layers, "layer offset");
    }

    alloc_device(&gctx->d_status, num_status, "status");
    alloc_device(&gctx->d_offset, num_offset, "offset");
    alloc_device(&gctx->d_edge_sequences, num_edge_sequences, "edge sequences");
    alloc_device(&gctx->d_vertices, num_vertices, "vertices");
    alloc_device(&gctx->d_gridline_offset, num_gridline_offset, "grid line offset");
    alloc_device(&gctx->d_gridline_nodes, num_gridline_nodes, "grid line nodes");

    alloc_device(&gctx->d_degree_degree_per_kilometer_latitude, 1, "degree per kilometer (latitude)");
    alloc_device(&gctx->d_degree_per_kilometer_longitude_arr, longitude_table_size, "degree per kilometer (longitude)");

    log("Scratchpad & Pairs Allocation:\n");

    alloc_device(&gctx->d_BufferInput, CUDA_SCRATCH_BUFFER_BYTES, "Buffer Input");
    alloc_device(&gctx->d_BufferOutput, CUDA_SCRATCH_BUFFER_BYTES, "Buffer Output");
    
    alloc_device(&gctx->d_bufferinput_size,  1, "input size");
    alloc_device(&gctx->d_bufferoutput_size, 1, "output size");
    alloc_device(&gctx->d_result,            1, "result count");

    size_t num_pairs = gctx->num_pairs;
    if (num_pairs > 0) {
        alloc_host_pinned(&gctx->h_candidate_pairs, num_pairs, "candidate pairs (Host)");
        alloc_device(&gctx->d_candidate_pairs, num_pairs, "candidate pairs (Device)");
    }
}

void preprocess_for_gpu(query_context *gctx)
{
    constexpr size_t longitude_table_bytes = sizeof(degree_per_kilometer_longitude_arr);
    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint32_t sidx = 0; // status
    uint32_t oidx = 0; // offset
    uint32_t eidx = 0; // edge
    uint32_t vidx = 0; // vertices
    uint32_t goidx = 0; // gridline offset
    uint32_t gnidx = 0; // gridline nodes
    uint32_t lidx = 0; // layers

    vector<Farm *> upload_objects;
    if (!gctx->referred_objects.empty()) {
        upload_objects = gctx->referred_objects;
    } else {
        upload_objects.insert(upload_objects.end(), gctx->source_objects.begin(), gctx->source_objects.end());
        upload_objects.insert(upload_objects.end(), gctx->target_objects.begin(), gctx->target_objects.end());
    }
    sort(upload_objects.begin(), upload_objects.end(), [](Farm *lhs, Farm *rhs) {
        return lhs->id < rhs->id;
    });

    auto set_offset_cursor = [&](uint32_t idx) {
        gctx->h_farm_offset[idx] = {sidx, oidx, eidx, vidx, goidx, gnidx, lidx};
    };

    uint32_t next_offset_slot = 0;

    auto serialize_objects = [&](auto* obj) {
        uint32_t id = obj->id;
        assert(id < gctx->num_polygons);
        assert(id >= next_offset_slot);

        // Kernels use farm_offset[id + 1] as this polygon's end cursor;
        // fill skipped global ids as empty ranges before writing the next object.
        while (next_offset_slot <= id) {
            set_offset_cursor(next_offset_slot++);
        }

        FarmOffset& farm_offset = gctx->h_farm_offset[id];

        // Info & MBB
        int dimx = obj->get_dimx();
        int dimy = obj->get_dimy();
        gctx->h_info[id] = {*obj->getMBB(), dimx, dimy, obj->get_step_x(), obj->get_step_y()};

        // Status
        uint32_t size_status = obj->get_status_size();
        memcpy(gctx->h_status + sidx, obj->get_status(), size_status);
        farm_offset.status_start = sidx;
        sidx += size_status;

        // Offset Map
        uint32_t size_offset = dimx * dimy + 1;
        memcpy(gctx->h_offset + oidx, obj->get_offset(), size_offset * sizeof(uint32_t));
        farm_offset.offset_start = oidx;
        oidx += size_offset;

        // Edge Sequences
        uint32_t size_edges = obj->get_len_edge_sequences();
        memcpy(gctx->h_edge_sequences + eidx, obj->get_edge_sequence(), size_edges * sizeof(EdgeSeq));
        farm_offset.edge_sequences_start = eidx;
        eidx += size_edges;

        // Vertices
        uint32_t size_verts = obj->get_num_vertices();
        memcpy(gctx->h_vertices + vidx, obj->get_boundary()->p, size_verts * sizeof(Point));
        farm_offset.vertices_start = vidx;
        vidx += size_verts;

        // Gridline Meta
        uint32_t size_go = obj->get_vertical()->get_num_grid_lines();
        memcpy(gctx->h_gridline_offset + goidx, obj->get_vertical()->get_offset(), size_go * sizeof(uint32_t));
        farm_offset.gridline_offset_start = goidx;
        goidx += size_go;

        uint32_t size_gn = obj->get_vertical()->get_num_crosses();
        memcpy(gctx->h_gridline_nodes + gnidx, obj->get_vertical()->get_intersection_nodes(), size_gn * sizeof(double));
        farm_offset.gridline_nodes_start = gnidx;
        gnidx += size_gn;

        // Hierarchy (Optional)
        if(gctx->use_hierarchy){
            uint32_t size_layer = obj->get_num_layers();
            memcpy(gctx->h_layer_info + lidx, obj->get_layer_info(), size_layer * sizeof(RasterInfo));
            memcpy(gctx->h_layer_offset + lidx, obj->get_layer_offset(), size_layer * sizeof(uint32_t));
            farm_offset.layer_start = lidx;
            lidx += size_layer;
        }
    };

    for (Farm *obj : upload_objects) {
        serialize_objects(obj);
    }

    while (next_offset_slot <= gctx->num_polygons) {
        set_offset_cursor(next_offset_slot++);
    }

    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_farm_offset, gctx->h_farm_offset, (gctx->num_polygons + 1) * sizeof(FarmOffset), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_info, gctx->h_info, gctx->num_polygons * sizeof(RasterInfo), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_status, gctx->h_status, gctx->num_status * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_offset, gctx->h_offset, gctx->num_offset * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_edge_sequences, gctx->h_edge_sequences, gctx->num_edge_sequences * sizeof(EdgeSeq), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_vertices, gctx->h_vertices, gctx->num_vertices * sizeof(Point), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_gridline_offset, gctx->h_gridline_offset, gctx->num_gridline_offset * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_gridline_nodes, gctx->h_gridline_nodes, gctx->num_gridline_nodes * sizeof(double), cudaMemcpyHostToDevice, stream));

    if(gctx->use_hierarchy){
        CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_layer_info, gctx->h_layer_info, gctx->num_layers * sizeof(RasterInfo), cudaMemcpyHostToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_layer_offset, gctx->h_layer_offset, gctx->num_layers * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    }

    memcpy(gctx->h_degree_degree_per_kilometer_latitude, &degree_per_kilometer_latitude, sizeof(float));
    memcpy(gctx->h_degree_per_kilometer_longitude_arr, degree_per_kilometer_longitude_arr,
        longitude_table_bytes);
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_degree_degree_per_kilometer_latitude, gctx->h_degree_degree_per_kilometer_latitude, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_degree_per_kilometer_longitude_arr,
        gctx->h_degree_per_kilometer_longitude_arr, longitude_table_bytes,
        cudaMemcpyHostToDevice, stream));

    CUDA_SAFE_CALL(cudaMemsetAsync(gctx->d_bufferinput_size, 0, sizeof(uint), stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(gctx->d_bufferoutput_size, 0, sizeof(uint), stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(gctx->d_result, 0, sizeof(uint), stream));

    if (gctx->num_pairs > 0 && gctx->h_candidate_pairs != nullptr) {
        memcpy(gctx->h_candidate_pairs, gctx->object_pairs.data(), gctx->num_pairs * sizeof(std::pair<uint32_t, uint32_t>));
        CUDA_SAFE_CALL(cudaMemcpyAsync(gctx->d_candidate_pairs, gctx->h_candidate_pairs, gctx->num_pairs * sizeof(std::pair<uint32_t, uint32_t>), cudaMemcpyHostToDevice, stream));

    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));    
    CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}
