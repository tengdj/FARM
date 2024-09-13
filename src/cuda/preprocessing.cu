#include "cuda_util.h"
#include "../include/Ideal.h"
#include "mygpu.h"

void cuda_create_buffer(query_context *gctx)
{
    size_t num_polygons = gctx->source_ideals.size() + gctx->target_ideals.size();
    size_t num_status = 0;
    size_t num_offset = 0;
    size_t num_edge_sequences = 0;
    size_t num_vertices = 0;
    size_t num_gridline_offset = 0;
    size_t num_gridline_nodes = 0;

    for (auto &ideal : gctx->source_ideals)
    {
        num_status += ideal->get_num_pixels() / 4 + 1;
        num_offset += ideal->get_num_pixels() + 1;
        num_edge_sequences += ideal->get_len_edge_sequences();
        num_vertices += ideal->get_num_vertices();
        num_gridline_offset += ideal->get_vertical()->get_num_grid_lines();
        num_gridline_nodes += ideal->get_vertical()->get_num_crosses();
    }
    for (auto &ideal : gctx->target_ideals)
    {
        num_status += ideal->get_num_pixels() / 4 + 1;
        num_offset += ideal->get_num_pixels() + 1;
        num_edge_sequences += ideal->get_len_edge_sequences();
        num_vertices += ideal->get_num_vertices();
        num_gridline_offset += ideal->get_vertical()->get_num_grid_lines();
        num_gridline_nodes += ideal->get_vertical()->get_num_crosses();
    }

    log("CPU momory:");

    gctx->h_info = new Idealinfo[num_polygons];
    log("\t%.2f MB\t\tideal info", 1.0 * sizeof(Idealinfo) * num_polygons / 1024 / 1024);

    gctx->h_status = new uint8_t[num_status];
    log("\t%.2f MB\t\tstatus", 1.0 * sizeof(uint8_t) * num_status / 1024 / 1024);

    gctx->h_offset = new uint16_t[num_offset];
    log("\t%.2f MB\toffset", 1.0 * sizeof(uint16_t) * num_offset / 1024 / 1024);

    gctx->h_edge_sequences = new EdgeSeq[num_edge_sequences];
    log("\t%.2f MB\tedge sequences", 1.0 * sizeof(EdgeSeq) * num_edge_sequences / 1024 / 1024);

    gctx->h_vertices = new Point[num_vertices];
    log("\t%.2f MB\tvertices", 1.0 * sizeof(Point) * num_vertices / 1024 / 1024);

    gctx->h_gridline_offset = new uint16_t[num_gridline_offset];
    log("\t%.2f MB\t\tgrid line offset", 1.0 * sizeof(uint16_t) * num_gridline_offset / 1024 / 1024);

    gctx->h_gridline_nodes = new double[num_gridline_nodes];
    log("\t%.2f MB\tgrid line nodes", 1.0 * sizeof(double) * num_gridline_nodes / 1024 / 1024);

    log("GPU memory:");
    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_info, sizeof(Idealinfo) * num_polygons));
    log("\t%.2f MB\t\tideal info", 1.0 * sizeof(Idealinfo) * num_polygons / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_status, sizeof(uint8_t) * num_status));
    log("\t%.2f MB\t\tstatus", 1.0 * sizeof(uint8_t) * num_status / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_offset, sizeof(uint16_t) * num_offset));
    log("\t%.2f MB\toffset", 1.0 * sizeof(uint16_t) * num_offset / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_edge_sequences, sizeof(EdgeSeq) * num_edge_sequences));
    log("\t%.2f MB\tedge sequences", 1.0 * sizeof(EdgeSeq) * num_edge_sequences / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_vertices, sizeof(Point) * num_vertices));
    log("\t%.2f MB\tvertices", 1.0 * sizeof(Point) * num_vertices / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_gridline_offset, sizeof(uint16_t) * num_gridline_offset));
    log("\t%.2f MB\t\tgrid line offset", 1.0 * sizeof(uint16_t) * num_gridline_offset / 1024 / 1024);

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_gridline_nodes, sizeof(double) * num_gridline_nodes));
    log("\t%.2f MB\tgrid line nodes", 1.0 * sizeof(double) * num_gridline_nodes / 1024 / 1024);

    gctx->num_polygons = num_polygons;
    gctx->num_status = num_status;
    gctx->num_offset = num_offset;
    gctx->num_edge_sequences = num_edge_sequences;
    gctx->num_vertices = num_vertices;
    gctx->num_gridline_offset = num_gridline_offset;
    gctx->num_gridline_nodes = num_gridline_nodes;
}

void preprocess_for_gpu(query_context *gctx)
{
    bool flag1 = false, flag2 = false;
    // compact data
    uint iidx = 0, sidx = 0, oidx = 0, eidx = 0, vidx = 0, goidx = 0, gnidx = 0;
    for (auto &tp : gctx->point_polygon_pairs)
    {
        flag1 = true;
        Ideal *source = tp.second;
        int dimx = source->get_dimx(), dimy = source->get_dimy();
        if (source->idealoffset == nullptr)
        {
            source->idealoffset = new IdealOffset{};

            uint info_size = gctx->polygon_pairs.size();
            Idealinfo idealinfo{source->getMBB(), dimx, dimy, source->get_step_x(), source->get_step_y()};
            memcpy(gctx->h_info + iidx, &idealinfo, sizeof(Idealinfo));
            source->idealoffset->info_start = iidx;
            iidx++;
            source->idealoffset->info_end = iidx;

            uint status_size = (dimx + 1) * (dimy + 1) / 4 + 1;
            memcpy(gctx->h_status + sidx, source->get_status(), status_size);
            source->idealoffset->status_start = sidx;
            sidx += status_size;
            source->idealoffset->status_end = sidx;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, source->get_offset(), offset_size * sizeof(uint16_t));
            source->idealoffset->offset_start = oidx;
            oidx += offset_size;
            source->idealoffset->offset_end = oidx;

            uint edge_sequences_size = source->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, source->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            source->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;
            source->idealoffset->edge_sequences_end = eidx;

            uint vertices_size = source->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, source->get_boundary()->p, vertices_size * sizeof(Point));
            source->idealoffset->vertices_start = vidx;
            vidx += vertices_size;
            source->idealoffset->vertices_end = vidx;

            uint gridline_offset_size = source->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, source->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            source->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            source->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = source->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, source->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            source->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;
            source->idealoffset->gridline_nodes_end = gnidx;
        }
    }

    for (auto &tp : gctx->polygon_pairs)
    {
        flag2 = true;
        Ideal *source = tp.first;
        int dimx = source->get_dimx(), dimy = source->get_dimy();
        if (source->idealoffset == nullptr)
        {
            source->idealoffset = new IdealOffset{};

            uint info_size = gctx->polygon_pairs.size();
            Idealinfo idealinfo{source->getMBB(), dimx, dimy, source->get_step_x(), source->get_step_y()};
            memcpy(gctx->h_info + iidx, &idealinfo, sizeof(Idealinfo));
            source->idealoffset->info_start = iidx;
            iidx++;
            source->idealoffset->info_end = iidx;

            uint status_size = (dimx + 1) * (dimy + 1) / 4 + 1;
            memcpy(gctx->h_status + sidx, source->get_status(), status_size);
            source->idealoffset->status_start = sidx;
            sidx += status_size;
            source->idealoffset->status_end = sidx;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, source->get_offset(), offset_size * sizeof(uint16_t));
            source->idealoffset->offset_start = oidx;
            oidx += offset_size;
            source->idealoffset->offset_end = oidx;

            uint edge_sequences_size = source->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, source->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            source->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;
            source->idealoffset->edge_sequences_end = eidx;

            uint vertices_size = source->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, source->get_boundary()->p, vertices_size * sizeof(Point));
            source->idealoffset->vertices_start = vidx;
            vidx += vertices_size;
            source->idealoffset->vertices_end = vidx;

            uint gridline_offset_size = source->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, source->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            source->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            source->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = source->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, source->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            source->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;
            source->idealoffset->gridline_nodes_end = gnidx;
        }

        Ideal *target = tp.second;
        dimx = target->get_dimx(), dimy = target->get_dimy();
        if (target->idealoffset == nullptr)
        {
            target->idealoffset = new IdealOffset{};

            uint info_size = gctx->polygon_pairs.size();
            Idealinfo idealinfo{target->getMBB(), dimx, dimy, target->get_step_x(), target->get_step_y()};
            memcpy(gctx->h_info + iidx, &idealinfo, sizeof(Idealinfo));
            target->idealoffset->info_start = iidx;
            iidx++;
            target->idealoffset->info_end = iidx;

            uint status_size = (dimx + 1) * (dimy + 1) / 4 + 1;
            memcpy(gctx->h_status + sidx, target->get_status(), status_size);
            target->idealoffset->status_start = sidx;
            sidx += status_size;
            target->idealoffset->status_end = sidx;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, target->get_offset(), offset_size * sizeof(uint16_t));
            target->idealoffset->offset_start = oidx;
            oidx += offset_size;
            target->idealoffset->offset_end = oidx;

            uint edge_sequences_size = target->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, target->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            target->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;
            target->idealoffset->edge_sequences_end = eidx;

            uint vertices_size = target->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, target->get_boundary()->p, vertices_size * sizeof(Point));
            target->idealoffset->vertices_start = vidx;
            vidx += vertices_size;
            target->idealoffset->vertices_end = vidx;

            uint gridline_offset_size = target->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, target->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            target->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            target->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = target->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, target->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            target->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;
            target->idealoffset->gridline_nodes_end = gnidx;
        }
    }

    assert(flag1 ^ flag2);

    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_info, gctx->h_info, gctx->num_polygons * sizeof(Idealinfo), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_status, gctx->h_status, gctx->num_status * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_offset, gctx->h_offset, gctx->num_offset * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_edge_sequences, gctx->h_edge_sequences, gctx->num_edge_sequences * sizeof(EdgeSeq), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_vertices, gctx->h_vertices, gctx->num_vertices * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_gridline_offset, gctx->h_gridline_offset, gctx->num_gridline_offset * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_gridline_nodes, gctx->h_gridline_nodes, gctx->num_gridline_nodes * sizeof(double), cudaMemcpyHostToDevice));
}