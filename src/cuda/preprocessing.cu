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
    size_t num_layer_info = 0;
    size_t num_layer_offset = 0;

    for (auto &ideal : gctx->source_ideals)
    {
        num_status += ideal->get_status_size();
        num_offset += ideal->get_num_pixels() + 1;
        num_edge_sequences += ideal->get_len_edge_sequences();
        num_vertices += ideal->get_num_vertices();
        num_gridline_offset += ideal->get_vertical()->get_num_grid_lines();
        num_gridline_nodes += ideal->get_vertical()->get_num_crosses();
        if(gctx->use_hierachy){
            num_layer_info += ideal->get_num_layers() + 1;
            num_layer_offset += ideal->get_num_layers() + 1;
        }
    }
    for (auto &ideal : gctx->target_ideals)
    {
        num_status += ideal->get_status_size();
        num_offset += ideal->get_num_pixels() + 1;
        num_edge_sequences += ideal->get_len_edge_sequences();
        num_vertices += ideal->get_num_vertices();
        num_gridline_offset += ideal->get_vertical()->get_num_grid_lines();
        num_gridline_nodes += ideal->get_vertical()->get_num_crosses();
        if(gctx->use_hierachy){
            num_layer_info += ideal->get_num_layers() + 1;
            num_layer_offset += ideal->get_num_layers() + 1;
        }
    }

    log("CPU momory:");

    gctx->h_info = new RasterInfo[num_polygons];
    log("\t%.2f MB\t\tideal info", 1.0 * sizeof(RasterInfo) * num_polygons / 1024 / 1024);

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

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_info, sizeof(RasterInfo) * num_polygons));
    log("\t%.2f MB\t\tideal info", 1.0 * sizeof(RasterInfo) * num_polygons / 1024 / 1024);

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
    
    if(gctx->use_hierachy){
        gctx->h_layer_info = new RasterInfo[num_layer_info];
        log("\t%.2f MB\tlayer info", 1.0 * sizeof(RasterInfo) * num_layer_info / 1024 / 1024);
        gctx->h_layer_offset = new uint16_t[num_layer_offset];
        log("\t%.2f MB\tlayer offset", 1.0 * sizeof(uint16_t) * num_layer_offset / 1024 / 1024);

        CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_layer_info, sizeof(RasterInfo) * num_layer_info));
        log("\t%.2f MB\tlayer info", 1.0 * sizeof(RasterInfo) * num_layer_info / 1024 / 1024);
        CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_layer_offset, sizeof(uint16_t) * num_layer_offset));
        log("\t%.2f MB\tlayer offset", 1.0 * sizeof(uint16_t) * num_layer_offset / 1024 / 1024);

        gctx->num_layer_info = num_layer_info;
        gctx->num_layer_offset = num_layer_offset;
    }
}

void preprocess_for_gpu(query_context *gctx)
{
    bool flag1 = false, flag2 = false;
    // compact data
    uint iidx = 0, sidx = 0, oidx = 0, eidx = 0, vidx = 0, goidx = 0, gnidx = 0, liidx = 0, loidx = 0;
    // for(int i = 0; i < gctx->point_polygon_pairs_size; i ++)
    for (auto &tp : gctx->point_polygon_pairs)
    {
        flag1 = true;
        // Ideal *source = gctx->point_polygon_pairs[i].second;
        Ideal *source = tp.second;
        int dimx = source->get_dimx(), dimy = source->get_dimy();
        if (source->idealoffset == nullptr)
        {
            source->idealoffset = new IdealOffset{};

            uint info_size = gctx->polygon_pairs.size();
            RasterInfo rasterinfo{source->getMBB(), dimx, dimy, source->get_step_x(), source->get_step_y()};
            memcpy(gctx->h_info + iidx, &rasterinfo, sizeof(RasterInfo));
            source->idealoffset->info_start = iidx;
            iidx++;

            uint status_size = source->get_status_size();
            memcpy(gctx->h_status + sidx, source->get_status(), status_size);
            source->idealoffset->status_start = sidx;
            sidx += status_size;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, source->get_offset(), offset_size * sizeof(uint16_t));
            source->idealoffset->offset_start = oidx;
            oidx += offset_size;

            uint edge_sequences_size = source->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, source->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            source->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;

            uint vertices_size = source->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, source->get_boundary()->p, vertices_size * sizeof(Point));
            source->idealoffset->vertices_start = vidx;
            vidx += vertices_size;

            uint gridline_offset_size = source->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, source->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            source->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            source->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = source->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, source->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            source->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;

            if(gctx->use_hierachy){
                uint layer_info_size = source->get_num_layers() + 1;
                memcpy(gctx->h_layer_info + liidx, source->get_layer_info(), layer_info_size * sizeof(RasterInfo));
                source->idealoffset->layer_info_start = liidx;
                liidx += layer_info_size;

                uint layer_offset_size = source->get_num_layers() + 1;
                memcpy(gctx->h_layer_offset + loidx, source->get_layer_offset(), layer_offset_size * sizeof(uint16_t));
                source->idealoffset->layer_offset_start = loidx;
                loidx += layer_offset_size;
            }
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
            RasterInfo rasterinfo{source->getMBB(), dimx, dimy, source->get_step_x(), source->get_step_y()};
            memcpy(gctx->h_info + iidx, &rasterinfo, sizeof(RasterInfo));
            source->idealoffset->info_start = iidx;
            iidx++;

            uint status_size = source->get_status_size();
            memcpy(gctx->h_status + sidx, source->get_status(), status_size);
            source->idealoffset->status_start = sidx;
            sidx += status_size;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, source->get_offset(), offset_size * sizeof(uint16_t));
            source->idealoffset->offset_start = oidx;
            oidx += offset_size;

            uint edge_sequences_size = source->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, source->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            source->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;

            uint vertices_size = source->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, source->get_boundary()->p, vertices_size * sizeof(Point));
            source->idealoffset->vertices_start = vidx;
            vidx += vertices_size;

            uint gridline_offset_size = source->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, source->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            source->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            source->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = source->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, source->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            source->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;

            if(gctx->use_hierachy){
                uint layer_info_size = source->get_num_layers() + 1;
                memcpy(gctx->h_layer_info + liidx, source->get_layer_info(), layer_info_size * sizeof(RasterInfo));
                source->idealoffset->layer_info_start = liidx;
                liidx += layer_info_size;

                uint layer_offset_size = source->get_num_layers() + 1;
                memcpy(gctx->h_layer_offset + loidx, source->get_layer_offset(), layer_offset_size * sizeof(uint16_t));
                source->idealoffset->layer_offset_start = loidx;
                loidx += layer_offset_size;
            }
        }

        Ideal *target = tp.second;
        dimx = target->get_dimx(), dimy = target->get_dimy();
        if (target->idealoffset == nullptr)
        {
            target->idealoffset = new IdealOffset{};

            uint info_size = gctx->polygon_pairs.size();
            RasterInfo rasterinfo{target->getMBB(), dimx, dimy, target->get_step_x(), target->get_step_y()};
            memcpy(gctx->h_info + iidx, &rasterinfo, sizeof(RasterInfo));
            target->idealoffset->info_start = iidx;
            iidx++;

            uint status_size = target->get_status_size();
            memcpy(gctx->h_status + sidx, target->get_status(), status_size);
            target->idealoffset->status_start = sidx;
            sidx += status_size;

            uint offset_size = (dimx + 1) * (dimy + 1) + 1;
            memcpy(gctx->h_offset + oidx, target->get_offset(), offset_size * sizeof(uint16_t));
            target->idealoffset->offset_start = oidx;
            oidx += offset_size;

            uint edge_sequences_size = target->get_len_edge_sequences();
            memcpy(gctx->h_edge_sequences + eidx, target->get_edge_sequence(), edge_sequences_size * sizeof(EdgeSeq));
            target->idealoffset->edge_sequences_start = eidx;
            eidx += edge_sequences_size;

            uint vertices_size = target->get_num_vertices();
            memcpy(gctx->h_vertices + vidx, target->get_boundary()->p, vertices_size * sizeof(Point));
            target->idealoffset->vertices_start = vidx;
            vidx += vertices_size;

            uint gridline_offset_size = target->get_vertical()->get_num_grid_lines();
            memcpy(gctx->h_gridline_offset + goidx, target->get_vertical()->get_offset(), gridline_offset_size * sizeof(uint16_t));
            target->idealoffset->gridline_offset_start = goidx;
            goidx += gridline_offset_size;
            target->idealoffset->gridline_offset_end = goidx;

            uint gridline_nodes_size = target->get_vertical()->get_num_crosses();
            memcpy(gctx->h_gridline_nodes + gnidx, target->get_vertical()->get_intersection_nodes(), gridline_nodes_size * sizeof(double));
            target->idealoffset->gridline_nodes_start = gnidx;
            gnidx += gridline_nodes_size;

            if(gctx->use_hierachy){
                uint layer_info_size = target->get_num_layers() + 1;
                memcpy(gctx->h_layer_info + liidx, target->get_layer_info(), layer_info_size * sizeof(RasterInfo));
                target->idealoffset->layer_info_start = liidx;
                liidx += layer_info_size;

                uint layer_offset_size = target->get_num_layers() + 1;
                memcpy(gctx->h_layer_offset + loidx, target->get_layer_offset(), layer_offset_size * sizeof(uint16_t));
                target->idealoffset->layer_offset_start = loidx;
                loidx += layer_offset_size;
            }
        }
    }

    assert(flag1 ^ flag2);

    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_info, gctx->h_info, gctx->num_polygons * sizeof(RasterInfo), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_status, gctx->h_status, gctx->num_status * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_offset, gctx->h_offset, gctx->num_offset * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_edge_sequences, gctx->h_edge_sequences, gctx->num_edge_sequences * sizeof(EdgeSeq), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_vertices, gctx->h_vertices, gctx->num_vertices * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_gridline_offset, gctx->h_gridline_offset, gctx->num_gridline_offset * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_gridline_nodes, gctx->h_gridline_nodes, gctx->num_gridline_nodes * sizeof(double), cudaMemcpyHostToDevice));
    if(gctx->use_hierachy){
        CUDA_SAFE_CALL(cudaMemcpy(gctx->d_layer_info, gctx->h_layer_info, gctx->num_layer_info * sizeof(RasterInfo), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(gctx->d_layer_offset, gctx->h_layer_offset, gctx->num_layer_offset * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }

    double h_degree_per_kilometer_latitude = 360.0/40076.0;
    double h_degree_per_kilometer_longitude_arr[] = {
            0.008983,0.008983,0.008983,0.008983,0.008983,0.008983,0.008983,0.008984,0.008984,0.008984
            ,0.008984,0.008985,0.008985,0.008985,0.008986,0.008986,0.008986,0.008987,0.008987,0.008988
            ,0.008988,0.008989,0.008990,0.008990,0.008991,0.008991,0.008992,0.008993,0.008994,0.008994
            ,0.008995,0.008996,0.008997,0.008998,0.008999,0.009000,0.009001,0.009002,0.009003,0.009004
            ,0.009005,0.009006,0.009007,0.009008,0.009009,0.009011,0.009012,0.009013,0.009015,0.009016
            ,0.009017,0.009019,0.009020,0.009022,0.009023,0.009024,0.009026,0.009028,0.009029,0.009031
            ,0.009032,0.009034,0.009036,0.009038,0.009039,0.009041,0.009043,0.009045,0.009047,0.009048
            ,0.009050,0.009052,0.009054,0.009056,0.009058,0.009060,0.009063,0.009065,0.009067,0.009069
            ,0.009071,0.009073,0.009076,0.009078,0.009080,0.009083,0.009085,0.009087,0.009090,0.009092
            ,0.009095,0.009097,0.009100,0.009103,0.009105,0.009108,0.009111,0.009113,0.009116,0.009119
            ,0.009122,0.009124,0.009127,0.009130,0.009133,0.009136,0.009139,0.009142,0.009145,0.009148
            ,0.009151,0.009154,0.009157,0.009161,0.009164,0.009167,0.009170,0.009174,0.009177,0.009180
            ,0.009184,0.009187,0.009190,0.009194,0.009197,0.009201,0.009205,0.009208,0.009212,0.009216
            ,0.009219,0.009223,0.009227,0.009231,0.009234,0.009238,0.009242,0.009246,0.009250,0.009254
            ,0.009258,0.009262,0.009266,0.009270,0.009274,0.009278,0.009283,0.009287,0.009291,0.009295
            ,0.009300,0.009304,0.009309,0.009313,0.009317,0.009322,0.009326,0.009331,0.009336,0.009340
            ,0.009345,0.009350,0.009354,0.009359,0.009364,0.009369,0.009374,0.009378,0.009383,0.009388
            ,0.009393,0.009398,0.009403,0.009409,0.009414,0.009419,0.009424,0.009429,0.009435,0.009440
            ,0.009445,0.009451,0.009456,0.009461,0.009467,0.009472,0.009478,0.009484,0.009489,0.009495
            ,0.009501,0.009506,0.009512,0.009518,0.009524,0.009530,0.009535,0.009541,0.009547,0.009553
            ,0.009559,0.009566,0.009572,0.009578,0.009584,0.009590,0.009597,0.009603,0.009609,0.009616
            ,0.009622,0.009628,0.009635,0.009642,0.009648,0.009655,0.009661,0.009668,0.009675,0.009682
            ,0.009688,0.009695,0.009702,0.009709,0.009716,0.009723,0.009730,0.009737,0.009744,0.009751
            ,0.009759,0.009766,0.009773,0.009781,0.009788,0.009795,0.009803,0.009810,0.009818,0.009825
            ,0.009833,0.009841,0.009848,0.009856,0.009864,0.009872,0.009880,0.009888,0.009896,0.009904
            ,0.009912,0.009920,0.009928,0.009936,0.009944,0.009952,0.009961,0.009969,0.009978,0.009986
            ,0.009994,0.010003,0.010012,0.010020,0.010029,0.010038,0.010046,0.010055,0.010064,0.010073
            ,0.010082,0.010091,0.010100,0.010109,0.010118,0.010127,0.010136,0.010146,0.010155,0.010164
            ,0.010174,0.010183,0.010193,0.010202,0.010212,0.010222,0.010231,0.010241,0.010251,0.010261
            ,0.010271,0.010281,0.010291,0.010301,0.010311,0.010321,0.010331,0.010341,0.010352,0.010362
            ,0.010373,0.010383,0.010394,0.010404,0.010415,0.010426,0.010436,0.010447,0.010458,0.010469
            ,0.010480,0.010491,0.010502,0.010513,0.010524,0.010535,0.010547,0.010558,0.010569,0.010581
            ,0.010592,0.010604,0.010616,0.010627,0.010639,0.010651,0.010663,0.010675,0.010687,0.010699
            ,0.010711,0.010723,0.010735,0.010748,0.010760,0.010772,0.010785,0.010797,0.010810,0.010823
            ,0.010835,0.010848,0.010861,0.010874,0.010887,0.010900,0.010913,0.010926,0.010939,0.010953
            ,0.010966,0.010980,0.010993,0.011007,0.011020,0.011034,0.011048,0.011062,0.011075,0.011089
            ,0.011104,0.011118,0.011132,0.011146,0.011160,0.011175,0.011189,0.011204,0.011218,0.011233
            ,0.011248,0.011263,0.011278,0.011293,0.011308,0.011323,0.011338,0.011353,0.011369,0.011384
            ,0.011400,0.011415,0.011431,0.011446,0.011462,0.011478,0.011494,0.011510,0.011526,0.011543
            ,0.011559,0.011575,0.011592,0.011608,0.011625,0.011642,0.011658,0.011675,0.011692,0.011709
            ,0.011726,0.011744,0.011761,0.011778,0.011796,0.011813,0.011831,0.011849,0.011867,0.011884
            ,0.011903,0.011921,0.011939,0.011957,0.011975,0.011994,0.012013,0.012031,0.012050,0.012069
            ,0.012088,0.012107,0.012126,0.012145,0.012164,0.012184,0.012203,0.012223,0.012243,0.012263
            ,0.012283,0.012303,0.012323,0.012343,0.012363,0.012384,0.012404,0.012425,0.012446,0.012467
            ,0.012488,0.012509,0.012530,0.012551,0.012573,0.012594,0.012616,0.012638,0.012660,0.012682
            ,0.012704,0.012726,0.012748,0.012771,0.012793,0.012816,0.012839,0.012862,0.012885,0.012908
            ,0.012931,0.012955,0.012978,0.013002,0.013026,0.013050,0.013074,0.013098,0.013122,0.013147
            ,0.013171,0.013196,0.013221,0.013246,0.013271,0.013296,0.013322,0.013347,0.013373,0.013399
            ,0.013425,0.013451,0.013477,0.013503,0.013530,0.013557,0.013584,0.013610,0.013638,0.013665
            ,0.013692,0.013720,0.013748,0.013775,0.013803,0.013832,0.013860,0.013888,0.013917,0.013946
            ,0.013975,0.014004,0.014033,0.014063,0.014093,0.014122,0.014152,0.014183,0.014213,0.014243
            ,0.014274,0.014305,0.014336,0.014367,0.014399,0.014430,0.014462,0.014494,0.014526,0.014558
            ,0.014591,0.014623,0.014656,0.014689,0.014723,0.014756,0.014790,0.014824,0.014858,0.014892
            ,0.014926,0.014961,0.014996,0.015031,0.015066,0.015102,0.015138,0.015174,0.015210,0.015246
            ,0.015283,0.015320,0.015357,0.015394,0.015431,0.015469,0.015507,0.015545,0.015584,0.015622
            ,0.015661,0.015700,0.015740,0.015779,0.015819,0.015860,0.015900,0.015941,0.015981,0.016023
            ,0.016064,0.016106,0.016148,0.016190,0.016233,0.016275,0.016318,0.016362,0.016405,0.016449
            ,0.016493,0.016538,0.016583,0.016628,0.016673,0.016719,0.016765,0.016811,0.016857,0.016904
            ,0.016952,0.016999,0.017047,0.017095,0.017143,0.017192,0.017241,0.017291,0.017341,0.017391
            ,0.017441,0.017492,0.017543,0.017595,0.017647,0.017699,0.017752,0.017805,0.017858,0.017912
            ,0.017966,0.018020,0.018075,0.018131,0.018186,0.018242,0.018299,0.018356,0.018413,0.018471
            ,0.018529,0.018587,0.018646,0.018706,0.018766,0.018826,0.018887,0.018948,0.019009,0.019072
            ,0.019134,0.019197,0.019261,0.019325,0.019389,0.019454,0.019520,0.019586,0.019652,0.019719
            ,0.019787,0.019855,0.019923,0.019992,0.020062,0.020132,0.020203,0.020274,0.020346,0.020419
            ,0.020492,0.020565,0.020639,0.020714,0.020790,0.020866,0.020942,0.021020,0.021098,0.021176
            ,0.021255,0.021335,0.021416,0.021497,0.021579,0.021662,0.021745,0.021829,0.021914,0.021999
            ,0.022085,0.022172,0.022260,0.022349,0.022438,0.022528,0.022619,0.022710,0.022803,0.022896
            ,0.022990,0.023085,0.023181,0.023278,0.023375,0.023474,0.023573,0.023673,0.023774,0.023877
            ,0.023980,0.024084,0.024189,0.024295,0.024402,0.024510,0.024619,0.024729,0.024840,0.024953
            ,0.025066,0.025181,0.025296,0.025413,0.025531,0.025650,0.025771,0.025892,0.026015,0.026139
            ,0.026264,0.026391,0.026519,0.026648,0.026779,0.026911,0.027044,0.027179,0.027315,0.027452
            ,0.027592,0.027732,0.027874,0.028018,0.028163,0.028310,0.028459,0.028609,0.028761,0.028914
            ,0.029069,0.029226,0.029385,0.029546,0.029708,0.029873,0.030039,0.030207,0.030378,0.030550
            ,0.030724,0.030901,0.031079,0.031260,0.031443,0.031628,0.031816,0.032006,0.032198,0.032393
            ,0.032590,0.032789,0.032991,0.033196,0.033404,0.033614,0.033827,0.034043,0.034261,0.034483
            ,0.034707,0.034935,0.035166,0.035400,0.035637,0.035877,0.036121,0.036368,0.036619,0.036873
            ,0.037132,0.037393,0.037659,0.037929,0.038202,0.038480,0.038762,0.039048,0.039338,0.039633
            ,0.039933,0.040237,0.040546,0.040860,0.041179,0.041503,0.041833,0.042167,0.042508,0.042854
            ,0.043206,0.043563,0.043927,0.044297,0.044674,0.045057,0.045447,0.045844,0.046248,0.046659
            ,0.047078,0.047505,0.047939,0.048382,0.048833,0.049293,0.049762,0.050239,0.050727,0.051224
            ,0.051731,0.052248,0.052776,0.053315,0.053865,0.054426,0.055000,0.055586,0.056185,0.056797
            ,0.057423,0.058063,0.058717,0.059387,0.060072,0.060774,0.061492,0.062228,0.062981,0.063753
            ,0.064545,0.065357,0.066189,0.067044,0.067921,0.068821,0.069746,0.070696,0.071672,0.072677
            ,0.073710,0.074773,0.075867,0.076994,0.078155,0.079352,0.080587,0.081861,0.083176,0.084534
            ,0.085938,0.087389,0.088890,0.090445,0.092054,0.093723,0.095453,0.097249,0.099114,0.101052
            ,0.103068,0.105166,0.107351,0.109630,0.112008,0.114492,0.117089,0.119806,0.122654,0.125640
            ,0.128776,0.132072,0.135543,0.139201,0.143062,0.147144,0.151467,0.156051,0.160922,0.166108
            ,0.171640,0.177553,0.183889,0.190694,0.198023,0.205939,0.214514,0.223836,0.234005,0.245143
            ,0.257394,0.270936,0.285983,0.302800,0.321719,0.343162,0.367668,0.395945,0.428935,0.467923
            ,0.514710,0.571895,0.643376,0.735281,0.857823,1.029381,1.286721,1.715622,2.573426,5.146844
    };

    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->d_degree_degree_per_kilometer_latitude, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->d_degree_degree_per_kilometer_latitude, &h_degree_per_kilometer_latitude, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&gctx->degree_per_kilometer_longitude_arr, sizeof(h_degree_per_kilometer_longitude_arr)));
    CUDA_SAFE_CALL(cudaMemcpy(gctx->degree_per_kilometer_longitude_arr, h_degree_per_kilometer_longitude_arr, sizeof(degree_per_kilometer_longitude_arr), cudaMemcpyHostToDevice));
}