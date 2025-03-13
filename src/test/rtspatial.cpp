/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#include "../index/RTree.h"
#include <queue>
#include <fstream>
#include "../include/Ideal.h"
#include <chrono>

// some shared parameters

RTree<Ideal *, double, 2, double> ideal_rtree;

bool IdealSearchCallback(Ideal *ideal, void *arg)
{
    query_context *ctx = (query_context *)arg;
    ctx->found ++;
    return true;
}

void *query(void *args)
{
    query_context *ctx = (query_context *)args;
    query_context *gctx = ctx->global_ctx;
    log("thread %d is started", ctx->thread_id);
    ctx->query_count = 0;
    // ctx->point_polygon_pairs = new pair<Point*, Ideal*>[gctx->target_num];

    while (ctx->next_batch(100))
    {
        for (int i = ctx->index; i < ctx->index_end; i++)
        {
            ctx->target = (void *)&gctx->points[i];
            ideal_rtree.Search((double *)(gctx->points + i), (double *)(gctx->points + i), IdealSearchCallback, (void *)ctx);
            ctx->report_progress();
        }
    }

    ctx->merge_global();

    return NULL;
}

int main(int argc, char **argv)
{

    query_context global_ctx;
    global_ctx = get_parameters(argc, argv);
    global_ctx.query_type = QueryType::contain;

    global_ctx.source_ideals = load_polygon_wkt(global_ctx.source_path.c_str());
    // read all the points
    global_ctx.points = load_point_wkt(global_ctx.target_path.c_str(), global_ctx.target_num, &global_ctx);

    timeval start = get_cur_time();
    for (auto p : global_ctx.source_ideals)
    {
        ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
    }
    logt("building R-Tree with %d nodes", start, global_ctx.source_ideals.size());

    auto total_runtime_start = std::chrono::high_resolution_clock::now();
    pthread_t threads[global_ctx.num_threads];
    query_context ctx[global_ctx.num_threads];
    for (int i = 0; i < global_ctx.num_threads; i++)
    {
        ctx[i] = global_ctx;
        ctx[i].thread_id = i;
        ctx[i].global_ctx = &global_ctx;
    }
    for (int i = 0; i < global_ctx.num_threads; i++)
    {
        pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
    }

    for (int i = 0; i < global_ctx.num_threads; i++)
    {
        void *status;
        pthread_join(threads[i], &status);
    }

    auto total_runtime_end = std::chrono::high_resolution_clock::now();
    auto total_runtime_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_runtime_end - total_runtime_start);
    std::cout << "rtree query: " << total_runtime_duration.count() << " ms" << std::endl;
    std::cout << "found: " << global_ctx.found << endl;

    return 0;
}