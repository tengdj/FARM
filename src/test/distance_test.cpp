#include "../include/Ideal.h"
#include <fstream>
#include "../index/RTree.h"
#include <queue>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

RTree<Ideal *, double, 2, double> ideal_rtree;
RTree<MyPolygon *, double, 2, double> poly_rtree;

int main(int argc, char **argv)
{
    query_context global_ctx;
    global_ctx = get_parameters(argc, argv);
    global_ctx.query_type = QueryType::within;
    global_ctx.geography = true;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
    global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(), global_ctx);


    // query
    int found = 0;
    printf("%d %d\n", global_ctx.source_ideals.size(), global_ctx.target_ideals.size());
    assert(global_ctx.source_ideals.size() == global_ctx.target_ideals.size());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < global_ctx.source_ideals.size(); i++)
    {
        Ideal *ideal = global_ctx.source_ideals[i];
        for(int j = 0; j < global_ctx.target_ideals.size(); j ++){
            Ideal *target = global_ctx.target_ideals[j];

            double reference_dist = 100000.0;
            for (int _i = 0; _i < ideal->get_num_vertices() - 1; _i ++){
                Point p1 = ideal->get_boundary()->p[_i];
                Point p2 = ideal->get_boundary()->p[_i + 1];
                for(int _j = 0; _j < target->get_num_vertices() - 1; _j ++){
                    Point p3 = target->get_boundary()->p[_j];
                    Point p4 = target->get_boundary()->p[_j + 1];
                    reference_dist = min(reference_dist, segment_to_segment_distance(p1, p2, p3, p4, true));
                }
            }
            printf("reference distance = %lf\n", reference_dist);
            if(reference_dist <= global_ctx.within_distance) found ++;
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    log("total time: %lfms\n", duration.count());
    log("TOTAL: %ld, WITHIN: %d", global_ctx.source_ideals.size(), found);
    // printf("Avarge Runtime: %lfs\n", time.count() / (global_ctx.source_ideals.size() - found));
    return 0;
}

/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

 #include "../include/Ideal.h"
 #include <fstream>
 #include "../index/RTree.h"
 #include <queue>
 #include <boost/program_options.hpp>
 
 namespace po = boost::program_options;
 using namespace std;
 
 RTree<Ideal *, double, 2, double> ideal_rtree;
 RTree<MyPolygon *, double, 2, double> poly_rtree;
 
 bool MySearchCallback(Ideal *ideal, void* arg){
     query_context *ctx = (query_context *)arg;
 
     Ideal *target= (Ideal *)ctx->target;
 
     if(ideal == target){
         return true;
     }
     // the minimum possible distance is larger than the threshold
     if(ideal->getMBB()->distance(*target->getMBB(), ctx->geography)>ctx->within_distance){
         return true;
     }
     // the maximum possible distance is smaller than the threshold
     if(ideal->getMBB()->max_distance(*target->getMBB(), ctx->geography)<=ctx->within_distance){
         ctx->found++;
         return true;
     }
 
     ctx->distance = ideal->distance(target,ctx);
     ctx->found += ctx->distance <= ctx->within_distance;
 
     return true;
 }

 
 void *query(void *args){
     query_context *ctx = (query_context *)args;
     query_context *gctx = ctx->global_ctx;
     log("thread %d is started",ctx->thread_id);
     ctx->query_count = 0;
     double buffer_low[2];
     double buffer_high[2];
 
     while(ctx->next_batch(100)){
         for(int i=ctx->index;i<ctx->index_end;i++){
            
            ctx->target = (void *)(gctx->source_ideals[i]);
            box qb = gctx->source_ideals[i]->getMBB()->expand(gctx->within_distance, ctx->geography);
            ideal_rtree.Search(qb.low, qb.high, MySearchCallback, (void *)ctx);
             
             ctx->report_progress();
         }
     }
     ctx->merge_global();
     return NULL;
 }
 
 
 
 int main(int argc, char** argv) {
     query_context global_ctx;
     global_ctx = get_parameters(argc, argv);
     global_ctx.query_type = QueryType::within;
     global_ctx.geography = true;
 

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
    timeval start = get_cur_time();
    for(Ideal *p : global_ctx.source_ideals){
        ideal_rtree.Insert(p->getMBB()->low, p->getMBB()->high, p);
    }
    logt("building R-Tree with %d nodes", start, global_ctx.source_ideals.size());
    // the target is also the source
    global_ctx.target_num = global_ctx.source_ideals.size();
 
     timeval start = get_cur_time();
     pthread_t threads[global_ctx.num_threads];
     query_context ctx[global_ctx.num_threads];
     for(int i=0;i<global_ctx.num_threads;i++){
         ctx[i] = global_ctx; //query_context(global_ctx);
         ctx[i].thread_id = i;
         ctx[i].global_ctx = &global_ctx;
     }
 
     for(int i=0;i<global_ctx.num_threads;i++){
         pthread_create(&threads[i], NULL, query, (void *)&ctx[i]);
     }
 
     for(int i = 0; i < global_ctx.num_threads; i++ ){
         void *status;
         pthread_join(threads[i], &status);
     }
     cout << endl;
     log("TOTAL: %ld, WITHIN: %d", global_ctx.source_ideals.size(), found);
 
     return 0;
 }
 
 
 