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

    // global_ctx.source_ideals.resize(1);

    preprocess(&global_ctx);

    global_ctx.load_points();

    // global_ctx.target_num = 1;

    // query
    int found = 0;
    printf("%d %d\n", global_ctx.source_ideals.size(), global_ctx.target_num);
    assert(global_ctx.source_ideals.size() == global_ctx.target_num);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < global_ctx.source_ideals.size(); i++)
    {
        Ideal *ideal = global_ctx.source_ideals[i];
        Point *p = global_ctx.points + i;

		// for(int i = 0; i <= ideal->get_num_layers(); i ++){
		// 	printf("level %d:\n", i);
		// 	printf("dimx=%d, dimy=%d\n", ideal->get_layers()[i].get_dimx(), ideal->get_layers()[i].get_dimy());
		// 	printf("step_x=%lf, step_y=%lf\n", ideal->get_layers()[i].get_step_x(), ideal->get_layers()[i].get_step_y());
		// 	// ideal->get_layers()[i].mbr->print();
		// 	ideal->get_layers()[i].print();
		// }

        if(ideal->getMBB()->distance(*p, true) > global_ctx.within_distance){
            continue;
        }

        if(ideal->contain(*p, &global_ctx)){
            found ++;
            continue;
        }
        printf("begin\n");
        ideal->MyPolygon::print();
        p->print();
        printf("end\n");
        global_ctx.point_polygon_pairs.push_back(make_pair(p, ideal));
    }

    preprocess_for_gpu(&global_ctx);

    found += cuda_within(&global_ctx);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("total time: %lfms\n", duration.count());
    printf("TOTAL: %ld, WITHIN: %d\n", global_ctx.source_ideals.size(), found);
    // printf("Avarge Runtime: %lfs\n", time.count() / (global_ctx.source_ideals.size() - found));
    return 0;
}
