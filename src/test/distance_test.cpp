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

    // global_ctx.source_ideals.resize(10);
    // global_ctx.target_ideals.resize(10);

    preprocess(&global_ctx);

    // query
    int found = 0;
    printf("%d %d\n", global_ctx.source_ideals.size(), global_ctx.target_ideals.size());
    assert(global_ctx.source_ideals.size() == global_ctx.target_ideals.size());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < global_ctx.source_ideals.size(); i++)
    {
        Ideal *ideal = global_ctx.source_ideals[i];
        Ideal *target = global_ctx.target_ideals[i];


        // printf("dimx = %d, dimy = %d, stepx = %lf, stepy = %lf\n", ideal->get_dimx(), ideal->get_dimy(), ideal->get_step_x(), ideal->get_step_y());
        // ideal->MyRaster::print();
        // printf("dimx = %d, dimy = %d, stepx = %lf, stepy = %lf\n", target->get_dimx(), target->get_dimy(), target->get_step_x(), target->get_step_y());
        // target->MyRaster::print();

        // ideal->MyPolygon::print();
        // ideal->getMBB()->print();

		// for(int i = 0; i <= ideal->get_num_layers(); i ++){
		// 	printf("level %d:\n", i);
		// 	printf("dimx=%d, dimy=%d\n", ideal->get_layers()[i].get_dimx(), ideal->get_layers()[i].get_dimy());
		// 	printf("step_x=%lf, step_y=%lf\n", ideal->get_layers()[i].get_step_x(), ideal->get_layers()[i].get_step_y());
		// 	// ideal->get_layers()[i].mbr->print();
		// 	ideal->get_layers()[i].print();
		// }

        // target->MyPolygon::print();
        // target->getMBB()->print();

        // for(int i = 0; i <= target->get_num_layers(); i ++){
		// 	printf("level %d:\n", i);
		// 	printf("dimx=%d, dimy=%d\n", target->get_layers()[i].get_dimx(), target->get_layers()[i].get_dimy());
		// 	// ideal->get_layers()[i].mbr->print();
		// 	target->get_layers()[i].print();
		// }

        if (ideal == target)
        {
            continue;
        }

        // the minimum possible distance is larger than the threshold
        if (ideal->getMBB()->distance(*target->getMBB(), global_ctx.geography) > global_ctx.within_distance)
        {
            continue;
        }
        // the maximum possible distance is smaller than the threshold
        if (ideal->getMBB()->max_distance(*target->getMBB(), global_ctx.geography) <= global_ctx.within_distance)
        {
            found++;
            continue;
        }

        if (ideal->getMBB()->contain(*target->getMBB()) && ideal->contain(target->get_boundary()->p[0], &global_ctx))
        {
            found++;
            continue;
        }
        if (target->getMBB()->contain(*ideal->getMBB()) && target->contain(ideal->get_boundary()->p[0], &global_ctx))
        {
            found++;
            continue;
        }

        global_ctx.polygon_pairs.push_back(make_pair(ideal, target));



        // double reference_dist = 100000.0;
        // for (int i = 0; i < ideal->get_num_vertices() - 1; i++)
        // {
        //     Point p1 = ideal->get_boundary()->p[i];
        //     Point p2 = ideal->get_boundary()->p[i + 1];
        //     for (int j = 0; j < target->get_num_vertices() - 1; j++)
        //     {
        //         Point p3 = target->get_boundary()->p[j];
        //         Point p4 = target->get_boundary()->p[j + 1];
        //         reference_dist = min(reference_dist, segment_to_segment_distance(p1, p2, p3, p4, false));
        //     }
        // }
        // printf("reference distance = %lf\n", reference_dist);
    }
    preprocess_for_gpu(&global_ctx);

    found += cuda_within_polygon(&global_ctx);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    log("total time: %lfms\n", duration.count());
    log("TOTAL: %ld, WITHIN: %d", global_ctx.source_ideals.size(), found);
    // printf("Avarge Runtime: %lfs\n", time.count() / (global_ctx.source_ideals.size() - found));
    return 0;
}
