#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char** argv) {
    query_context global_ctx;
	global_ctx.num_threads = 1;
	vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/cpu/IDEAL/src/source.txt");
	// vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/data/simple.txt");

    // for(auto poly : polygons){
    //     poly->MyPolygon::print();
    // }

    dump_polygons_to_file(polygons, "/home/qmh/data/simple_source.idl");
    return 0;
}