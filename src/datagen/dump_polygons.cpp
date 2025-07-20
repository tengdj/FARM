#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char** argv) {
    query_context global_ctx;
	global_ctx.num_threads = 1;
	// vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/IDEAL/src/input.txt");
	vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/data/wkt/complex_normal.wkt");

    // for(auto poly : polygons){
    //     poly->MyPolygon::print();
    // }

    // dump_polygons_to_file(polygons, "/home/qmh/IDEAL/src/random.idl");
    dump_polygons_to_file(polygons, "/home/qmh/data/exp/complex_normal.idl");
    return 0;
}