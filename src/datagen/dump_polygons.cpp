#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char** argv) {
    query_context global_ctx;
	global_ctx.num_threads = 1;
	vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/IDEAL/src/inputA.wkt");
	// vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/data/simple.txt");

    // for(auto poly : polygons){
    //     poly->MyPolygon::print();
    // }

    dump_polygons_to_file(polygons, "/home/qmh/IDEAL/src/inputA.idl");
    return 0;
}