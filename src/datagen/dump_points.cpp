#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv) {
    query_context global_ctx;
	global_ctx.num_threads = 1;
	// Point* source = load_point_wkt("/home/qmh/data/test_point.csv", global_ctx.target_num, &global_ctx);
    vector<Ideal *> polygons = load_polygon_wkt("/home/qmh/data/rayjoin/parks/Asia/parks_Asia_Point.csv");
    unsigned long long size = 0;
    for(auto polygon : polygons){
        size += polygon->get_boundary()->num_vertices; 
    }
    Point* source = new Point[size];

    int idx = 0;
    for(auto polygon : polygons){
        for(int i = 0; i < polygon->get_boundary()->num_vertices; i ++){
            source[idx ++] = polygon->get_boundary()->p[i];
        }
    }

    // for(int i = 0; i < global_ctx.target_num; i ++){
    //     Point p = source[i];
    //     p.print();
    // }

    dump_to_file("/home/qmh/data/rayjoin/parks_Asia_Point.dat", (char*)source, size * sizeof(Point));
    // dump_to_file("/home/qmh/data/rayjoin/USAZIPCodeArea_Point.dat", (char*)source, global_ctx.target_num * sizeof(Point));


    return 0;
}