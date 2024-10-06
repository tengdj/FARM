#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv) {
    query_context global_ctx;
	global_ctx.num_threads = 1;
	Point* source = load_point_wkt("/home/qmh/data/test_point.csv", global_ctx.target_num, &global_ctx);

    // for(int i = 0; i < global_ctx.target_num; i ++){
    //     Point p = source[i];
    //     p.print();
    // }

    dump_to_file("/home/qmh/data/test_point.dat", (char*)source, global_ctx.target_num * sizeof(Point));


    return 0;
}