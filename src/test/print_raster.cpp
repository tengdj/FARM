#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	// global_ctx.num_threads = 1;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);

	unsigned long long sum = 0;
	int size = global_ctx.source_ideals.size();
	for(auto p : global_ctx.source_ideals){
		sum += p->get_num_vertices();
		// if(p->get_num_vertices() > 1000){
			// p->MyPolygon::print();
		// }
	}
	printf("%lu\n", sum);
	// preprocess(&global_ctx);
	// cout << "rasterization finished!" << endl;

	// // read all the points
	// global_ctx.load_points();

	return 0;
}