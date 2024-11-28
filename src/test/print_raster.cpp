#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);

	for(auto item : global_ctx.source_ideals){
		cout << item->get_num_vertices() << endl;;
	}

	// preprocess(&global_ctx);
	// cout << "rasterization finished!" << endl;

	// // read all the points
	// global_ctx.load_points();


	return 0;
}