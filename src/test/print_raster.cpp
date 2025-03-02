#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	// global_ctx.num_threads = 1;

    global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);

	vector<MyPolygon*> polygons;
	for (size_t i = 20; i < 25; i++)
	{
		bool flag = false;;
		for(auto p : global_ctx.source_ideals){
			if(p->get_num_vertices() > i * 1500 && p->get_num_vertices() < (i + 1) * 1500){
				polygons.push_back(p);
				flag = true;
				break;
			}
		}
		// assert(!flag);
	}

	for(auto p : polygons){
		cout << p->get_num_vertices() << endl;
		p->MyPolygon::print();
	}

	// preprocess(&global_ctx);
	// cout << "rasterization finished!" << endl;

	// // read all the points
	// global_ctx.load_points();

	return 0;
}