#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char **argv)
{
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.geography = false;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	// global_ctx.source_ideals = load_polygon_wkt(global_ctx.source_path.c_str());
	
	// preprocess(&global_ctx);

	float x_min = 0;
	float x_max = 110540;
	float y_min = 0;
	float y_max = 48690;
	
	for(auto p : global_ctx.source_ideals){
		// for(int i = 0; i < p->get_num_vertices(); i ++){
		// 	p->get_boundary()->p[i].x = (p->get_boundary()->p[i].x / x_max) * 240 - 120.0;
		// 	p->get_boundary()->p[i].y = (p->get_boundary()->p[i].y / y_max) * 120 - 60.0;
		// }
		// p->MyPolygon::print();
		// if(p->get_num_vertices() > 500)
			p->MyPolygon::print();
	}



	// preprocess(&global_ctx);

	// int i = 0;
	// for (auto p : global_ctx.source_ideals)
	// {

	// 	printf("id = %d\n", i++);
	// 	printf("dimx = %d dimy = %d step_x = %lf step_y = %lf\n", p->get_dimx(), p->get_dimy(), p->get_step_x(), p->get_step_y());
	// 	p->getMBB()->print();
	// 	p->MyPolygon::print();
	// 	p->MyRaster::print();
	// }

	return 0;
}