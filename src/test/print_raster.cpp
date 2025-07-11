#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char **argv)
{
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.geography = false;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	
	preprocess(&global_ctx);
	
	for(auto p : global_ctx.source_ideals){
		p->MyPolygon::print();
		p->MyRaster::print();
		for(int i = 0; i < p->get_num_pixels(); i ++){
			int sum = 0;
			for(int j = 0; j < p->get_num_sequences(i); j ++){
				auto edge_sequence = p->get_edge_sequence()[p->get_offset(i) + j];
				sum += edge_sequence.second;
			}
			printf("%d ", sum);
			if(i % p->get_dimx() == p->get_dimx() - 1){
				printf("\n");
			}
		}
		return 0;
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