#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char **argv)
{
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.use_ideal = true;
	global_ctx.use_hierachy = true;
	global_ctx.num_threads = 128;
	global_ctx.NLow = 20;
	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);

	preprocess(&global_ctx);

	printf("num_layers = %d\n", global_ctx.max_layers);

	for (auto ideal : global_ctx.source_ideals)
	{
		printf("dimx = %d, dimy = %d\n", ideal->get_dimx(), ideal->get_dimy());
		printf("step_x = %lf, step_y = %lf\n", ideal->get_step_x(), ideal->get_step_y());

		ideal->getMBB()->print();
		ideal->MyPolygon::print();
		ideal->MyRaster::print();

		puts("-------------------------HIERARCHY---------------------------------");
		for (int i = 0; i <= ideal->get_num_layers(); i++)
		{
			printf("level %d:\n", i);
			printf("dimx=%d, dimy=%d\n", ideal->get_layers()[i].get_dimx(), ideal->get_layers()[i].get_dimy());
			ideal->get_layers()[i].mbr->print();
			ideal->get_layers()[i].print();
		}
		// int _dimx = ideal->get_dimx(), _dimy = ideal->get_dimy();
		// for(int i = 0; i < ideal->get_num_layers(); i ++){
		// 	_dimx = ideal->get_layer_info()[i].dimx;
		// 	_dimy = ideal->get_layer_info()[i].dimy;
		// 	for(int j = 0; j < (_dimx+1) * (_dimy+1); j ++){
		// 		if(j % (_dimx+1) == 0) cout << endl;
		// 		cout << ideal->show_status(ideal->get_layer_offset()[i]+j) << " ";
		// 	}
		// }
	}

	// cout << "Hierarchy Grid" << endl;

	// vector<Hraster*> layers = source->get_layers();

	// for(int i = 0; i < layers.size(); i ++){
	// 	cout << "Level " << i << endl;
	// 	layers[i]->print();
	// }

	// for(int i = 0; i < source.size(); i ++){
	// 	auto p = source[i];
	// 	p->rasterization(100);
	// 	p->print();
	// 	p->get_rastor()->print();
	// }
	cout << "rasterization finished!" << endl;

	return 0;
}