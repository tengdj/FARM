/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */

#include "../include/Ideal.h"
#include <fstream>
#include <queue>
#include <boost/program_options.hpp>
#include "UniversalGrid.h"

using namespace std;

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within_polygon;
	global_ctx.num_threads = 1;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(), global_ctx);

	global_ctx.num_threads = 128;

	// the target is also the source
	global_ctx.target_num = global_ctx.target_ideals.size();

	if(!global_ctx.batch_size) global_ctx.batch_size = global_ctx.target_num;

	indexBuild(&global_ctx);

	auto rtree_query_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < global_ctx.target_num; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.target_num);
		indexQuery(&global_ctx);
	}
	auto rtree_query_end = std::chrono::high_resolution_clock::now();
	auto rtree_query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_query_end - rtree_query_start);
	std::cout << "rtree query: " << rtree_query_duration.count() << " ms" << std::endl;
	indexDestroy(&global_ctx);
	
	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

	auto preprocess_gpu_start = std::chrono::high_resolution_clock::now();
	preprocess_for_gpu(&global_ctx);
	auto preprocess_gpu_end = std::chrono::high_resolution_clock::now();
	auto preprocess_gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_gpu_end - preprocess_gpu_start);
	std::cout << "preprocess for gpu time: " << preprocess_gpu_duration.count() << " ms" << std::endl;

	auto gpu_start = std::chrono::high_resolution_clock::now();
	// printf("%lf %lf\n", UniversalGrid::getInstance().get_step_x(), UniversalGrid::getInstance().get_step_y());
	// for(auto p : global_ctx.source_ideals){
	// 	if(p->id == 12787){
	// 		printf("id = %d\n", p->id);
	// 		printf("\ndimx = %d, dimy = %d\n", p->get_dimx(), p->get_dimy());
	// 		p->MyPolygon::print();
	// 		p->MyRaster::print();
	// 		for(int i = 0; i < p->get_num_pixels(); i ++){
	// 			if(i % p->get_dimx() == 0) printf("\n");
	// 			printf("%d ", p->get_offset(i));
	// 		}
	// 	}
	// }

	// return 0;

	// int a = global_ctx.h_candidate_pairs[0].first;
	// int b = global_ctx.h_candidate_pairs[0].second - global_ctx.source_ideals.size();
	// printf("polygon id = %d\n", global_ctx.source_ideals[a]->id);
	// global_ctx.source_ideals[a]->MyPolygon::print();
	// global_ctx.source_ideals[a]->MyRaster::print();
	
	// for (int i = 0; i <= global_ctx.source_ideals[a]->get_num_layers(); i++)
	// {
	// 	printf("level %d:\n", i);
	// 	printf("dimx=%d, dimy=%d, step_x = %lf, step_y = %lf\n", global_ctx.source_ideals[a]->get_layers()[i].get_dimx(), global_ctx.source_ideals[a]->get_layers()[i].get_dimy(), global_ctx.source_ideals[a]->get_layers()[i].get_step_x(), global_ctx.source_ideals[a]->get_layers()[i].get_step_y());
	// 	global_ctx.source_ideals[a]->get_layers()[i].mbr->print();
	// 	global_ctx.source_ideals[a]->get_layers()[i].print();
	// }
	
	// printf("polygon id = %d\n", global_ctx.target_ideals[b]->id);
	// global_ctx.target_ideals[b]->MyPolygon::print();
	// global_ctx.target_ideals[b]->MyRaster::print();

	// for (int i = 0; i <= global_ctx.target_ideals[b]->get_num_layers(); i++)
	// {
	// 	printf("level %d:\n", i);
	// 	printf("dimx=%d, dimy=%d, step_x = %lf, step_y = %lf\n", global_ctx.target_ideals[b]->get_layers()[i].get_dimx(), global_ctx.target_ideals[b]->get_layers()[i].get_dimy(), global_ctx.target_ideals[b]->get_layers()[i].get_step_x(), global_ctx.target_ideals[b]->get_layers()[i].get_step_y());
	// 	global_ctx.target_ideals[b]->get_layers()[i].mbr->print();
	// 	global_ctx.target_ideals[b]->get_layers()[i].print();
	// }
	// global_ctx.batch_size = 1;
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);

		ResetDevice(&global_ctx);

		auto batch_start = std::chrono::high_resolution_clock::now();
		cuda_within_polygon(&global_ctx);
		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
		// return 0;
	}
	auto gpu_end = std::chrono::high_resolution_clock::now();
	auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
	std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

	cout << endl;
	printf("Found: %d\n", global_ctx.found);
	return 0;
}



