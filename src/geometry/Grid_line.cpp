#include "../include/Ideal.h"

Grid_line::Grid_line(int size){
    num_grid_lines = size + 2;
    offset = new uint32_t[size + 2];
    memset(offset, 0, sizeof(uint32_t) * (size+2));
}

Grid_line::~Grid_line(){
    if(offset) delete []offset;
    if(intersection_nodes) delete []intersection_nodes;
}

void Grid_line::init_intersection_node(int num_nodes){
    intersection_nodes = new double[num_nodes];
}