#pragma once

#include "Ideal.h"
#include "geometry.cuh"

__global__ void comparePolygons(pair<uint32_t,uint32_t>* pairs, IdealOffset *idealoffset, Point *d_vertices, int size, int8_t* flags) {
    int pair_id = blockIdx.x;
    
    if (pair_id < size) {
        const pair<uint32_t, uint32_t> pair = pairs[pair_id];
        const uint32_t src_idx = pair.first;
        const uint32_t tar_idx = pair.second;
        const IdealOffset source = idealoffset[src_idx];
        const IdealOffset target = idealoffset[tar_idx];
        const uint32_t src_size = idealoffset[src_idx + 1].vertices_start - source.vertices_start;
        const uint32_t tar_size = idealoffset[tar_idx + 1].vertices_start - target.vertices_start;
        if(src_size != tar_size) {
            flags[pair_id] = 1;
            return;
        }
        
        if(threadIdx.x == 0){
            flags[pair_id] = 0;
        }

        __syncthreads();

        int numPoints = src_size;
        int pointsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;
        int startPoint = threadIdx.x * pointsPerThread;
        int endPoint = min(startPoint + pointsPerThread, numPoints);
        
        for(int i = startPoint; i < endPoint; i ++){
            Point p1 = (d_vertices + source.vertices_start)[i];
            Point p2 = (d_vertices + target.vertices_start)[i];
                
            if (!pointsEqual(p1, p2)) {
                flags[pair_id] = 1;
                break;
            }
        }
    }
}
