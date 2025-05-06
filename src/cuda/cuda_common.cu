/*
 *
 * with some common gpu related operations
 * */

#include <cuda.h>
#include "mygpu.h"
#include "cuda_util.h"
#include "../include/util.h"

using namespace std;

int get_num_cores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
      }
    return cores;
}
vector<gpu_info *> get_gpus(){
	vector<gpu_info *> gpus;
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		//cudaGetDeviceProperties(&prop, i);
		gpu_info *info = new gpu_info();
		strcpy(info->name, prop.name);
		info->busy = false;
		info->mem_size = prop.totalGlobalMem/1024/1024;
		info->device_id = i;
		info->clock_rate = prop.memoryClockRate;
		info->bus_width = prop.memoryBusWidth;
		info->compute_capability_major = prop.major;
		info->compute_capability_minor = prop.minor;
		info->num_cores = get_num_cores(prop);

//		if(info->mem_size>2048){
//			info->mem_size = 2048;
//		}
		gpus.push_back(info);
	}
	return gpus;
}

void print_gpus(){
	vector<gpu_info *> gpus = get_gpus();
	for(gpu_info *g:gpus){
		g->print();
		delete g;
	}
	gpus.clear();
}

void gpu_info::print(){
	log("Device ID: %d", device_id);
	log("  Device name: %s", name);
	log("  Memory Clock Rate (KHz): %d", clock_rate);
	log("  Memory Bus Width (bits): %d", bus_width);
	log("  Peak Memory Bandwidth (GB/s): %f", 2.0*clock_rate*(bus_width/8)/1.0e6);
	log("  Memory size (MB): %ld", mem_size);
	log("  Compute Capability: %d.%d", this->compute_capability_major, this->compute_capability_minor);
	log("  Number of cores: %ld", num_cores);
}

void gpu_info::init(){
	for(int i=0;i<MAX_DATA_SPACE;i++){
		data_size[i] = 0;
		d_data[i] = NULL;
	}
}

void *gpu_info::allocate(size_t ss){
	int tid = 0;
	for(int i=0;i<MAX_DATA_SPACE;i++){
		if(!d_data[i]){
			tid = i;
			break;
		}
	}
	assert(tid<MAX_DATA_SPACE);
	cudaSetDevice(this->device_id);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data[tid], ss));
	assert(d_data[tid]);
	data_size[tid] = ss;

	return d_data[tid];
}

void gpu_info::clear(){
	cudaSetDevice(this->device_id);
	for(int i=0;i<MAX_DATA_SPACE;i++){
		if(d_data[i]){
			CUDA_SAFE_CALL(cudaFree(d_data[i]));
			d_data[i] = NULL;
			data_size[i] = 0;
		}
	}
}

gpu_info::~gpu_info(){
	clear();
}
