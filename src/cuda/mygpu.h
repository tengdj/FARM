/*
 * mygpu.h
 *
 *  Created on: Dec 9, 2019
 *      Author: teng
 */

#ifndef MYGPU_H_
#define MYGPU_H_

#include <pthread.h>
#include <vector>
#include "../include/util.h"
using namespace std;

#define MAX_DATA_SPACE 40

class gpu_info{

public:
	int device_id;
	char name[256];
	int clock_rate = 0;
	int bus_width = 0;
	int num_cores = 0;
	size_t mem_size;
	bool busy;
	pthread_mutex_t lk;
	void *d_data[MAX_DATA_SPACE];
	size_t data_size[MAX_DATA_SPACE];
	int compute_capability_major = 0;
	int compute_capability_minor = 0;


	void init();
	~gpu_info();
	void clear();
	void *allocate(size_t ss);
	void print();
	size_t size_allocated(){
		size_t size = 0;
		for(int i=0;i<MAX_DATA_SPACE;i++){
			size += data_size[i];
		}
		return size;
	}
	void lock(){
		pthread_mutex_lock(&lk);
	}
	void unlock(){
		pthread_mutex_unlock(&lk);
	}
};

vector<gpu_info *> get_gpus();
void print_gpus();


#endif /* MYGPU_H_ */
