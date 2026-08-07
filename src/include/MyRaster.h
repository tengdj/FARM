#ifndef MYRASTER_H
#define MYRASTER_H

#include <cassert>

#include "BaseGeometry.h"
#include "packed_status.h"
#include "../include/MyPolygon.h"

enum PartitionStatus{
	OUT = 0,
	BORDER = 1,
	IN = 2
};

class RasterInfo{
public:
	box mbr;
	int dimx = 0;
	int dimy = 0;
	double step_x = 0.0;
	double step_y = 0.0;
};

class MyRaster : virtual public BaseGeometry{
    pthread_mutex_t raster_lock;
    pthread_mutex_t qtree_lock;
protected:
    uint8_t *status = nullptr;
    double step_x = 0.0;
	double step_y = 0.0;
	int dimx = 0;
	int dimy = 0;
    uint status_size = 0;
    uint8_t bitwidth = 4;
public:
    MyRaster() {
        pthread_mutex_init(&raster_lock, NULL);
        pthread_mutex_init(&qtree_lock, NULL);
    }
    ~MyRaster();
    void init_raster(int num_pixels);
    void init_raster(int dimx, int dimy);

    inline int get_id(int x, int y) {
        assert(x >= 0 && x < dimx);
        assert(y >= 0 && y < dimy);
        return y * dimx + x;
    }
	int get_x(int id);
	int get_y(int id);
	int get_offset_x(double x);
	int get_offset_y(double y);

    void set_status(int id, uint8_t status);
    void set_status(uint8_t *_status) { status = _status; }
    void set_bitwidth(int width) {
        assert(width >= 2 && width <= 8);
        bitwidth = static_cast<uint8_t>(width);
    }
    uint8_t get_bitwidth() const { return bitwidth; }
    uint16_t get_category_count() const { return status_category_count(bitwidth); }
    PartitionStatus show_status(int id);
    uint8_t* get_status() {return status;}
    uint8_t get_fullness(int id) const {
        return read_packed_status(status, static_cast<uint32_t>(id), bitwidth);
    }
    void set_status_size();

	vector<int> get_closest_pixels(box &target);
    int get_closest_pixel(Point &p);
    vector<int> get_pixels(PartitionStatus status);
    box get_pixel_box(int x, int y);
    int get_pixel_id(Point &p);
    vector<int> retrieve_pixels(box *);

	vector<int> expand_radius(int lowx, int highx, int lowy, int highy, int step);
	vector<int> expand_radius(int center, int step);

    void grid_align();
    void merge(int level);
    
    // statistic collection
    size_t get_num_pixels();
    size_t get_num_pixels(PartitionStatus status);
    // double get_pixel_portion(PartitionStatus status);

    // utility
    void print();
    box *extractMER(int starter);

    // get functions
    inline double get_step_x() const {return step_x;}
		inline double get_step_y() const {return step_y;}
		inline int get_dimx() const { return dimx;}
		inline int get_dimy() const { return dimy;}	
	    inline double get_step(bool geography) const {
			if(geography){
				return min(step_x/degree_per_kilometer_longitude(mbr->low[1]), step_y/degree_per_kilometer_latitude);
			}else{
				return min(step_x, step_y);
			}
		}
	    inline double get_pixel_area() const { return step_x * step_y; }
	};


#endif // MYRASTER_H
