#ifndef MYRASTER_H
#define MYRASTER_H

#include "BaseGeometry.h"
#include "../index/QTree.h"
#include "../include/MyPolygon.h"

enum PartitionStatus{
	OUT = 0,
	BORDER = 1,
	IN = 2
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
public:
    MyRaster() {
        pthread_mutex_init(&raster_lock, NULL);
        pthread_mutex_init(&qtree_lock, NULL);
    }
    ~MyRaster();
    void init_raster(int num_pixels);
    void init_raster(int dimx, int dimy);

    int get_id(int x, int y);
	int get_x(int id);
	int get_y(int id);
	int get_offset_x(double x);
	int get_offset_y(double y);

    void set_status(int id, PartitionStatus status);
    PartitionStatus show_status(int id);
    uint8_t* get_status() {return status;}

	vector<int> get_intersect_pixels(box *pix);
    vector<int> get_closest_pixels(box &target);
    int get_closest_pixel(Point &p);
    vector<int> get_pixels(PartitionStatus status);
    box get_pixel_box(int x, int y);
    int get_pixel_id(Point &p);
    vector<int> retrieve_pixels(box *);

    bool contain(box *b, bool &contained);
    vector<int> expand_radius(int lowx, int highx, int lowy, int highy, int step);
	vector<int> expand_radius(int center, int step);

    void grid_align(query_context *gctx);
    
    // statistic collection
    size_t get_num_pixels();
    size_t get_num_pixels(PartitionStatus status);
    // double get_pixel_portion(PartitionStatus status);


    // utility
    void print();

    box *extractMER(int starter);

    // get functions
    inline double get_step_x() {return step_x;}
	inline double get_step_y( ){return step_y;}
	inline int get_dimx(){ return dimx;}
	inline int get_dimy(){ return dimy;}	
    inline double get_step(bool geography){
		if(geography){
			return min(step_x/degree_per_kilometer_longitude(mbr->low[1]), step_y/degree_per_kilometer_latitude);
		}else{
			return min(step_x, step_y);
		}
	}
};


#endif // MYRASTER_H