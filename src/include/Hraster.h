#ifndef SRC_HRASTER_H_
#define SRC_HRASTER_H_

#include "MyRaster.h"

class Hraster : public MyRaster{
	double *areas = nullptr;
	bool owns_status = false;
	bool owns_areas = false;
	bool owns_mbr = false;
public:
	Hraster() = default;
	~Hraster();
	void init(double _step_x, double _step_y, int& _dimx, int& _dimy, box *mbr, bool last_layer);
	void attach_base_storage(uint8_t *_status, double *_areas);
	double get_area(int id) const { return areas[id]; }
	double *get_areas() const { return areas; }
	void set_area(int id, double area) { areas[id] = area; }
	void print();
};

#endif /* SRC_HRASTER_H_ */
