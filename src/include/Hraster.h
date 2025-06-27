#ifndef SRC_HRASTER_H_
#define SRC_HRASTER_H_

#include "MyRaster.h"

class Hraster : public MyRaster{
public:
    void init(double _step_x, double _step_y, int _dimx, int _dimy, box *mbr, bool last_layer);
	void print();
};

#endif /* SRC_HRASTER_H_ */