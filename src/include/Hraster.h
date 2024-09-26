#ifndef SRC_HRASTER_H_
#define SRC_HRASTER_H_

#include "MyRaster.h"

class Hraster : public MyRaster{
public:
    void init(double _step_x, double _step_y, int _dimx, int _dimy, box *mbr, bool last_layer);
	inline void set_status(uint8_t *_status) { status = _status; }
    PartitionStatus merge_status (box target);
    pair<uint32_t, uint32_t> *merge_edeg_sequences (box target);
	void merge(Hraster &r);
	void print();
};

#endif /* SRC_HRASTER_H_ */