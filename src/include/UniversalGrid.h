#ifndef SRC_UNIVERSAL_GRID_H_
#define SRC_UNIVERSAL_GRID_H_

#include "Box.h"
#include "util.h"
#include <cmath>
using namespace std;

class UniversalGrid
{
private:
    double step_x;
    double step_y;

    double max_layers;

    UniversalGrid() {}

    UniversalGrid(const UniversalGrid &) = delete;
    UniversalGrid &operator=(const UniversalGrid &) = delete;

public:
    static UniversalGrid &getInstance()
    {
        static UniversalGrid instance;
        return instance;
    }

    void configure(int _max_layers)
    {
        max_layers = _max_layers;
        step_x = 360.0 / pow(2, max_layers);
        step_x = roundToSignificantDigits(step_x, 2);
        step_y = step_x;
    }

    double get_step_x() { return step_x; }
    double get_step_y() { return step_y; }
    int get_max_layers() { return max_layers; }
};

#endif /* SRC_UNIVERSAL_GRID_H_ */
