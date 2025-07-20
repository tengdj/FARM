#include <iostream>
#include <fstream>
#include <string>
#include "Ideal.h"

bool is_valid_polygon(Ideal *poly) {
    for (int i = 0; i < poly->get_num_vertices(); i ++) {
        Point pt = poly->get_boundary()->p[i];
        double x = pt.x;
        double y = pt.y;
        if (!(x > -180.0 && x < 180.0 && y > -90.0 && y < 90.0)) {
            return false;
        }
    }
    return true;
}

int main() {
    string infile = "/home/qmh/data/wkt/complex_normal.wkt";
    vector<Ideal*> source_ideals = load_polygon_wkt(infile.c_str());
    vector<Ideal*> polygons;

    for(auto p : source_ideals){
        if(is_valid_polygon(p)){
            polygons.push_back(p);
        }
    }

    log("%d %d\n", source_ideals.size(), polygons.size());

    dump_polygons_to_file(polygons, "/home/qmh/data/exp/complex_normal.idl");
    return 0;
}
