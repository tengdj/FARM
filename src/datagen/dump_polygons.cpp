#include "../include/Ideal.h"
#include "../include/query_context.h"

int main(int argc, char** argv) {
	if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output>\n";
        return 1;
    }

    string input_path = argv[1];;
    string output_path = argv[2];
	
	vector<Ideal *> polygons = load_polygon_wkt(input_path.c_str());
    dump_polygons_to_file(polygons, output_path.c_str());
    return 0;
}