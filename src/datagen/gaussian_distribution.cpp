#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/io/wkt/read.hpp>
#include <boost/geometry/io/wkt/write.hpp>
#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/algorithms/transform.hpp>
#include <boost/geometry/algorithms/envelope.hpp>
#include <boost/geometry/algorithms/intersects.hpp>

namespace bg = boost::geometry;
using Point = bg::model::d2::point_xy<double>;
using Polygon = bg::model::polygon<Point>;
using Box = bg::model::box<Point>;

const double MIN_X = -120.0;
const double MAX_X =  120.0;
const double MIN_Y = -60.0;
const double MAX_Y =  60.0;

// 高斯分布参数
const double MEAN_X = 0.0;
const double MEAN_Y = 0.0;
const double STDDEV_X = 40.0;  // 控制聚集程度
const double STDDEV_Y = 20.0;

const int MAX_ATTEMPTS = 1000;

bool intersects_any(const Box& box, const std::vector<Box>& placed_boxes) {
    for (const auto& b : placed_boxes) {
        if (bg::intersects(box, b)) return true;
    }
    return false;
}

int main() {
    std::ifstream infile("/home/data/areawater.wkt");
    std::ofstream outfile("/home/data/areawater_gaussian.wkt");

    std::vector<Polygon> polygons;
    std::string line;

    while (std::getline(infile, line)) {
        Polygon poly;
        try {
            bg::read_wkt(line, poly);
            polygons.push_back(poly);
        } catch (...) {
            std::cerr << "WKT parse failed: " << line << std::endl;
        }
    }

    size_t n = polygons.size();
    std::cout << "Loaded " << n << " polygons.\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist_x(MEAN_X, STDDEV_X);
    std::normal_distribution<> dist_y(MEAN_Y, STDDEV_Y);

    std::vector<Box> placed_boxes;
    int success = 0;

    for (size_t i = 0; i < n; ++i) {
        const auto& poly = polygons[i];

        Point centroid;
        bg::centroid(poly, centroid);

        bool placed = false;

        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            double tx = dist_x(gen);
            double ty = dist_y(gen);

            // 限制范围
            if (tx < MIN_X || tx > MAX_X || ty < MIN_Y || ty > MAX_Y) continue;

            double dx = tx - centroid.x();
            double dy = ty - centroid.y();

            Polygon moved;
            bg::strategy::transform::translate_transformer<double, 2, 2> translate(dx, dy);
            bg::transform(poly, moved, translate);

            Box moved_box;
            bg::envelope(moved, moved_box);

            if (!intersects_any(moved_box, placed_boxes)) {
                placed_boxes.push_back(moved_box);
                outfile << bg::wkt(moved) << "\n";
                placed = true;
                success++;
                break;
            }
        }

        if (!placed) {
            std::cerr << "Failed to place polygon #" << i << " after " << MAX_ATTEMPTS << " attempts.\n";
        }
    }

    std::cout << "Placed " << success << " / " << n << " polygons.\n";
    return 0;
}
