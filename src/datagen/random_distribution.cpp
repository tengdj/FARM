#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/read.hpp>
#include <boost/geometry/io/wkt/write.hpp>
#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/algorithms/transform.hpp>

namespace bg = boost::geometry;

using Point = bg::model::d2::point_xy<double>;
using Polygon = bg::model::polygon<Point>;

// 设置输出区域的范围
const double MIN_X = -180.0;
const double MAX_X = 180.0;
const double MIN_Y = -90.0;
const double MAX_Y = 90.0;

int main() {
    std::ifstream infile("/home/qmh/data/wkt/valid_child.wkt");
    std::ofstream outfile("/home/qmh/data/wkt/valid_child_normal.wkt");

    std::vector<Polygon> polygons;
    std::string line;

    // 1. 读取所有 polygon
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

    // 2. 生成目标中心点位置（均匀分布）
    std::vector<Point> target_centers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_x(MIN_X, MAX_X);
    std::uniform_real_distribution<> dist_y(MIN_Y, MAX_Y);

    for (size_t i = 0; i < n; ++i) {
        target_centers.emplace_back(dist_x(gen), dist_y(gen));
    }

    // 3. 打乱目标中心点顺序
    std::shuffle(target_centers.begin(), target_centers.end(), gen);

    // 4. 平移每个 polygon
    for (size_t i = 0; i < n; ++i) {
        Point centroid;
        bg::centroid(polygons[i], centroid);

        double dx = target_centers[i].x() - centroid.x();
        double dy = target_centers[i].y() - centroid.y();

        Polygon moved;
        bg::strategy::transform::translate_transformer<double, 2, 2> translate(dx, dy);
        bg::transform(polygons[i], moved, translate);

        outfile << bg::wkt(moved) << "\n";
    }

    std::cout << "Finished.\n";
    return 0;
}
