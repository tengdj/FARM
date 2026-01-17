#include "Ideal.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

using namespace std;

// 设置输出区域的范围
const double MIN_X = -120.0;
const double MAX_X = 120.0;
const double MIN_Y = -60.0;
const double MAX_Y = 60.0;

// 计算多边形质心
Point polygonCentroid(Ideal *ideal) {
    size_t n = ideal->get_num_vertices();
    double A = 0.0, C_x = 0.0, C_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const auto& [x_i, y_i] = ideal->get_boundary()->p[i];
        const auto& [x_next, y_next] = ideal->get_boundary()->p[(i + 1) % n]; // 下一个顶点，循环到第一个
        double cross = x_i * y_next - x_next * y_i;
        A += cross;
        C_x += (x_i + x_next) * cross;
        C_y += (y_i + y_next) * cross;
    }

    A *= 0.5; // 面积
    if (std::abs(A) < 1e-10) { // 避免除以零
        return {0.0, 0.0}; // 无效多边形，返回默认值
    }

    C_x /= (6.0 * A);
    C_y /= (6.0 * A);

    return {C_x, C_y};
}

int main() {
    // 1. 读取所有 polygon

    string infile = "/home/data/wkt/complex.wkt";
    // std::ofstream outfile("/home/data/wkt/valid_lakes_polygons_100_normal.wkt");

    vector<Ideal*> source_ideals = load_polygon_wkt(infile.c_str());

    std::string line;



    size_t n = source_ideals.size();

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
        Point centroid = polygonCentroid(source_ideals[i]); 

        double dx = target_centers[i].x - centroid.x;
        double dy = target_centers[i].y - centroid.y;

        for(int j = 0; j < source_ideals[i]->get_num_vertices(); j ++){
            source_ideals[i]->get_boundary()->p[j].x += dx;
            source_ideals[i]->get_boundary()->p[j].y += dy;
        }

        source_ideals[i]->MyPolygon::print();
    }

    return 0;
}
