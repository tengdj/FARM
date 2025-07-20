#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

void compute_stats(const std::string& filename, double& mean, double& stddev) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        mean = stddev = NAN;
        return;
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double val;
        if (iss >> val)
            values.push_back(val);
    }

    if (values.empty()) {
        mean = stddev = NAN;
        return;
    }

    double sum = 0.0;
    for (double v : values) sum += v;
    mean = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) sq_sum += (v - mean) * (v - mean);
    stddev = std::sqrt(sq_sum / values.size());  // 若想用样本标准差，可用 values.size() - 1
}

int main() {
    std::ofstream outfile("result.txt");
    if (!outfile) {
        std::cerr << "无法创建输出文件 result.txt" << std::endl;
        return 1;
    }

    outfile << std::fixed << std::setprecision(6); // 控制输出精度

    for (int i = 1; i <= 18; ++i) {
        std::string filename = "class_" + std::to_string(i) + ".txt";
        double mean, stddev;
        compute_stats(filename, mean, stddev);
        outfile << mean << ", " << stddev << std::endl;
    }

    outfile.close();
    return 0;
}
