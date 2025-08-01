#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

int main() {
    std::ofstream outfile("result.txt");
    if (!outfile.is_open()) {
        std::cerr << "无法打开输出文件 result.txt" << std::endl;
        return 1;
    }

    for (int i = 1; i <= 18; ++i) {
        std::string filename = "class_" + std::to_string(i) + ".txt";
        std::ifstream infile(filename);

        if (!infile.is_open()) {
            outfile << "文件不存在" << std::endl;
            continue;
        }

        int total = 0;
        int ones = 0;
        std::string line;

        while (std::getline(infile, line)) {
            if (line == "1") {
                ++ones;
            }
            ++total;
        }

        double ratio = (total > 0) ? static_cast<double>(ones) / total : 0.0;
        outfile << std::fixed << std::setprecision(6) << ratio << std::endl;
    }

    outfile.close();
    return 0;
}
