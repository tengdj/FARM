#include "../include/Ideal.h"
#include "../include/query_context.h"


int main(int argc, char** argv){
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::within;
	global_ctx.num_threads = 1;

    // global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(), global_ctx);

	// for(auto item : global_ctx.source_ideals){
	// 	cout << item->get_num_vertices() << endl;;
	// }

	// preprocess(&global_ctx);
	// cout << "rasterization finished!" << endl;

	// // read all the points
	// global_ctx.load_points();

	Point* source = load_point_wkt("/home/selected_points.csv", global_ctx.target_num, &global_ctx);

	std::string filename = "/home/output.csv";

	std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return 1;
    }

    // 存储整数的容器
    std::vector<int> numbers;
    std::string line;

    // 逐行读取文件
    while (std::getline(file, line)) {
        try {
            int number = std::stoi(line); // 将字符串转换为整数
            numbers.push_back(number);   // 添加到向量中
        } catch (const std::invalid_argument& e) {
            std::cerr << "非整数值，跳过: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "数值超出范围，跳过: " << line << std::endl;
        }
    }
    
    file.close(); // 关闭文件


	for(int i = 0; i < numbers.size(); i ++){
		source[numbers[i]].print();
	}


	return 0;
}