#include "../include/Ideal.h"
#include "ThreadPool.h"
#include <fstream>
#include <queue>
#include <chrono>

#include <sstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <cmath>

struct SegmentWithStatus {
    Segment segment;
    bool is_pip1;
    
    SegmentWithStatus(const Segment& seg, bool status) 
        : segment(seg), is_pip1(status) {}
};

// 将Segments转换为多个Polygons
std::vector<MyPolygon*> segmentsToPolygons(const std::vector<SegmentWithStatus>& data, Point* vertices, pair<uint32_t, uint32_t>* pairs, IdealOffset* idealoffset){
    std::vector<MyPolygon*> polygons; 
    
    if (data.empty()) return polygons;
    
    // 创建segment端点的邻接表
    // 键: 端点坐标的字符串表示，值: 与该端点相连的所有segments的索引
    std::unordered_map<std::string, std::vector<size_t>> adjacencyList;
	// 跟踪已使用的segments
	std::vector<bool> used(data.size(), false);
    
    auto pointToKey = [](const Point& p) {
        // 使用足够精度将点转换为唯一的字符串键
        return std::to_string(p.x) + ":" + std::to_string(p.y);
    };
    
    // 构建邻接表
    for (size_t i = 0; i < data.size(); ++i) {
        Segment segment = data[i].segment;
		adjacencyList[pointToKey(segment.start)].push_back(i);
    }
    
    // 寻找并构建所有可能的多边形
    for (size_t startIdx = 0; startIdx < data.size(); ++startIdx) {
        bool status = data[startIdx].is_pip1;
        Segment segment = data[startIdx].segment;
        if (used[startIdx] || !status) continue;
		// printf("START IDX POINT(%lf %lf) POINT(%lf %lf) %d %d %d\n", segments[startIdx].start.x, segments[startIdx].start.y, 
        //     segments[startIdx].end.x, segments[startIdx].end.y, segments[startIdx].edge_start, segments[startIdx].edge_end, segments[startIdx].pair_id);
        // 开始一个新的多边形
        vector<Point> currentVertices;
        // currentVertices.reserve(100000000);
        
        // 当前segment和端点
        size_t currentSegIdx = startIdx;
        Point currentPoint = segment.start;
        Point startPoint = currentPoint;
        
        bool foundCycle = false;
  
        // 尝试找到一个闭合的路径
        while (!used[currentSegIdx]) {
            used[currentSegIdx] = true;
            // 添加当前点到多边形
            currentVertices.push_back(currentPoint);
			const Segment& seg = data[currentSegIdx].segment;

            // printf("POINT(%lf %lf) POINT(%lf %lf) %d %d %d status: %d\n", seg.start.x, seg.start.y, 
            //    seg.end.x, seg.end.y, seg.edge_start, seg.edge_end, seg.pair_id, status[currentSegIdx]);
			
            if(seg.edge_start != -1){
                if(seg.edge_start <= seg.edge_end){
                    for(int verId = seg.edge_start; verId <= seg.edge_end; verId ++){
                        currentVertices.push_back(vertices[verId]);
                    }
                }else{
                    pair<uint32_t, uint32_t> pair = pairs[seg.pair_id];
                    uint32_t offset_idx = seg.is_source ? pair.first : pair.second;
                    // uint vertex_size = idealoffset[offset_idx + 1].vertices_start - idealoffset[offset_idx].vertices_start;
                    for(int verId = seg.edge_start; verId < idealoffset[offset_idx + 1].vertices_start - 1; verId ++){
                        currentVertices.push_back(vertices[verId]);
                    }
                    for(int verId = idealoffset[offset_idx].vertices_start; verId <= seg.edge_end; verId ++){
                        currentVertices.push_back(vertices[verId]);
                    }
                }
			}

            
            // 确定segment的另一个端点
           
            Point nextPoint = currentPoint == seg.start ? seg.end : seg.start;
            
            // 寻找连接到nextPoint的未使用segment
            bool foundNext = false;
            std::string nextKey = pointToKey(nextPoint);
            
            if (adjacencyList.find(nextKey) != adjacencyList.end()) {
                for (size_t idx : adjacencyList[nextKey]) {
                    if (!used[idx]) {
                        currentSegIdx = idx;
                        currentPoint = nextPoint;
                        foundNext = true;
                        break;
                    }
                }
            }
            
            // 如果回到起点，我们找到了一个闭合的多边形
            if (!foundNext && nextPoint == startPoint) {
                currentVertices.push_back(nextPoint); // 添加最后一个点闭合多边形
                foundCycle = true;
                break;
            }
            
            // 如果没有找到下一个segment，则路径不能闭合
            if (!foundNext) break;
        }
        
        // 如果找到了闭合的多边形并且至少有3个点，添加到结果中
        if (foundCycle && currentVertices.size() >= 3) {
			VertexSequence* vs = new VertexSequence(currentVertices.size(), currentVertices.data());
			MyPolygon *currentPolygon = new MyPolygon();
			currentPolygon->set_boundary(vs);
            polygons.push_back(currentPolygon);
        } 
        // else {
        //     // printf("有错误数据：\n");
        //     // 恢复标记，这些segments可能是其他多边形的一部分
        //     for (size_t idx : polygonSegments) {
        //         used[idx] = false;
        //     }
        // }
    }

    
    // 处理未使用的segments
    // 可能有些segments不能形成闭合多边形，可以根据需要处理这些segments

    return polygons;
}

std::vector<MyPolygon*> processSegmentsGroup(const std::vector<SegmentWithStatus>& data, Point* vertices, pair<uint32_t, uint32_t>* pairs, IdealOffset* idealoffset) {
    if (data.empty()) return std::vector<MyPolygon*>();
    
    std::vector<MyPolygon*> results = segmentsToPolygons(data, vertices, pairs, idealoffset);
    
    return results;
}

int main(int argc, char** argv) {
	query_context global_ctx;
	global_ctx = get_parameters(argc, argv);
	global_ctx.query_type = QueryType::intersection;

	global_ctx.source_ideals = load_binary_file(global_ctx.source_path.c_str(),global_ctx);
	global_ctx.target_ideals = load_binary_file(global_ctx.target_path.c_str(),global_ctx);
    // global_ctx.source_ideals.resize(1000);
    // global_ctx.target_ideals.resize(1000);
	global_ctx.target_num = global_ctx.target_ideals.size();
    // global_ctx.batch_size = global_ctx.target_num;
    
	indexBuild(&global_ctx);

	auto rtree_query_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < global_ctx.target_num; i += global_ctx.batch_size){
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.target_num);
        printf("index = %d index_end = %d\n", global_ctx.index, global_ctx.index_end);
		indexQuery(&global_ctx);
	}
	auto rtree_query_end = std::chrono::high_resolution_clock::now();
	auto rtree_query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rtree_query_end - rtree_query_start);
	std::cout << "rtree query: " << rtree_query_duration.count() << " ms" << std::endl;
	indexDestroy(&global_ctx);

    global_ctx.batch_size = 1000000;
    // global_ctx.batch_size = global_ctx.num_pairs / 3;
	auto preprocess_start = std::chrono::high_resolution_clock::now();
	preprocess(&global_ctx);
	auto preprocess_end = std::chrono::high_resolution_clock::now();
	auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start);
	std::cout << "preprocess time: " << preprocess_duration.count() << " ms" << std::endl;

	auto preprocess_gpu_start = std::chrono::high_resolution_clock::now();
	preprocess_for_gpu(&global_ctx);
	auto preprocess_gpu_end = std::chrono::high_resolution_clock::now();
	auto preprocess_gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_gpu_end - preprocess_gpu_start);
	std::cout << "preprocess for gpu time: " << preprocess_gpu_duration.count() << " ms" << std::endl;

	auto total_start = std::chrono::high_resolution_clock::now();
    
	for(int i = 0; i < global_ctx.num_pairs; i += global_ctx.batch_size){
        auto batch_start = std::chrono::high_resolution_clock::now();
		global_ctx.index = i;
		global_ctx.index_end = min(i + global_ctx.batch_size, global_ctx.num_pairs);

		ResetDevice(&global_ctx);

	    auto gpu_start = std::chrono::high_resolution_clock::now();

		cuda_contain_polygon(&global_ctx);

        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        std::cout << "total gpu time: " << gpu_duration.count() << " ms" << std::endl;

        /*
        std::unordered_map<int, std::vector<SegmentWithStatus>> groupedData;

        groupedData.reserve(global_ctx.num_pairs);

        // std::unordered_map<int, std::vector<Segment>> groupedSegments;
        // std::unordered_map<int, std::vector<bool>> groupedStatus;

        auto group_start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < global_ctx.num_segments; j ++){
            if(global_ctx.pip[j] == 1){
                auto& segment = global_ctx.segments[j];
                groupedData[segment.pair_id].emplace_back(segment, true);
            }
        }

        for (int j = 0; j < global_ctx.num_segments; j ++){
            if(global_ctx.pip[j] == 2){
                auto& segment = global_ctx.segments[j];
                groupedData[segment.pair_id].emplace_back(segment, false);
            }
        }

        auto group_end = std::chrono::high_resolution_clock::now();
        auto group_duration = std::chrono::duration_cast<std::chrono::milliseconds>(group_end - group_start);
        std::cout << "group time: " << group_duration.count() << " ms" << std::endl;

        auto cpu_start = std::chrono::high_resolution_clock::now();
        unsigned int num_threads = global_ctx.num_threads;
        Point* vertices = global_ctx.h_vertices;
        pair<uint32_t, uint32_t>* pairs = global_ctx.h_candidate_pairs + global_ctx.index;
        IdealOffset* idealoffset = global_ctx.h_idealoffset;
        uint8_t *pip = global_ctx.pip;

        ThreadPool pool(num_threads);
        
        std::atomic<int> completed_tasks(0);
        const int total_tasks = groupedData.size();
        
        std::vector<MyPolygon*> allPolygons;
        std::mutex resultsMutex;

        struct LocalBuffer {
            std::vector<MyPolygon*> polygons;
            LocalBuffer() { polygons.reserve(5000); }
        };

        std::vector<std::shared_ptr<LocalBuffer>> localBuffers(num_threads);
        for (unsigned int i = 0; i < num_threads; ++i) {
            localBuffers[i] = std::make_shared<LocalBuffer>();
        }

        int task_id = 0;
        for (const auto& group : groupedData) {
            const int thread_id = task_id % num_threads;
            task_id++;
            
            std::shared_ptr<LocalBuffer> localBuffer = localBuffers[thread_id];
            
            pool.enqueue([&group, &completed_tasks, total_tasks, localBuffer, vertices, pairs, idealoffset]() {
                std::vector<MyPolygon*> groupPolygons = processSegmentsGroup(group.second, vertices, pairs, idealoffset);

                if (!groupPolygons.empty()) {
                    localBuffer->polygons.insert(
                        localBuffer->polygons.end(), 
                        std::make_move_iterator(groupPolygons.begin()),
                        std::make_move_iterator(groupPolygons.end())
                    );
                }
                
                int completed = ++completed_tasks;
                // if (completed % 10 == 0 || completed == total_tasks) {
                //     std::cout << "处理进度: " << completed << "/" << total_tasks 
                //             << " (" << (completed * 100 / total_tasks) << "%)\r" << std::flush;
                // }
            });
        }

        pool.waitAll();
        std::cout << std::endl << "所有任务完成，合并结果..." << std::endl;

        size_t totalPolygons = 0;
        for (const auto& buffer : localBuffers) {
            totalPolygons += buffer->polygons.size();
        }
        
        allPolygons.reserve(totalPolygons);
        
        for (const auto& buffer : localBuffers) {
            allPolygons.insert(
                allPolygons.end(),
                std::make_move_iterator(buffer->polygons.begin()),
                std::make_move_iterator(buffer->polygons.end())
            );
        }
        
        std::cout << "处理完成！共生成 " << allPolygons.size() << " 个多边形" << std::endl;
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        std::cout << "cpu time: " << cpu_duration.count() << " ms" << std::endl;
        
        // for(auto p : allPolygons){
        //     p->MyPolygon::print();
        // }
        */

		auto batch_end = std::chrono::high_resolution_clock::now();
		auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
		std::cout << "batch time: " << batch_duration.count() << " ms" << std::endl;
    }

	auto total_end = std::chrono::high_resolution_clock::now();
	auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
	std::cout << "total query time: " << total_duration.count() << " ms" << std::endl;

    return 0;
}