/*
 * cuda_util.h
 *
 *  Created on: Jun 1, 2020
 *      Author: teng
 */

#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <cuda.h>
#include "../include/util.h"
#include "../include/Box.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/box.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <thrust/device_vector.h>
#include <rtspatial/rtspatial.h>

using coord_t = float;
using point_t = boost::geometry::model::d2::point_xy<coord_t>;
using polygon_t = boost::geometry::model::polygon<point_t>;

#define CUDA_SAFE_CALL(call) 										  	  \
	do {																  \
		cudaError_t err = call;											  \
		if (cudaSuccess != err) {										  \
			fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
					__FILE__, __LINE__, cudaGetErrorString(err) );	      \
			exit(EXIT_FAILURE);											  \
		}																  \
	} while (0);

#define CUDA_SWAP_BUFFER()                                                                                              \
    do {																                                                \
        std::swap(gctx->d_BufferInput, gctx->d_BufferOutput);                                                           \
        std::swap(gctx->d_bufferinput_size, gctx->d_bufferoutput_size);                                                 \
        CUDA_SAFE_CALL(cudaMemcpy(&h_bufferinput_size, gctx->d_bufferinput_size, sizeof(uint), cudaMemcpyDeviceToHost));\
        CUDA_SAFE_CALL(cudaMemset(gctx->d_bufferoutput_size, 0, sizeof(uint)));                                         \
    } while(0);     
                                                                                      
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start, 0);
    }

    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    }

    float getElapsedTime() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }

private:
    cudaEvent_t start, stop;
};

inline void check_execution(){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		log(cudaGetErrorString(err));
	}
}

inline void check_execution(string name){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("%s launch failed: %s\n", name.c_str(), cudaGetErrorString(err));
	}
}

// return the distance of two segments

const static double degree_per_meter_latitude_cuda = 360.0/(40076.0*1000);

__device__
inline double degree_per_meter_longitude_cuda(double latitude){
	return 360.0/(sin((90-abs(latitude))*PI/180)*40076.0*1000.0);
}

__device__
inline double distance(const double x1, const double y1, const double x2, const double y2){
	double dx = x1-x2;
	double dy = y1-y2;
	dx = dx/degree_per_meter_longitude_cuda(y1);
	dy = dy/degree_per_meter_latitude_cuda;
	return sqrt(dx*dx+dy*dy);
}

__device__
inline uint getpid1(size_t z){
    size_t w = floor((sqrt(8.0 * z + 1) - 1)/2);
    size_t t = (w*w + w) / 2;
    uint y = (uint)(z - t);
    uint x = (uint)(w - y);
    return x;
}

__device__
inline uint getpid2(size_t z){
    size_t w = floor((sqrt(8.0 * z + 1) - 1)/2);
    size_t t = (w*w + w) / 2;
    uint y = (uint)(z - t);
    return y;
}

__device__
inline uint64_t d_MurmurHash2_x64( const void * key, int len, uint32_t seed ){
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len/8);

    while(data != end)
    {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const uint8_t * data2 = (const uint8_t*)data;

    switch(len & 7)
    {
        case 7: h ^= ((uint64_t)data2[6]) << 48;
        case 6: h ^= ((uint64_t)data2[5]) << 40;
        case 5: h ^= ((uint64_t)data2[4]) << 32;
        case 4: h ^= ((uint64_t)data2[3]) << 24;
        case 3: h ^= ((uint64_t)data2[2]) << 16;
        case 2: h ^= ((uint64_t)data2[1]) << 8;
        case 1: h ^= ((uint64_t)data2[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

__device__ __forceinline__ int binary_search_count(const double *arr, uint32_t start, uint32_t end, double target)
{
	int count = 0;
	uint32_t left = start;
	uint32_t right = end;

	while (left < right)
	{
		uint32_t mid = (left + right) / 2;
		if (arr[mid] <= target)
		{
			count = mid - start + 1;
			left = mid + 1;
		}
		else
		{
			right = mid;
		}
	}

	return count;
}

__device__
inline uint float_to_uint(float xy) {
//    uint ret = 0;
//    if(xy<0){
//        ret = 10000000;
//        xy = 0-xy;
//    }
////        uint inte = (uint)xy;
////        uint decimals = (xy - inte)*10000;
//    ret += (uint)(xy*10000);
//    return ret;
    xy += 180;
    return (uint)(xy*100000);
}

// inline void CopyBoxes(const std::vector<box> &boxes, 
//                       thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> &d_boxes) 
// {
//   pinned_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> h_boxes;

//   h_boxes.resize(boxes.size());

//   for (size_t i = 0; i < boxes.size(); i++) {
//     rtspatial::Point<coord_t, 2> p_min(boxes[i].low[0],
//                                        boxes[i].low[1]);
//     rtspatial::Point<coord_t, 2> p_max(boxes[i].high[0],
//                                        boxes[i].high[1]);

//     h_boxes[i] =
//         rtspatial::Envelope<rtspatial::Point<coord_t, 2>>(p_min, p_max);
//   }

//   d_boxes = h_boxes;
// }

inline void CopyBoxes(const std::vector<box> &boxes, 
    thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> &d_boxes) 
{
    // 首先把boxes复制到设备上
    thrust::device_vector<box> d_input_boxes = boxes;

    // 调整输出向量大小
    d_boxes.resize(boxes.size());

    // 在设备上执行转换
    thrust::transform(d_input_boxes.begin(), d_input_boxes.end(), d_boxes.begin(),
    [] __device__ (const box& b) {
    rtspatial::Point<coord_t, 2> p_min(b.low[0], b.low[1]);
    rtspatial::Point<coord_t, 2> p_max(b.high[0], b.high[1]);
    return rtspatial::Envelope<rtspatial::Point<coord_t, 2>>(p_min, p_max);
    });
}

inline void CopyPoints(Point *points, size_t num,
           thrust::device_vector<rtspatial::Point<coord_t, 2>> &d_points) {
  pinned_vector<rtspatial::Point<coord_t, 2>> h_points;

  h_points.resize(num);

  for (size_t i = 0; i < num; i++) {
    h_points[i] = rtspatial::Point<coord_t, 2>(points[i].x, points[i].y);
  }

  d_points = h_points;
}

template <typename T>
inline void PrintBuffer(T* d_Buffer, uint size){
    T* h_Buffer = new T[size];
    CUDA_SAFE_CALL(cudaMemcpy(h_Buffer, d_Buffer, size * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        if constexpr (std::is_fundamental<T>::value) {
            std::cout << (float)h_Buffer[i] << " ";
            if ((i + 1) % 5 == 0) printf("\n");
        }else{
            h_Buffer[i].print();
        }
    }
    printf("\n");
}

#endif /* CUDA_UTIL_H_ */
