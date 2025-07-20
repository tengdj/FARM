CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I ./include -I /usr/local/include -I /usr/local/cuda-12.2/targets/x86_64-linux/include  -I /home/qmh/optix/include
#-lgeos_c  
LIBS		= -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -lboost_program_options -lgeos -lpthread
CPPFLAGS	= -O2 -std=c++17 -Wno-cpp

NVCCFLAGS   = -O2 -arch=sm_89 -std=c++17 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math
CUDALIBS    = -L/usr/local/cuda-12.2/lib64/stubs -lcuda -L/usr/local/cuda-12.2/lib64/ -lcudart


GEOMETRY_SRCS := $(wildcard geometry/*.cpp)
GEOMETRY_SRCS += $(wildcard geometry/triangulate/*.cpp)
GEOMETRY_SRCS += $(wildcard storage/*.cpp)
GEOMETRY_OBJS := $(patsubst %.cpp,%.o,$(GEOMETRY_SRCS))

PARTITION_SRCS := $(wildcard partition/*.cpp)
PARTITION_OBJS := $(patsubst %.cpp,%.o,$(PARTITION_SRCS))

#compile all the cu files
CU_SRCS := $(wildcard cuda/*.cu)
CU_OBJS := $(patsubst %.cu,%_cu.o,$(CU_SRCS))

ifdef USE_GPU
	CPPFLAGS += -DUSE_GPU
	LIBS  += ${CUDALIBS}
	GEOMETRY_OBJS += ${CU_OBJS}
endif

%_cu.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<	
	
%.o:	%.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<	

all:	contain_polygon within_polygon


intersect:	query/intersect.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS)

intersect_cpu:	query/intersect_cpu.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS)
	
within_polygon:	query/within_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS) 

within_polygon_cpu:	query/within_polygon_cpu.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS) 

intersection:	query/intersection.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS)

intersection_cpu:	query/intersection_cpu.o $(GEOMETRY_OBJS)
	$(CXX) -O2 -o ../build/$@ $^ $(LIBS)
	
print_raster:	test/print_raster.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 
		
print_layers:	test/print_layers.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 

dump_points:	datagen/dump_points.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

dump_polygons:	datagen/dump_polygons.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

random_distribution:	datagen/random_distribution.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 

filter_polygons:	datagen/filter_polygons.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 

clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)