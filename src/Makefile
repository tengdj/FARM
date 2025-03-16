CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I ./include -I /usr/local/include -I /usr/local/cuda-12.2/targets/x86_64-linux/include  -I /home/qmh/optix/include
#-lgeos_c  
LIBS		= -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -lboost_program_options -lgeos -lpthread
CPPFLAGS	= -O3 -std=c++17 -Wno-cpp

NVCCFLAGS   = -O3 -arch=sm_89 -std=c++17 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math
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

PTXS := $(wildcard index/*.ptx)

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


contain_polygon:	query/contain_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS)
	
within_polygon:	query/within_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 
	
within:	query/within.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 
	
contain:	query/contain.o $(GEOMETRY_OBJS)
	$(CXX) -O3 -o ../build/$@ $^ $(LIBS) 
	
print_raster:	test/print_raster.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 
		
print_layers:	test/print_layers.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

distance_test:	test/distance_test.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

dump_points:	datagen/dump_points.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

dump_polygons:	datagen/dump_polygons.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

dist_point_polygon:	test/dist_point_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

rtspatial:	test/rtspatial_cu.o $(GEOMETRY_OBJS) $(PTXS)
	$(NVCC) -o ../build/$@ $^ $(LIBS) 

clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)