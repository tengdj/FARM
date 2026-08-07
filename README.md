# FARM

FARM is a spatial database management system that supports efficient querying of complex polygons with Fill-ratio-Aware Raster Model and GPU Acceleration.

### Installation

The project source code is located in the `src` directory. Please use the `make` command to compile the project. Set the environment variable `USE_GPU=1` before compilation to enable GPU acceleration support.

Install FARM:

```bash
export USE_GPU=1
cd src
make all
```

**Prerequisites:**

Operating System: Linux (tested on Ubuntu 23.10).

Compiler: GCC (with C++17 support), NVCC (CUDA compiler).

CUDA Toolkit: Version 12.2 or later.

### Data Preparation

FARM requires the geometric input data to be in a specific format. We developed a dedicated tool to convert WKT format into such specific datasets. You can use it following the below instructions:

```bash
./dump_polygons <input_wkt_file> <output_wkt_file>
```

**Examples**

```bash
./dump_polygons complex.wkt complex.dt
```

### Execution

To run the program, use the following commands:

```bash
# intersect query
./intersect -s <DATA1> -t <DATA2> <arguments>
# intersection query
./intersection -s <DATA1> -t <DATA2> <arguments>
# within distance query
./within_polygon -s <DATA1> -t <DATA2> <arguments>
```

arguments:

`-a` uses approximation query.

`-h` enables hierarchical raster processing.

`-c` sets the confidence threshold for approximate queries, this controls the balance between precision and recall.

`-m` sets the merge threshold for GPU exact intersect and within queries.

`-b` sets the batch size for GPU processing.

`-v` sets the rasterization granularity.

`-w` sets the fill ratio bit widths.

**Examples**

Run intersect query (use the default parameter values)

```bash
# gpu version
./intersect -s complex.dt -t lakes.dt -h
# cpu version
./intersect_cpu -s complex.dt -t lakes.dt -h
# approximate query
./intersect -s complex.dt -t lakes.dt -h -a
```

Run within distance query (use the default parameter values, within-distance threshold = 10KM)

```bash
# gpu version
./within_polygon -s lakes.dt -h
# cpu version
./within_polygon_cpu -s lakes.dt -h
# approximate query
./within_polygon -s lakes.dt -h -a
```
