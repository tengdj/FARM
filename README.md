# FARM
FARM is a spatial database management system that supports efficient querying of complex polygons with Fill-ratio-Aware Raster Model and GPU Acceleration.

### Installation

The project source code is located in the `src` directory. Please use the `make` command to compile it. Set the environment variable `USE_GPU=1` to enable GPU acceleration support.

install FARM:

```bash
export USE_GPU=1
cd src
make
```

**Prerequisites: **

Operating System: Linux (tested on Ubuntu 23.10).

Compiler: GCC (with C++17 support), NVCC (CUDA compiler).

CUDA Toolkit: Version 12.8 or later.

### Data Preparation

FARM requires the geometric input data to be in binary format. You can generate datasets from WKT format with the following instruction:

```bash
./dump_polygons <input_file> <output_file>
```

**Examples**

```bash
./dump_polygons complex.wkt complex.idl
```

### Execution

To run the program, use the following format:

```bash
# intersect query
./intersect -s <DATA1> -t <DATA2> <arguments>
# intersection query
./intersection -s <DATA1> -t <DATA2> <arguments>
# within distance query
./within_polygon -s <DATA1> -t <DATA2> <arguments>
```

arguments:

`-r` enables the raster model for filtering.

`-g` enables GPU acceleration. If omitted (and using the `_cpu` executable) it runs on the CPU.

`-a` uses approximation query.

`-m` sets the confidence threshold for approximate queries, this controls the balance between precision and recall.

`-b` sets the batch size for GPU processing.

`-v` sets the rasterization granularity.

`-w` sets the fill ratio bit widths.

**Examples**

Run intersect query (use the default parameter values)

``` bash
# gpu version
./intersect -s complex.idl -t lakes.idl -r -g
# cpu version
./intersect_cpu -s complex.idl -t lakes.idl -r
# approximate query
./intersect -s complex.idl -t lakes.idl -r -g -a
```

Run within distance query (use the default parameter values)

```bash
# gpu version
./within_polygon -s lakes.idl -r -g
# cpu version
./within_polygon_cpu -s lakes.idl -r
# approximate query
./within_polygon -s lakes.idl -r -g -a
```

