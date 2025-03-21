#!/bin/bash
make clean
make within USE_GPU=1 -j
../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -h -g -v 10 --batch_size 1000000 > output.txt
../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -g -v 10 --batch_size 1000000 > output1.txt
# ../build/within -s /home/qmh/data/wkt/single_polygon.wkt -t /home/qmh/data/wkt/single_point.wkt -r -h -g -v 10 --batch_size 1000000 > output.txt
# ../build/within -s /home/qmh/data/wkt/single_polygon.wkt -t /home/qmh/data/wkt/single_point.wkt -r -g -v 10 --batch_size 1000000 > output1.txt
