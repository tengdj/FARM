#!/bin/bash
make clean
make within USE_GPU=1 -j
> output.txt
for i in {1..1}
do
../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -h -g --batch_size 1000000 >> output.txt
done
# ../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -g -v 10 --batch_size 1000000 > output.txt
# ../build/within -s /home/qmh/data/wkt/single_polygon.wkt -t /home/qmh/data/wkt/single_point.wkt -r -h -g -v 10 --batch_size 1000000 > output.txt
# ../build/within -s /home/qmh/data/wkt/single_polygon.wkt -t /home/qmh/data/wkt/single_point.wkt -r -g -v 10 --batch_size 1000000 > output1.txt
