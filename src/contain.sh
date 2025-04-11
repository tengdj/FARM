#!/bin/bash
make clean
make contain USE_GPU=1 -j
# ../build/contain -s /home/qmh/data/rayjoin/lakes_Africa_Point.idl -t /home/qmh/data/rayjoin/parks_Africa_Point.dat -r -g -v 10 > output.txt
# ../build/contain -s /home/qmh/data/test_polygon.idl -t /home/qmh/data/test_point.dat -r -g -v 10 > output.txt
# for i in {1..20}
# do
../build/contain -s ~/IDEAL/src/input_polygons.wkt -t ~/IDEAL/src/input_points.wkt -r -g -v 10 > contain.txt
# done

# for i in $(seq 0 1 37);
# do
#     ../build/contain -s /home/qmh/data/dist.idl -t /home/qmh/data/docker/selected_points.dat -r -g -n 1 --big_threshold $i >> output.txt
# done