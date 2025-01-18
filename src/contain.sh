#!/bin/bash
make clean
make contain USE_GPU=1 -j
# ../build/contain -s /home/qmh/data/rayjoin/lakes_Africa_Point.idl -t /home/qmh/data/rayjoin/parks_Africa_Point.dat -r -g -v 10 > output.txt
../build/contain -s /home/qmh/data/dist.idl -t /home/qmh/data/sampled.points.dat -r -g -v 100 >> output.txt

# for i in $(seq 0 1 37);
# do
#     ../build/contain -s /home/qmh/data/dist.idl -t /home/qmh/data/docker/selected_points.dat -r -g -n 1 --big_threshold $i >> output.txt
# done