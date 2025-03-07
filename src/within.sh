#!/bin/bash
make clean
make within USE_GPU=1 -j
# ../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -h -g -v 10 --batch_size 10000000 > output1.txt
../build/within -s /home/qmh/data/dist.idl -t /home/qmh/data/idl/sampled.points.dat -r -h -g -v 10 --batch_size 10000000 > output.txt
../build/within -s /home/qmh/data/dist.idl -t /home/qmh/data/idl/sampled.points.dat -r -g -v 10 --batch_size 10000000 > output1.txt


