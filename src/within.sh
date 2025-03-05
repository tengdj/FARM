#!/bin/bash
make clean
make within USE_GPU=1 -j
../build/within -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/points.dat -r -g -v 10 --batch_size 100000 > output.txt
