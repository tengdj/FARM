#!/bin/bash
make clean
make within_polygon USE_GPU=1 -j
../build/within_polygon -s /home/qmh/data/idl/child.idl -t /home/qmh/data/idl/child.idl -r -g -h > output.txt