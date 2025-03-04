#!/bin/bash
make clean
make contain_polygon USE_GPU=1 -j
../build/contain_polygon -s /home/qmh/data/idl/has_child.idl -t /home/qmh/data/idl/child.idl -r -g -v 10 --batch_size 10000000 >> output.txt