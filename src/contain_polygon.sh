#!/bin/bash
make clean
make contain_polygon USE_GPU=1 -j
# for i in {1..20}
# do
# ../build/contain_polygon -s /home/qmh/data/dist.idl -t /home/qmh/data/dist.idl -r -g -v 10 > output.txt
# done
../build/contain_polygon -s /home/qmh/data/idl/has_child.idl -t /home/qmh/data/idl/child.idl -r -g -v 10 > output.txt