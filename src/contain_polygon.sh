#!/bin/bash
make clean
make contain_polygon USE_GPU=1 -j
# ../build/contain_polygon -s /home/qmh/data/rayjoin/USACensusBlockGroupBoundaries_Point.idl -t /home/qmh/data/rayjoin/USADetailedWaterBodies_Point.idl -r -g -v 100 > output.txt
../build/contain_polygon -s /home/qmh/data/rayjoin/dtl_cnty_Point.idl -t /home/qmh/data/rayjoin/USAZIPCodeArea_Point.idl -r -g -v 10 > output.txt