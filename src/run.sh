#!/bin/bash
DATA_PATH="/home/qmh/data/idl"
BUILD_PATH="/home/qmh/IDEAL/build"
DATA1="has_child.idl"
DATA2="child.idl"
DATA3="complex.idl"
DATA4="points.dat"
DATA5="sampled.points.dat"

> output.txt
# make clean
# make contain USE_GPU=1 -j
# $BUILD_PATH/contain -s $DATA_PATH/$DATA3 -t $DATA_PATH/$DATA4 -r -g -b 1000000 -l 0.00005 >> output.txt

# make clean
# make contain_polygon USE_GPU=1 -j
# $BUILD_PATH/contain_polygon -s $DATA_PATH/$DATA1 -t $DATA_PATH/$DATA2 -r -g >> output.txt

# make clean
# make within USE_GPU=1  -j
# $BUILD_PATH/within -s $DATA_PATH/$DATA2 -t $DATA_PATH/$DATA4 -r -h -g -b 1000000 -l 0.0002 >> output.txt

# make clean
# make within_polygon USE_GPU=1 -j
# $BUILD_PATH/within_polygon -s $DATA_PATH/$DATA2 -t $DATA_PATH/$DATA2 -r -h -g -l 0.1 > output.txt

make clean
make intersection USE_GPU=1 -j
$BUILD_PATH/intersection -s $DATA_PATH/$DATA1 -t $DATA_PATH/$DATA1 -r -g >> output.txt
# for i in {1..100}; do
# $BUILD_PATH/intersection -s inputA.wkt -t inputB.wkt -r -g -n 1 >> output.txt
# done