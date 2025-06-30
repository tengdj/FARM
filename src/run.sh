#!/bin/bash
DATA_PATH="/home/qmh/data"
IDL_PATH="/home/qmh/data/idl"
BUILD_PATH="/home/qmh/IDEAL/build"
DATA1="has_child.idl"
DATA2="child.idl"
DATA3="complex.idl"
DATA4="points.dat"
DATA5="sampled.points.dat"
DATA6="zipcode.idl"
DATA7="water.idl"
DATA8="areawater.idl"

> output.txt
# > output1.txt
# make clean
# make contain USE_GPU=1 -j
# $BUILD_PATH/contain -s $IDL_PATH/$DATA3 -t $IDL_PATH/$DATA4 -r -g -b 1000000 -l 0.00005 >> output.txt

# make clean
# make contain_polygon USE_GPU=1 -j
# $BUILD_PATH/contain_polygon -s $IDL_PATH/$DATA1 -t $IDL_PATH/$DATA2 -r -g >> output.txt

# make clean
# make within USE_GPU=1  -j
# $BUILD_PATH/within -s $IDL_PATH/$DATA2 -t $IDL_PATH/$DATA4 -r -h -g -b 1000000 -l 0.0002 >> output.txt

make clean
make within_polygon USE_GPU=1 USE_RT=1 -j
$BUILD_PATH/within_polygon -s $IDL_PATH/$DATA2 -t $IDL_PATH/$DATA2 -r -h -g -b 2000000 -l 0.1 > output.txt

# make clean
# make intersection USE_GPU=1 USE_RT=1 -j
# $BUILD_PATH/intersection -s $IDL_PATH/$DATA1 -t $IDL_PATH/$DATA2 -r -g >> output1.txt
# for i in {1..10};
# do
# $BUILD_PATH/intersection -s $DATA_PATH/$DATA6 -t $DATA_PATH/$DATA8 -r -g -b 1000000 -l 0.008 >> output1.txt
# $BUILD_PATH/intersection -s inputA.idl -t inputB.idl -r -g -n 1 >> output1.txt
# done



# CPU
# make clean
# make intersection_cpu USE_RT=1  -j
# $BUILD_PATH/intersection_cpu -s $IDL_PATH/$DATA1 -t $IDL_PATH/$DATA2 -r >> output.txt
# $BUILD_PATH/intersection_cpu -s $DATA_PATH/$DATA6 -t $DATA_PATH/$DATA8 -r -b 1000000 -l 0.00005 >> output.txt
# $BUILD_PATH/intersection_cpu -s inputA.idl -t inputB.idl -r -n 1 >> output.txt