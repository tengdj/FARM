#!/bin/bash
DATA_PATH="/home/qmh/data"
IDL_PATH="/home/qmh/data/idl"
EXP_PATH="/home/qmh/data/exp"
BUILD_PATH="/home/qmh/IDEAL/build"
DATA1="lakes.idl"
DATA2="pathology.idl"
DATA3="complex.idl"

> output.txt
# > output1.txt
# make clean
# make contain USE_GPU=1 -j
# $BUILD_PATH/contain -s $IDL_PATH/$DATA3 -t $IDL_PATH/$DATA4 -r -g -b 1000000 -l 0.00005 >> output.txt

# make clean
# make contain_polygon USE_GPU=1 -j
# $BUILD_PATH/contain_polygon -s $IDL_PATH/$DATA1 -t $IDL_PATH/$DATA2 -r -g >> output.txt

# make clean
# make within USE_GPU=1 -j
# $BUILD_PATH/within -s $IDL_PATH/$DATA2 -t $IDL_PATH/$DATA4 -r -h -g -b 1000000 -l 0.0002 >> output.txt

make clean
make within_polygon USE_GPU=1 USE_RT=1 -j
$BUILD_PATH/within_polygon -s $EXP_PATH/$DATA1 -t $EXP_PATH/$DATA1 -r -h -g -b 2000000 -l 0.1 > output.txt

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
# $BUILD_PATH/intersection_cpu -s $DATA_PATH/$DATA6 -t $DATA_PATH/$DATA8 -r -b 1000000 -l 0.00005 >> output1.txt
# $BUILD_PATH/intersection_cpu -s inputA.idl -t inputB.idl -r -n 1 >> output.txt

# make clean
# make within_polygon_cpu USE_RT=1 -j
# $BUILD_PATH/within_polygon_cpu -s $IDL_PATH/$DATA9 -t $IDL_PATH/$DATA9 -r -h -b 2000000 -l 0.1 > output1.txt