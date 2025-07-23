#!/bin/bash
DATA_PATH="/home/qmh/data"
IDL_PATH="/home/qmh/data/idl"
EXP_PATH="/home/qmh/data/exp"
BUILD_PATH="/home/qmh/IDEAL/build"
DATA1="lakes.idl"
DATA2="lakes_normal.idl"
DATA3="complex.idl"
DATA4="complex_normal.idl"
DATA5="oligo1_div.idl"
DATA6="oligo2_div.idl"
DATA7="zipcode.idl"
DATA8="areawater.idl"

# > output.txt
# > output1.txt
# make clean
# make intersect USE_GPU=1 -j
# $BUILD_PATH/intersect -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA1 -r -h -g > output3.txt

# make clean
# make within_polygon USE_GPU=1 -j
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA1 -r -h -g > output.txt

# make clean
# make intersection USE_GPU=1 -j
# $BUILD_PATH/intersection -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h -g -b 1000000 > output.txt



# CPU

# make clean
# make intersect_cpu -j
# $BUILD_PATH/intersect_cpu -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA1 -r -h > output1.txt

make clean
make intersection_cpu  -j
$BUILD_PATH/intersection_cpu -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h > output1.txt

# make clean
# make within_polygon_cpu -j
# $BUILD_PATH/within_polygon_cpu -s $EXP_PATH/$DATA1 -r -h > output1.txt

