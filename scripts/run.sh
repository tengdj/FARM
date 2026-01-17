#!/bin/bash
DATA_PATH="/home/data"
IDL_PATH="/home/data/idl"
EXP_PATH="/home/data/exp"
BUILD_PATH="/home/sigmod/FARM/build"
DATA1="lakes.idl"
DATA2="lakes_normal.idl"
DATA3="complex.idl"
DATA4="complex_normal.idl"
DATA5="oligo1_div.idl"
DATA6="oligo2_div.idl"

> output.txt
> output1.txt
# make clean
# make intersect USE_GPU=1 -j
# for val in $(awk 'BEGIN{for(i=0.0;i<=1.0;i+=0.1) printf "%.1f ", i}')
# do
# $BUILD_PATH/intersect -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA2 -r -h -g -m $val >> output.txt
# done
# $BUILD_PATH/intersect -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA1 -r -h -g >> output.txt
# $BUILD_PATH/intersect -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h -g -b 1000000 >> output.txt

# make clean
# make within_polygon USE_GPU=1 -j
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA4 -r -h -g >> output.txt
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA4 -r -h -g -m 0.1 >> output.txt

# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA1 -r -h -g > output.txt
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA3 -r -h -g -b 1000000 >> output.txt
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h -g -b 1000000 >> output.txt
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA6 -r -h -g -b 1000000 >> output.txt
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA4 -r -h -g -b 100000 -u 4 >> output.txt


# for val in $(awk 'BEGIN{for(i=0.0;i<=1.0;i+=0.1) printf "%.1f ", i}')
# do
# $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA4 -r -h -g -m $val >> output.txt
# done
# for ((i=0; i<=16; i++))
# do
#     para=$((2**i))
#     $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA1 -r -h -g --NLow $para >> output.txt
# done
# for ((i=1; i<=10; i++))
# do
#     para=$((2**i))
#     $BUILD_PATH/within_polygon -s $EXP_PATH/$DATA4 -r -h -g -b 100000 -u $para >> output.txt
# done

make clean
make intersection USE_GPU=1 -j
$BUILD_PATH/intersection -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h -g -b 1000000 -v 80 >> output.txt
# $BUILD_PATH/intersection -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA2 -r -h -g >> output.txt



# CPU

# make clean
# make intersect_cpu -j
# $BUILD_PATH/intersect_cpu -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA1 -r -h >> output1.txt
# $BUILD_PATH/intersect_cpu -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h >> output1.txt

# make clean
# make intersection_cpu  -j
# $BUILD_PATH/intersection_cpu -s $EXP_PATH/$DATA3 -t $EXP_PATH/$DATA1 -r -h > output1.txt
# $BUILD_PATH/intersection_cpu -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h > output1.txt

# make clean
# make within_polygon_cpu -j
# $BUILD_PATH/within_polygon_cpu -s $EXP_PATH/$DATA5 -t $EXP_PATH/$DATA6 -r -h >> output1.txt
# $BUILD_PATH/within_polygon_cpu -s $EXP_PATH/$DATA3 -r -h >> output1.txt
# $BUILD_PATH/within_polygon_cpu -s $EXP_PATH/$DATA1 -r -h >> output1.txt