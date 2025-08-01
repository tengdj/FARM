#!/bin/bash
for i in $(seq 0 1 9999)
do
   /home/teng/git/IDEAL/build/contain -s /gisdata/ideal/part/polygons/$i.dat -t /gisdata/ideal/part/points/$i.dat -r -v 10
done

for i in $(seq 0 1 9999)
do
   /home/teng/git/IDEAL/build/contain -s /gisdata/ideal/part/polygons/$i.dat -t /gisdata/ideal/part/points/$i.dat -q -v 10
done
