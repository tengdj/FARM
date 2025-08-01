#!/bin/bash
for i in $(seq 0 1 $2)
do
   ../build/triangulate -s $1 -v "$i"
done
