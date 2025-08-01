#!/bin/bash

st=0
ed=1000

while [ $st -le 171000 ]
do

echo $st

echo "select count(*) from polygon_target t1, polygon t2 where t2.geog.STContains(t1.geog)=1 and t1.id<$ed and t1.id>=$st"

sqlcmd -S localhost -U SA -P '$Terry08161043' -d tengdb -I -Q "select count(*) from polygon_target t1, polygon t2 where t2.geog.STContains(t1.geog)=1 and t1.id<$ed and t1.id>=$st"
st=$(expr $st + 1000)
ed=$(expr $ed + 1000)

done


