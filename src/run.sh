make clean
make dist_point_polygon USE_GPU=1 -j
../build/dist_point_polygon -s /home/qmh/data/test_polygon.idl -t /home/qmh/data/test_point.dat -r -n 1 > output.txt

