make clean
make within_polygon USE_GPU=1 -j
../build/within_polygon -s /home/qmh/data/test_simple.idl -r -v 10 > output.txt

