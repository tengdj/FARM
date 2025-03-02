make clean
make within_polygon USE_GPU=1 -j
../build/within_polygon -s /home/qmh/data/child.idl -r -g -h -v 10 --batch_size 10000 > output.txt

