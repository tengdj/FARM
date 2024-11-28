make clean
make distance_test USE_GPU=1 -j
../build/distance_test -s /home/qmh/data/simple_source.idl -t /home/qmh/data/simple_target.idl -r -v 10 -n 1 > output.txt

