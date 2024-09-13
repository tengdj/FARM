make clean
make distance_test USE_GPU=1 -j
../build/distance_test -s /home/qmh/data/dist.csv -t /home/qmh/data/dist2.csv -r

