#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for i in "${arr[@]}";
do
    echo "Running CIFAR. Anomaly Class: $i "
    python train.py --dataset cifar10 --isize 32 --niter 30 --abnormal_class $i --model skipganomaly --save_test_images --batchsize 64 --name cifar10_2
done
exit 0
