#!/bin/bash
if [ $# != 3 ]
then
  echo "Usage: bash run_distribute_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|cifar100] [DATASET_PATH]"
  exit 1
fi

if [ $1 != "resnetv2_50" ] && [ $1 != "resnetv2_101" ] && [ $1 != "resnetv2_152" ]
then 
  echo "error: the selected net is neither resnetv2_50 nor resnetv2_101 and resnetv2_152"
  exit 1
fi

if [ $2 != "cifar10" ] && [ $2 != "cifar100" ]
then 
    echo "error: the selected dataset is neither cifar10 nor cifar100"
    exit 1
fi

if [ ! -d $3 ]
then
    echo "error: DATASET_PATH=$3 is not a directory"
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8         
export RANK_SIZE=8     

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
python train.py --net=$1 --dataset=$2 --run_distribute=True \
--device_num=$DEVICE_NUM --device_target="GPU" --dataset_path=$3 &> log &