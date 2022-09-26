#!/bin/bash
if [ $# != 3 ]
then
  echo "Usage: bash run_standalone_train.sh [resnetv2_50|resnetv2_101|resnetv2_152] [cifar10|cifar100] [DATASET_PATH]"
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
    echo "error: DATASET_PATH=$4 is not a directory"
    exit 1
fi

ulimit -u unlimited
export DEVICE_ID=2
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

echo "start training for device $DEVICE_ID"
python train.py --net $1 --dataset $2 --device_num=$DEVICE_NUM --device_target="GPU" --dataset_path $3 &> log.$DEVICE_ID &
