#!/bin/bash
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

if [ ! -f $4 ]
then 
    echo "error: CHECKPOINT_PATH=$4 is not a file"
    exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

echo "start evaluation for device $DEVICE_ID"
python eval.py --net=$1 --dataset=$2 --dataset_path=$3 --checkpoint_path=$4 &> eval.log &