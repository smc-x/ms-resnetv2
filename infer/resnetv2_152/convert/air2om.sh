#!/usr/bin/env bash

model_path=$1
output_model_name=$2

/usr/local/Ascend/atc/bin/atc --model=$model_path \
        --framework=1 \
        --output=$output_model_name \
        --input_format=NCHW \
        --soc_version=Ascend310 \
        --output_type=FP32
