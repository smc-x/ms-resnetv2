#!/usr/bin/env bash

air_path=$1
aipp_cfg_path=$2
om_path=$3

atc --model="$air_path" \
--framework=1 \
--output="$om_path" \
--input_format=NCHW --input_shape="actual_input_1:1,3,32,32" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf="$aipp_cfg_path" \
--output_type=FP32
