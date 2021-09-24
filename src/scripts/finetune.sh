# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

tpu_num=$1
config_file=$2
output_dir=$3

if [ $# -ne 3 ];
then
  echo $0 [tpu_num] [config_file] [output_dir]
  exit 1
fi

PARAMS=runtime.distribution_strategy=tpu
PARAMS=$PARAMS,runtime.mixed_precision_dtype='bfloat16'
PARAMS=$PARAMS,runtime.enable_xla=True

TPU_NAME=v3-8-sw25-${tpu_num}
TPU_ZONE=us-central1-a

EXPERIMENT=mmt/classification

python3 train.py \
 --tpu=$TPU_NAME \
 --tpu_zone=$TPU_ZONE \
 --experiment=$EXPERIMENT \
 --mode=train_and_eval \
 --model_dir=$output_dir \
 --config_file=$config_file \
 --params_override=$PARAMS
