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

config_file=$1
dataset=$2
inference_name=$3
predict_split=$4
model_dir=$5
ckpt_name=$6

if [ $# -ne 6 ];
then
  echo $0 [config_file] [dataset] [inference_name] [predict_split] [model_dir] [ckpt_name]
  exit 1
fi

batch_size=2048
output_dir=$model_dir/results/$ckpt_name/$inference_name/$predict_split
input_meta_data=gs://mmt/$dataset/inference_data/$inference_name/input_meta_data
ckpt_path=$model_dir/$ckpt_name

TPU_NAME=v3-8-sw25-1
TPU_ZONE=us-central1-a

python predict.py \
  --distribution_strategy tpu \
  --tpu $TPU_NAME \
  --tpu_zone $TPU_ZONE \
  --config_file $config_file \
  --predict_global_batch_size $batch_size \
  --input_meta_data_path $input_meta_data \
  --test_output_dir $output_dir \
  --predict_split $predict_split \
  --init_checkpoint $ckpt_path
