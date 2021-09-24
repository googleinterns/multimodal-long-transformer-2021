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
input_meta_data=$2
output_dir=$3
predict_split=$4
ckpt_path=$5

if [ $# -ne 5 ];
then
  echo $0 [config_file] [input_meta_data] [output_dir] [predict_split] [ckpt_path]
  exit 1
fi

batch_size=2048

TPU_NAME=v3-8-sw25-1
TPU_ZONE=us-central1-a

python predict3.py \
  --distribution_strategy tpu \
  --tpu $TPU_NAME \
  --tpu_zone $TPU_ZONE \
  --config_file $config_file \
  --predict_global_batch_size $batch_size \
  --input_meta_data_path $input_meta_data \
  --test_output_dir $output_dir \
  --predict_split $predict_split \
  --init_checkpoint $ckpt_path
