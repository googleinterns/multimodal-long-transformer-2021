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
# limitations under the License.

import json

import tensorflow as tf

import utils


txt_info = dict()

path = f'gs://mmt/raw_data/fashion_gen/full_train_info.txt'
txt_info['train'] = utils.get_txt_info(path)

path = f'gs://mmt/raw_data/fashion_gen/full_valid_info.txt'
txt_info['valid'] = utils.get_txt_info(path)

# Using local paths for image files will be faster.
basename = 'gs://mmt/raw_data/fashion_gen/extracted_{}_images/{}.png'
tfrecord_basename = ('gs://mmt/fashion_gen/split/'
                     'fashion_gen.{}.recordio-{:05d}-of-{:05d}')

input_meta_filename = 'gs://mmt/fashion_gen/fashion_gen_meta_data'

meta_data = {
    'processor_type': 'fashion_gen',
    'max_seq_length': 512,
    'task_type': 'mmt_classification',
    'train_data_size': len(txt_info['train']),
    'eval_data_size': len(txt_info['valid'])
}

with tf.io.gfile.GFile(input_meta_filename, 'w') as f:
  json.dump(meta_data, f, indent=4)

for split, num_records in [('valid', 8), ('train', 128)]:

  num_examples = len(txt_info[split])
  num_examples_per_record = num_examples // num_records

  print(f'# examples per record in {split}: {num_examples_per_record}')

  record_idx = 0

  # Converts valid to val to align naming of other datasets.
  _split = {'train': 'train', 'valid': 'val'}[split]
  tfrecord_filename = tfrecord_basename.format(_split, record_idx, num_records)
  print(f'  Write to {tfrecord_filename}.')
  writer = tf.io.TFRecordWriter(tfrecord_filename)

  for i, (image_id, txt) in enumerate(txt_info[split].items(), start=1):

    img_path = basename.format(split, image_id)
    im = tf.io.gfile.GFile(img_path, 'rb').read()
  
    tf_ex = utils.image_example(im, txt)

    is_last_record = record_idx == (num_records - 1)
    if i % num_examples_per_record == 0 and not is_last_record:
      writer.close()
      record_idx += 1
      tfrecord_filename = tfrecord_basename.format(_split,
                                                   record_idx,
                                                   num_records)
      print(f'  Write to {tfrecord_filename}.')
      writer = tf.io.TFRecordWriter(tfrecord_filename)

    writer.write(tf_ex.SerializeToString())

    if i % 1000 == 0:
      print(f'Processing {split} tf example: {i}')

  writer.close()
