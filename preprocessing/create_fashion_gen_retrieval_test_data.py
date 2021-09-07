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

import pandas as pd
import tensorflow as tf

import utils


"""
The first 2 rows of txt_info file.

86605,86605_0,JACKETS & COATS,7,DENIM JACKETS,39,Denim-like jogg jacket in blue. Fading and whiskering throughout. Spread collar. Copper tone button closures at front. Flap pockets at chest with metallic logo plaque. Seam pockets at sides. Cinch tabs at back waistband. Single button sleeve cuffs. Tone on tone stitching.
86605,86605_1,JACKETS & COATS,7,DENIM JACKETS,39,Denim-like jogg jacket in blue. Fading and whiskering throughout. Spread collar. Copper tone button closures at front. Flap pockets at chest with metallic logo plaque. Seam pockets at sides. Cinch tabs at back waistband. Single button sleeve cuffs. Tone on tone stitching.
"""

txt_info_filename = f'gs://mmt/raw_data/fashion_gen/full_valid_info.txt'
i2t_meta_filename = f'gs://mmt/fashion_gen/metadata/fashion_bert_i2t_test.csv'
t2i_meta_filename = f'gs://mmt/fashion_gen/metadata/fashion_bert_t2i_test.csv'

num_records = 8
# Using local paths for image files will be faster.
# {} will be an image_id.
basename = 'gs://mmt/raw_data/fashion_gen/extracted_valid_images/{}.png'

tfrecord_basename = ('gs://mmt/fashion_gen/inference_data/{}/'
                     'fashion_gen.fashion_bert.valid.recordio-{:05d}-of-{:05d}')

# {} will be the name of the task, i2t or t2i.
meta_data_basename = 'gs://mmt/fashion_gen/inference_data/{}/input_meta_data'
val_input_path_basename = ('gs://mmt/fashion_gen/inference_data/{}/'
                           'fashion_gen.fashion_bert.valid.recordio-*')


def get_txt_info():
  """Gets metadata of each image_id (image file).
  
  Returns:
  txt_info: A dictionary. key is the image_id and value is a dictionary of the
  corresponding metadata.
  
  """
  txt_info = dict()
  with tf.io.gfile.GFile(txt_info_filename, 'r') as f:
    for i, line in enumerate(f, start=1):
      line = line.split('\x01')
  
      image_main_id = line[0]
      image_id = line[1]
      category = line[2]
      sub_category = line[4]
      description = line[6] 
  
      txt_info[image_id] = {
        'image_main_id': image_main_id.encode(),
        'image_id': image_id.encode(),
        'category': category.encode(),
        'sub_category': sub_category.encode(),
        'original_description': description.encode(),
      }
  
      if i % 10000 == 0:
        print(f'Read txt info: {i}')
  return txt_info

dtype = {
    'text_prod_id': str,  
    'image_prod_id': str, 
    'prod_img_id': str,   
    'image_id': str,      
    'image_index': int,   
    'text_index': int,    
    'gt_image_index': int,
}
with tf.io.gfile.GFile(i2t_meta_filename, 'r') as f:
  i2t_df = pd.read_csv(f, dtype=dtype)
with tf.io.gfile.GFile(t2i_meta_filename, 'r') as f:
  t2i_df = pd.read_csv(f, dtype=dtype)

txt_info = get_txt_info()

meta_data = {
    'processor_type': 'fashion_gen',
    'max_seq_length': 512,
    'task_type': 'mmt_retrieval',
}

for task, df in [('i2t', i2t_df), ('t2i', t2i_df)]:
  num_examples = len(df)
  num_examples_per_record = num_examples // num_records

  record_idx = 0
  tfrecord_filename = tfrecord_basename.format(task,
                                               record_idx,
                                               num_records)
  writer = tf.io.TFRecordWriter(tfrecord_filename)

  for i, (_, row) in enumerate(df.iterrows(), start=1):
    # Some description in fashion bert data does not exactly match description
    # in txt_info. The difference is very tiny, such as an extra quote symbol.<Plug>_
    # We keep both and set the one in txt_info as original_description.
    fashion_bert_txt = row.desc
    string_dict = txt_info[row.image_id]
    string_dict['description'] = fashion_bert_txt.encode()

    img_path = basename.format(row.image_id)
    im = tf.io.gfile.GFile(img_path, 'rb').read()

    int_dict = {}
    int_dict['image_index'] = row.image_index
    int_dict['text_index'] = row.text_index
    int_dict['gt_image_index'] = row.gt_image_index
  
    tf_ex = utils.image_example(im, string_dict, int_dict)

    is_last_record = record_idx == (num_records - 1)
    if i % num_examples_per_record == 0 and not is_last_record: 
      writer.close()
      record_idx += 1
      tfrecord_filename = tfrecord_basename.format(task,
                                                   record_idx,
                                                   num_records)
      writer = tf.io.TFRecordWriter(tfrecord_filename)
    writer.write(tf_ex.SerializeToString())

    if i % 1000 == 0:
      print(f'Processing {task} tf example: {i}')

  writer.close()

  meta_data['val_input_path'] = val_input_path_basename.format(task)
  meta_data['val_num_examples'] = len(df)

  with tf.io.gfile.GFile(meta_data_basename.format(task), 'w') as f:
    json.dump(meta_data, f, indent=4)
