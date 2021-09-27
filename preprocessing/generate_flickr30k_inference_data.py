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

from absl import app
from absl import flags

import json
import os

import tensorflow as tf

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_files',
    default=None,
    help='Input tfrecord files.')

flags.DEFINE_string(
    'eval_data_dir',
    default=None,
    help='The directory for output evaluation sets.')

name_to_features = {
    'image/key': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'caption/tokenized_text': tf.io.FixedLenSequenceFeature(shape=[],
                                                            dtype=tf.string,
                                                            allow_missing=True),
}

def main(_):

  NUM_SHARDS = 1
  # Uses topk images to generate smaller inference set.
  TOPK_IMAGES = 100
  
  TFRECORD_BASENAME = '{}/flickr30k.{}.{}.recordio-{:05d}-of-{:05d}'
  input_meta_data = {'max_seq_length': 512}
  max_num_examples = {
      'val': {'image': 1014, 'text': 5070},
      'test': {'image': 1000, 'text': 5000},
  }
  
  parse_fn = utils.get_parse_single_example_fn(name_to_features)
  
  def process_split(split):
    print(f'Start to process {split}.')
  
    img_shard_idx = 0
    tfrecord_filename = TFRECORD_BASENAME.format(
        FLAGS.eval_data_dir, split, 'image', img_shard_idx, NUM_SHARDS)
    img_writer = tf.io.TFRecordWriter(tfrecord_filename)
  
    txt_shard_idx = 0
    tfrecord_filename = TFRECORD_BASENAME.format(
        FLAGS.eval_data_dir, split, 'text', txt_shard_idx, NUM_SHARDS)
    txt_writer = tf.io.TFRecordWriter(tfrecord_filename)
    
    dataset = tf.data.Dataset.list_files(FLAGS.input_files.format(split))
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
    image_key_to_index = dict()
    text_key_to_index = dict()
    num_img_per_shard = TOPK_IMAGES // NUM_SHARDS
    num_txt_per_shard = max_num_examples[split]['text'] // NUM_SHARDS
  
    ii = 0
    for i, ex in enumerate(dataset, start=1):
      image_key = ex['image/key'].numpy()
  
      if len(image_key_to_index) < TOPK_IMAGES:
  
        image_index = image_key_to_index.get(image_key, len(image_key_to_index))
        image_key_to_index[image_key] = image_index
        img_int_features = {'image_index': image_index}
        img_string_features = {'image_key': image_key}
        tf_ex = utils.image_example(ex['image/encoded'], 
                                    img_string_features,
                                    img_int_features)
        tf_ex_string = tf_ex.SerializeToString()
  
        is_last_shards = img_shard_idx == NUM_SHARDS - 1
        if (len(image_key_to_index) % num_img_per_shard == 0 and 
                not is_last_shards):
          img_writer.close()
          img_shard_idx += 1
          tfrecord_filename = TFRECORD_BASENAME.format(
              FLAGS.eval_data_dir, split, 'image', img_shard_idx, NUM_SHARDS)
          img_writer = tf.io.TFRecordWriter(tfrecord_filename)
        img_writer.write(tf_ex_string)
  
      # Text
      # 5 captions per image
      captions = ex['caption/tokenized_text'].numpy()
      for idx, caption in enumerate(captions):
        text_key = f'{image_key.decode("utf-8")}_{idx}'.encode()
        text_index = text_key_to_index.get(text_key, len(text_key_to_index))
        text_key_to_index[text_key] = text_index
        gt_image_index = image_key_to_index.get(image_key, -1)
        txt_string_features = {
            'caption': caption,
            'text_key': text_key,
        }
        txt_int_features = {
            'text_index': text_index,
            'gt_image_index': gt_image_index,
        }
        tf_ex = utils.text_example(txt_string_features, txt_int_features)
        tf_ex_string = tf_ex.SerializeToString()
  
        is_last_shards = txt_shard_idx == NUM_SHARDS - 1
        if (len(text_key_to_index) % num_txt_per_shard == 0 and 
                not is_last_shards):
          txt_writer.close()
          txt_shard_idx += 1
          tfrecord_filename = TFRECORD_BASENAME.format(
              FLAGS.eval_data_dir, split, 'text', txt_shard_idx, NUM_SHARDS)
          txt_writer = tf.io.TFRecordWriter(tfrecord_filename)
        txt_writer.write(tf_ex_string)
  
      if i % 100 == 0:
        print(f'  Process {i}')
  
    split_input_meta_data = {
        f'{split}_image_input_path':
          f'{FLAGS.eval_data_dir}/flickr30k.{split}.image.recordio-*',
        f'{split}_text_input_path':
          f'{FLAGS.eval_data_dir}/flickr30k.{split}.text.recordio-*',
        f'{split}_num_image_examples': len(image_key_to_index),
        f'{split}_num_text_examples': len(text_key_to_index),
    }
    return split_input_meta_data
  
  for split in ['val', 'test']:
    split_input_meta_data = process_split(split)
    input_meta_data.update(split_input_meta_data)
  
  filename = os.path.join(FLAGS.eval_data_dir, 'input_meta_data')
  with tf.io.gfile.GFile(filename, 'w') as f:
    json.dump(input_meta_data, f, indent=4)


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_files', 'eval_data_dir'])
  app.run(main)
