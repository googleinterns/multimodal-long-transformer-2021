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

import functools

import tensorflow as tf

import input_utils


def create_pretrain_dataset(input_patterns,
                            input_config,
                            model_config,
                            batch_size,
                            tokenizer,
                            is_training=True,
                            input_pipeline_context=None,
                            output_fake_labels=True):
  """Creates input dataset from (tf)records files for pretraining."""

  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature(
        [], tf.string, default_value='')

  for input_pattern in input_patterns:
    if not tf.io.gfile.glob(input_pattern):
      raise ValueError('%s does not match any files.' % input_pattern)

  dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)
  if is_training:
    dataset = dataset.repeat()

    # We set shuffle buffer to exactly match total number of
    # training files to ensure that training data is well shuffled.
    input_files = []
    for input_pattern in input_patterns:
      input_files.extend(tf.io.gfile.glob(input_pattern))
    dataset = dataset.shuffle(len(input_files))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=8,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(100)

  decode_fn = input_utils.get_decode_fn(tokenizer,
                                        input_config,
                                        model_config,
                                        is_training=is_training)

  masking_fn = input_utils.get_masking_fn(tokenizer, input_config)

  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size, drop_remainder=is_training)

  dataset = dataset.map(
      masking_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.map(
        functools.partial(input_utils.add_side_input_features,
                          input_config, model_config),
        tf.data.experimental.AUTOTUNE)

  if output_fake_labels:
    dataset = dataset.map(
        input_utils.add_fake_labels,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
