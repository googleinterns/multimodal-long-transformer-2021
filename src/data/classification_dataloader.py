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

"""Loads dataset for the Mmt classification task."""

import json
from typing import Optional

import dataclasses
import tensorflow as tf
import tensorflow_text as tf_text
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

from data import data_utils
from data import configs


@dataclasses.dataclass
class MmtClassificationDataConfig(configs.MmtDataConfig):
  """Data config for Mmt classification task."""
  negative_positive_ratio: int = 1
  pos_weight: float = 1.0


@data_loader_factory.register_data_loader_cls(MmtClassificationDataConfig)
class MmtClassificationDataLoader(data_loader.DataLoader):
  """A class to load dataset for mmt image-text matching classification task."""

  def __init__(self, params):
    """Inits `MmtClassificationDataLoader` class.

    Args:
      params: A `MmtDataConfig` object.

    """
    self._params = params
    self._image_data_field = params.image_data_field
    self._image_key_field = params.image_key_field
    self._text_special_token_field_dict = json.loads(
        params.text_special_token_field_dict)
    self.name_to_features = self._name_to_features()

  def _name_to_features(self):

    name_to_features = {
        self._image_data_field: tf.io.FixedLenFeature([], tf.string),
        self._image_key_field: tf.io.FixedLenFeature([], tf.string),
    }
    for k in self._text_special_token_field_dict.keys():
      name_to_features[k] = tf.io.FixedLenFeature([], tf.string,
                                                  default_value='')

    return name_to_features

  def load(self,
           input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""

    input_keys = ['word_ids',
                  'segment_ids',
                  'patch_embeddings',
                  'att_mask',
                  'relative_att_ids']

    label_keys = ['itm_label_ids',
                  'itm_label_weights',
                  'itm_pos_weights']

    keep_feature_keys = [*input_keys, *label_keys]

    config = self._params
    tokenizer = tf_text.BertTokenizer(config.vocab_filename,
                                      lower_case=True,
                                      preserve_unused_token=True,
                                      token_out_type=tf.int32)

    use_rand_aug = config.use_rand_aug
    is_training = config.is_training
    input_patterns = config.input_path.split(',')
    data_utils.check_input_patterns(input_patterns)

    batch_size_per_replica = input_context.get_per_replica_batch_size(
        config.global_batch_size)

    dataset = tf.data.Dataset.list_files(input_patterns,
                                         shuffle=is_training,

                                         seed=config.seed)
    if is_training:
      dataset = dataset.repeat()

      # We set shuffle buffer to exactly match total number of
      # training files to ensure that training data is well shuffled.
      input_files = []
      for input_pattern in input_patterns:
        input_files.extend(tf.io.gfile.glob(input_pattern))
      dataset = dataset.shuffle(len(input_files))

    if (input_context and
        input_context.num_input_pipelines > 1):
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=config.cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Sets shuffle buffer size larger than # records in a single shard.
    if is_training:
      dataset = dataset.shuffle(4096)

    # Gets mapping functions.
    decode_fn = data_utils.get_decode_fn(self.name_to_features,
                                         tokenizer,
                                         config,
                                         merge_dims=True,
                                         is_training=is_training,
                                         keep_unnormalized_patch_embeddings=False,
                                         use_rand_aug=use_rand_aug)

    # Makes sure batch size for matching function is also larger than
    # negative_positive_ratio to avoid creating false negatives examples.
    max_shift = config.negative_positive_ratio + config.min_shift
    # Use enough batch size for creating negative examples.
    batch_size_in_matching_fn = ((max_shift // batch_size_per_replica + 2) *
                                 batch_size_per_replica)
    matching_fn = data_utils.get_matching_fn(config,
                                             batch_size_in_matching_fn,
                                             config.negative_positive_ratio)

    word_ids_fn = data_utils.get_word_ids_fn(config)
    split_features_fn = data_utils.get_split_features_fn(input_keys, label_keys)

    # TODO(roylu): relative_pos_max_distance and relative_att_num_core_layers 
    # should be defined in the model config. No model config is passed into this
    # function.
    add_side_input_features_fn = data_utils.get_add_side_input_features_fn(
        config,
        config.relative_pos_max_distance,
        config.relative_att_num_core_layers)
    pop_fn = data_utils.get_pop_fn(keep_feature_keys)

    dataset = dataset.map(
        decode_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Creates in-batch negatives. Thus, we batch and unbatch.
    # TODO(roylu): revisit here. `drop_remainder=False` will raise an issue.
    dataset = dataset.batch(batch_size_in_matching_fn, drop_remainder=True)
    dataset = dataset.map(
        matching_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()

    dataset = dataset.map(
        add_side_input_features_fn,
        tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        word_ids_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(
        pop_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Makes sure that we will have batches with mixing labels for ITM task.
    if is_training:
      dataset = dataset.shuffle(4096, seed=config.seed)

    dataset = dataset.map(
        split_features_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size_per_replica, drop_remainder=is_training)
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
