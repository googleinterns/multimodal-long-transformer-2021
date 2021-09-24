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

"""Loads dataset for the Mmt prediction."""

import dataclasses
import json
from typing import Optional

import tensorflow as tf
import tensorflow_text as tf_text
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

from data import data_utils
from data import configs


@dataclasses.dataclass
class MmtRetrievalDataConfig(configs.MmtDataConfig):
  """Data config for Mmt retrieval task."""
  # If we store image and text records seperately, we provide image and text
  # record paths separately and enumerate all possible combinations on the fly.
  image_input_path: str = ''
  text_input_path: str = ''
  num_image_examples: int = 0
  num_text_examples: int = 0

  negative_positive_ratio: int = 1
  pos_weight: float = 1.0
  drop_remainder: bool = False
  include_image_text_index: bool = True


@data_loader_factory.register_data_loader_cls(MmtRetrievalDataConfig)
class MmtRetrievalDataLoader(data_loader.DataLoader):
  """A class to load dataset for mmt retrieval task."""

  def __init__(self, params):
    """Inits `MmtRetrievalDataLoader` class.

    Args:
      params: A `MmtDataConfig` object.

    """
    self._params = params
    self._image_data_field = params.image_data_field
    self._text_special_token_field_dict = json.loads(params.text_special_token_field_dict)
    self.image_name_to_features = self._image_name_to_features()
    self.text_name_to_features = self._text_name_to_features()

  def _image_name_to_features(self):
    name_to_features = {
        'image_index': tf.io.FixedLenFeature([], tf.int64),
        self._image_data_field: tf.io.FixedLenFeature([], tf.string)
    }
    return name_to_features

  def _text_name_to_features(self):
    name_to_features = {
        'text_index': tf.io.FixedLenFeature([], tf.int64),
        'gt_image_index': tf.io.FixedLenFeature([], tf.int64),
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

    label_keys = ['label_ids',
                  'label_weights']

    keep_feature_keys = []
    keep_feature_keys.extend(label_keys)
    keep_feature_keys.extend(input_keys)

    config = self._params

    tokenizer = tf_text.BertTokenizer(config.vocab_filename,
                                      lower_case=True,
                                      preserve_unused_token=True,
                                      token_out_type=tf.int32)

    # This dataloader is currently for prediction only. However, it could be
    # modified for training.
    is_training = config.is_training
    index_keys = ['image_index', 'text_index', 'gt_image_index']
    input_keys.extend(index_keys)
    keep_feature_keys.extend(index_keys)

    batch_size_per_replica = input_context.get_per_replica_batch_size(
        config.global_batch_size)

    drop_remainder = config.drop_remainder

    if config.input_path:
      # Image-text pairs are already provided in records.
      input_patterns = config.input_path.split(',')
      data_utils.check_input_patterns(input_patterns)
      dataset = tf.data.Dataset.list_files(input_patterns,
                                           shuffle=is_training,
                                           seed=config.seed)

      dataset = dataset.interleave(
          tf.data.TFRecordDataset,
          cycle_length=config.cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      name_to_features = {**self.image_name_to_features,
                          **self.text_name_to_features}
      decode_fn = data_utils.get_decode_fn(name_to_features,
                                           tokenizer,
                                           config,
                                           merge_dims=True,
                                           is_training=is_training)
      dataset = dataset.map(
          decode_fn,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      # Creates image-text pairs from image and text records.
      image_input_patterns = config.image_input_path.split(',')
      text_input_patterns = config.text_input_path.split(',')
      data_utils.check_input_patterns(image_input_patterns)
      data_utils.check_input_patterns(text_input_patterns)

      # No need to shuffle files here because we will enumerate all combinations.
      image_dataset = tf.data.Dataset.list_files(image_input_patterns,
                                                 shuffle=is_training,
                                                 seed=config.seed)
      text_dataset = tf.data.Dataset.list_files(text_input_patterns,
                                                shuffle=is_training,
                                                seed=config.seed)

      image_dataset = image_dataset.interleave(
          tf.data.TFRecordDataset,
          cycle_length=config.cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      text_dataset = text_dataset.interleave(
          tf.data.TFRecordDataset,
          cycle_length=config.cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Gets mapping functions.
      image_decode_fn = data_utils.get_decode_fn(self.image_name_to_features,
                                                 tokenizer,
                                                 config,
                                                 merge_dims=True,
                                                 is_training=is_training,
                                                 decode_image=True,
                                                 decode_text=False)

      text_decode_fn = data_utils.get_decode_fn(self.text_name_to_features,
                                                tokenizer,
                                                config,
                                                merge_dims=True,
                                                is_training=is_training,
                                                decode_image=False,
                                                decode_text=True)

      image_dataset = image_dataset.map(
          image_decode_fn,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      text_dataset = text_dataset.map(
          text_decode_fn,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Enumerates all possible combinations of image-text pairs.
      # For example, image = [a, b, c]; text = [x, y].
      # All combinations = [(a, x), (a, y), (b, x), (b, y), (c, x), (c, y)].
      dataset = text_dataset.interleave(
          lambda x: image_dataset.interleave(
              lambda y: data_utils.combine_image_text(x, y),
              num_parallel_calls=tf.data.experimental.AUTOTUNE),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Gets the label of the image-text pair by matching image_index in
    # image_dataset and gt_image_indx in text_dataset.
    get_label = data_utils.get_retrieval_label_fn(self._params.pos_weight)
    dataset = dataset.map(
        get_label,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shards image pairs after enumerate all possible combinations.
    if input_context and input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    word_ids_fn = data_utils.get_word_ids_fn(config)

    # TODO(roylu): relative_pos_max_distance and relative_att_num_core_layers 
    # should be defined in the model config. No model config is passed into this
    # function.
    add_side_input_features_fn = data_utils.get_add_side_input_features_fn(
        config,
        config.relative_pos_max_distance,
        config.relative_att_num_core_layers)
    pop_fn = data_utils.get_pop_fn(keep_feature_keys)
    split_features_fn = data_utils.get_split_features_fn(input_keys, label_keys)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(4096, seed=config.seed)

    dataset = dataset.map(
        word_ids_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        add_side_input_features_fn,
        tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        pop_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        split_features_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size_per_replica,
                            drop_remainder=drop_remainder)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
