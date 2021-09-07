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

"""Loads dataset for the Mmt pretraining MTM task."""
import json
import dataclasses
from typing import Optional

import tensorflow as tf
import tensorflow_text as tf_text
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory

from data import data_utils
from data import configs


@dataclasses.dataclass
class MmtPretrainDataConfig(configs.MmtDataConfig):
  """Data config for Mmt pretraining task (tasks/mtm)."""

  mlm_use_whole_word: bool = True
  mlm_fraction_to_mask: float = 0.15
  mpp_fraction_to_mask: float = 0.5
  # we set this value to be the same as max_seq_len b/c when applying whole word
  # masking, the number of masked tokens might be larger than the acutally vlaues.
  mlm_max_selections_per_seq: int = 512
  mpp_max_selections_per_seq: int = 98
  output_channel_bits: int = 3 
  input_channels: int =3
  use_patch_mask_token_id: bool = False
  negative_sample_rate: int = 1
  alternate_mlm_mpp: bool = False


@data_loader_factory.register_data_loader_cls(MmtPretrainDataConfig)
class MmtPretrainDataLoader(data_loader.DataLoader):
  """A class to load dataset for Mmt pretraining task."""

  def __init__(self, params):
    """Inits `MmtPretrainDataLoader` class.

    Args:
      params: A `MmtPretrainDataConfig` object.

    """
    self._params = params
    self._image_key = params.image_key
    self._text_key_dict = json.loads(params.text_key_dict)
    self.name_to_features = self._name_to_features()

  def _name_to_features(self):

    name_to_features = {
        self._image_key: tf.io.FixedLenFeature([], tf.string)
    }
    for k in self._text_key_dict.keys():
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
                  'relative_att_ids',
                  'mlm_positions',
                  'mpp_positions']

    label_keys = ['mlm_label_ids',
                  'mlm_label_weights',
                  'mpp_label_ids',
                  'mpp_label_weights',
                  'itm_label_ids',
                  'itm_label_weights']

    keep_feature_keys = [*input_keys, *label_keys]

    config = self._params

    tasks = config.tasks.split(',')

    tokenizer = tf_text.BertTokenizer(config.vocab_filename,
                                      lower_case=True,
                                      preserve_unused_token=True,
                                      token_out_type=tf.int32)

    is_training = config.is_training
    input_patterns = config.input_path.split(',')
    batch_size_per_replica = input_context.get_per_replica_batch_size(
        config.global_batch_size)

    data_utils.check_input_patterns(input_patterns)

    dataset = tf.data.Dataset.list_files(input_patterns,
                                         shuffle=is_training,
                                         seed=config.seed)

    if is_training:
      # We set shuffle buffer to exactly match total number of
      # training files to ensure that training data is well shuffled.
      input_files = []
      for input_pattern in input_patterns:
        input_files.extend(tf.io.gfile.glob(input_pattern))
      dataset = dataset.shuffle(len(input_files), seed=config.seed)

    if (input_context and
        input_context.num_input_pipelines > 1):
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    if is_training:
      dataset = dataset.repeat()

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=config.cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      dataset = dataset.shuffle(100, seed=config.seed)

    decode_fn = data_utils.get_decode_fn(self.name_to_features,
                                          tokenizer,
                                          config,
                                          merge_dims=False,
                                          is_training=is_training)

    dataset = dataset.map(
        decode_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=config.deterministic)

    masking_fn = data_utils.get_masking_fn(tokenizer, config)
    word_ids_fn = data_utils.get_word_ids_fn(config)
    split_features_fn = data_utils.get_split_features_fn(input_keys, label_keys)

    #TODO(roylu): relative_pos_max_distance and relative_att_num_core_layers 
    # should be defined in the model config. No model config is passed into this
    # function.
    add_side_input_features_fn = data_utils.get_add_side_input_features_fn(
        config,
        config.relative_pos_max_distance,
        config.relative_att_num_core_layers)
    pop_fn = data_utils.get_pop_fn(keep_feature_keys)

    dataset = dataset.map(
        masking_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=config.deterministic)

    if 'itm' in tasks:
      matching_fn = data_utils.get_matching_fn(config, batch_size_per_replica)
      #TODO(roylu): revisit here. `drop_remainder=False` will raise an issue.
      dataset = dataset.batch(batch_size_per_replica, drop_remainder=True)
      dataset = dataset.map(
          matching_fn,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=config.deterministic)
      dataset = dataset.unbatch()

    dataset = dataset.map(
        add_side_input_features_fn,
        tf.data.experimental.AUTOTUNE,
        deterministic=config.deterministic)

    dataset = dataset.map(
        word_ids_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(
        pop_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=config.deterministic)

    # Makes sure that we will have batches with mixing labels.
    if 'itm' in tasks and is_training:
      dataset = dataset.shuffle(100, seed=config.seed)

    dataset = dataset.map(
        split_features_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size_per_replica, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
