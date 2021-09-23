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

"""Creates `keras.layers.Input`."""

from official.core import config_definitions as cfg
import tensorflow as tf


def create_mmt_encoder_inputs(data_cfg: cfg.DataConfig):
  """Creates inputs for `MmtEncoder`."""

  patch_embedding_size = (data_cfg.patch_size ** 2) * 3
  num_patch_per_row = data_cfg.image_size // data_cfg.patch_size 
  num_patches = num_patch_per_row ** 2
  max_seq_len = data_cfg.max_seq_len

  word_ids = tf.keras.layers.Input(
      shape=(max_seq_len,),
      name='word_ids', dtype=tf.int32)
  segment_ids = tf.keras.layers.Input(
      shape=(max_seq_len,),
      name='segment_ids', dtype=tf.int32)
  relative_att_ids = tf.keras.layers.Input(
      shape=(max_seq_len, max_seq_len),
      name='relative_att_ids', dtype=tf.int32)
  att_mask = tf.keras.layers.Input(
      shape=(max_seq_len, max_seq_len),
      name='att_mask', dtype=tf.int32)
  patch_embeddings = tf.keras.layers.Input(
      shape=(num_patches, patch_embedding_size,),
      name='patch_embeddings',
      dtype=tf.float32)

  inputs = {
      'word_ids': word_ids,
      'segment_ids': segment_ids,
      'att_mask': att_mask,
      'relative_att_ids': relative_att_ids,
      'patch_embeddings': patch_embeddings,
  }
  return inputs


def create_mtm_inputs_and_labels(data_cfg: cfg.DataConfig):
  """Creates inputs and labels for `MmtPretrainingModel`.

  Masked Token Modeling (MTM) includes Masked Language Modeling (MLM) and 
  Masked Patch Prediction (MPP).

  """

  mlm_max_selections_per_seq = data_cfg.mlm_max_selections_per_seq
  mpp_max_selections_per_seq = data_cfg.mpp_max_selections_per_seq

  inputs, labels = dict(), dict()
  mlm_positions = tf.keras.layers.Input(
      shape=(mlm_max_selections_per_seq,),
      name='mlm_positions',
      dtype=tf.int32)
  mlm_label_ids = tf.keras.layers.Input(
      shape=(mlm_max_selections_per_seq,),
      name='mlm_label_ids',
      dtype=tf.int32)
  mlm_label_weights = tf.keras.layers.Input(
      shape=(mlm_max_selections_per_seq,),
      name='mlm_label_weights',
      dtype=tf.int32)
  mpp_positions = tf.keras.layers.Input(
      shape=(mpp_max_selections_per_seq,),
      name='mpp_positions',
      dtype=tf.int32)
  mpp_label_ids = tf.keras.layers.Input(
      shape=(mpp_max_selections_per_seq,),
      name='mpp_label_ids',
      dtype=tf.int32)
  mpp_label_weights = tf.keras.layers.Input(
      shape=(mpp_max_selections_per_seq,),
      name='mpp_label_weights',
      dtype=tf.int32)

  inputs.update({
      'mlm_positions': mlm_positions,
      'mpp_positions': mpp_positions,
  })
  labels = {
      'mlm_label_ids': mlm_label_ids,
      'mlm_label_weights': mlm_label_weights,
      'mpp_label_ids': mpp_label_ids,
      'mpp_label_weights': mpp_label_weights,
  }

  return inputs, labels
