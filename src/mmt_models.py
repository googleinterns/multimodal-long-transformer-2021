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

import tensorflow as tf
from official.modeling import tf_utils

import configs
import layers as mmt_layers
import models as mmt_models



def get_transformer_encoder(mmt_config):
  """Gets a `MmtEncoder` object.
  Args:
    mmt_config: A 'MmtConfig' object.
  Returns:
    A encoder object.
  """
  assert isinstance(mmt_config, configs.MmtConfig), 'We only support `MmtConfig` now.'

  kwargs = dict(
      vocab_size=mmt_config.vocab_size,
      segment_vocab_size=mmt_config.segment_vocab_size,
      hidden_size=mmt_config.hidden_size,
      num_hidden_layers=mmt_config.num_hidden_layers,
      num_attention_heads=mmt_config.num_attention_heads,
      intermediate_size=mmt_config.intermediate_size,
      inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
      hidden_dropout_prob=mmt_config.hidden_dropout_prob,
      attention_probs_dropout_prob=mmt_config.attention_probs_dropout_prob,
      max_absolute_position_embeddings=mmt_config.max_absolute_position_embeddings,
      relative_vocab_size=mmt_config.relative_vocab_size,
      relative_pos_max_distance=mmt_config.relative_pos_max_distance,
      initializer_range=mmt_config.initializer_range,
      use_pre_activation_order=mmt_config.use_pre_activation_order,
      use_one_hot_lookup=mmt_config.use_one_hot_lookup,
  )
  return mmt_models.MmtEncoder(**kwargs)


def pretrain_model(mmt_config: configs.MmtConfig,
                   mpp_output_num_classes: int,
                   patch_embedding_size: int,
                   return_core_pretrainer_model: bool = False):
  """Returns model to be used for pre-training.
  Args:
    mmt_config: Configuration that defines the core BERT model.
    mpp_output_num_classes: The number of output classes for patch prediction.
      it should be (2 ** output_channel_bits ) ** 3.
    return_core_pretrainer_model: Whether to also return the `MmtPretrainer`
      object.
  Returns:
    A Tuple of
    (1) MmtPretrainer `with MmtPretrainLossAndMetricLayer`.
    (2) core Mmt submodel from which to save weights after pretraining.
    (3) [optional] core `MmtPretrainer` object.
  """
  word_ids = tf.keras.layers.Input(
      shape=(None,),
      name='word_ids', dtype=tf.int32)
  segment_ids = tf.keras.layers.Input(
      shape=(None,),
      name='segment_ids', dtype=tf.int32)
  relative_att_ids = tf.keras.layers.Input(
      shape=(None, None,),
      name='relative_att_ids', dtype=tf.int32)
  att_mask = tf.keras.layers.Input(
      shape=(None,),
      name='att_mask', dtype=tf.int32)
  patch_embeddings = tf.keras.layers.Input(
      shape=(None, patch_embedding_size,),
      name='patch_embeddings',
      dtype=tf.float32)
  masked_text_positions = tf.keras.layers.Input(
      shape=(None,),
      name='masked_text_positions',
      dtype=tf.int32)
  masked_text_label_ids = tf.keras.layers.Input(
      shape=(None,),
      name='masked_text_label_ids',
      dtype=tf.int32)
  masked_text_label_weights = tf.keras.layers.Input(
      shape=(None,),
      name='masked_text_label_weights',
      dtype=tf.int32)
  masked_patch_positions = tf.keras.layers.Input(
      shape=(None,),
      name='masked_patch_positions',
      dtype=tf.int32)
  masked_patch_label_ids = tf.keras.layers.Input(
      shape=(None,),
      name='masked_patch_label_ids',
      dtype=tf.int32)
  masked_patch_label_weights = tf.keras.layers.Input(
      shape=(None,),
      name='masked_patch_label_weights',
      dtype=tf.int32)

  transformer_encoder = get_transformer_encoder(mmt_config)
  transformer_encoder._word_embedding_layer.build(word_ids.shape)

  pretrainer_model = mmt_models.MmtPretrainer(
      mmt_encoder=transformer_encoder,
      mpp_output_num_classes=mpp_output_num_classes,
      mlm_activation=tf_utils.get_activation(mmt_config.hidden_act),
      mlm_initializer='glorot_uniform',
      mpp_activation=tf_utils.get_activation(mmt_config.hidden_act),
      mpp_initializer='glorot_uniform')

  outputs = pretrainer_model(
      word_ids=word_ids,
      segment_ids=segment_ids,
      att_mask=att_mask,
      relative_att_ids=relative_att_ids,
      patch_embeddings=patch_embeddings,
      masked_text_positions=masked_text_positions,
      masked_patch_positions=masked_patch_positions)

  pretrain_loss_layer = mmt_layers.MmtPretrainLossAndMetricLayer(
      vocab_size=mmt_config.vocab_size,
      mpp_output_num_classes=mpp_output_num_classes)
  output_loss = pretrain_loss_layer(outputs['mlm_logits'],
                                    masked_text_label_ids,
                                    masked_text_label_weights,
                                    outputs['mpp_logits'],
                                    masked_patch_label_ids,
                                    masked_patch_label_weights)
  inputs = {
      'word_ids': word_ids,
      'segment_ids': segment_ids,
      'att_mask': att_mask,
      'relative_att_ids': relative_att_ids,
      'patch_embeddings': patch_embeddings,
      'masked_text_positions': masked_text_positions,
      'masked_text_label_ids': masked_text_label_ids,
      'masked_text_label_weights': masked_text_label_weights,
      'masked_patch_positions': masked_patch_positions,
      'masked_patch_label_ids': masked_patch_label_ids,
      'masked_patch_label_weights': masked_patch_label_weights,
  }

  keras_model = tf.keras.Model(inputs=inputs, outputs=output_loss)
  
  if return_core_pretrainer_model:
    return keras_model, transformer_encoder, pretrainer_model
  else:
    return keras_model, transformer_encoder
