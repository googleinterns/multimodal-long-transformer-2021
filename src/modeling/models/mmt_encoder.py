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

import collections
from typing import Optional

import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.keras_nlp import layers

from etcmodel import feature_utils as etc_feature_utils
from etcmodel import layers as etc_layers


_NUM_OTHER_RELATIVE_IDS = 3


class MmtEncoder(tf.keras.Model):
  """Multimodal Transformer-based Encoder.

  This network implements a multimodal transformer-based encoder for 
  image and language understanding. It includes the embedding lookups and 
  relative transformer layers, but not the masked language model (mlm),
  masked patch prediction (mpp), or classification task networks.
  We follow BERT implementation and use approximated gelu function for faster
  TPU computation.
  (1) Bert implementation: https://github.com/tensorflow/models/blob/2de518be2d6a6e3670b223a4582b1353538d3489/official/nlp/keras_nlp/encoders/bert_encoder.py#L26inner_activation
  (2) Related issue: https://github.com/google/jax/issues/4428#issuecomment-701793190

  Args: refer to `MmtEncoderConfig` for more details.

  """

  def __init__(self,
               vocab_size: int,
               segment_vocab_size: int = 16,
               embedding_size: int = None,
               hidden_size: int = 768,
               num_hidden_layers: int = 12,
               num_attention_heads: int = 12,
               intermediate_size: int = 3072,
               inner_activation=lambda x: tf.keras.activations.gelu(
                   x, approximate=True),
               hidden_dropout_prob: float = 0.1, 
               attention_probs_dropout_prob: float = 0.1,
               max_absolute_position_embeddings: Optional[int] = None,
               relative_vocab_size: int = 32,
               relative_pos_max_distance: int = 12,
               initializer_range: float = 0.02,
               use_pre_activation_order: bool = False,
               use_one_hot_lookup: bool = True,
               use_pooler_layer: bool = False,
               name: str = 'mmt_encoder',
               **kwargs):

     super(MmtEncoder, self).__init__(name=name, **kwargs)

     if relative_vocab_size is None:
       if relative_pos_max_distance != 0:
         raise ValueError(
             '`relative_pos_max_distance` must be 0 when `relative_vocab_size` '
             'is None.')
     elif relative_vocab_size < (
         etc_feature_utils.RelativePositionGenerator(
             relative_pos_max_distance).relative_vocab_size +
             _NUM_OTHER_RELATIVE_IDS):
       raise ValueError(
           f'`relative_vocab_size` ({relative_vocab_size}) too small for '
           f'`relative_pos_max_distance` ({relative_pos_max_distance}')

     if embedding_size is None:
       embedding_size = hidden_size

     activation = tf.keras.activations.get(inner_activation)
     initializer = tf.keras.initializers.TruncatedNormal(
         stddev=initializer_range)
     initializer = tf.keras.initializers.get(initializer)

     self._word_embedding_layer = etc_layers.EmbeddingLookup(
         vocab_size=vocab_size,
         embedding_size=embedding_size,
         projection_size=hidden_size,
         initializer_range=initializer_range,
         name='word_embeddings')

     if max_absolute_position_embeddings is None:
       self._position_embedding_layer = None
     else:
       self._position_embedding_layer = layers.PositionEmbedding(
           initializer=initializer,
           max_length=max_absolute_position_embeddings,
           name='absolute_position_embeddings')

     self._segment_embedding_layer = etc_layers.EmbeddingLookup(
         vocab_size=segment_vocab_size,
         embedding_size=embedding_size,
         projection_size=hidden_size,
         initializer_range=initializer_range,
         use_one_hot_lookup=use_one_hot_lookup,
         name='segment_embeddings')
     
     self._patch_embedding_projection = tf.keras.layers.Dense(
         units=hidden_size,
         kernel_initializer=initializer,
         name='patch_embedding_projection')

     self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
         axis=-1, epsilon=1e-12, name='embeddings/layer_norm')

     self._embedding_dropout_layer = tf.keras.layers.Dropout(
         rate=hidden_dropout_prob, name='embeddings/dropout')

     self._transformer_layers = etc_layers.RelativeTransformerLayers(
         hidden_size=hidden_size,
         num_hidden_layers=num_hidden_layers,
         num_attention_heads=num_attention_heads,
         intermediate_size=intermediate_size,
         hidden_act=activation,
         hidden_dropout_prob=hidden_dropout_prob,
         attention_probs_dropout_prob=attention_probs_dropout_prob,
         initializer_range=initializer_range,
         relative_vocab_size=relative_vocab_size,
         use_pre_activation_order=use_pre_activation_order,
         use_one_hot_lookup=use_one_hot_lookup)

     if use_pooler_layer:
       self._pooler_layer = tf.keras.layers.Dense(
           units=hidden_size,
           activation='tanh',
           kernel_initializer=initializer,
           name='pooler_transform')

     config_dict = {
         'vocab_size': vocab_size,
         'segment_vocab_size': segment_vocab_size,
         'hidden_size': hidden_size,
         'num_hidden_layers': num_hidden_layers,
         'num_attention_heads': num_attention_heads,
         'intermediate_size': intermediate_size,
         'inner_activation': tf.keras.activations.serialize(activation),
         'hidden_dropout_prob': hidden_dropout_prob,
         'attention_probs_dropout_prob': attention_probs_dropout_prob,
         'max_absolute_position_embeddings': max_absolute_position_embeddings,
         'relative_vocab_size': relative_vocab_size,
         'relative_pos_max_distance': relative_pos_max_distance,
         'initializer_range': initializer_range,
         'use_pre_activation_order': use_pre_activation_order,
         'use_one_hot_lookup': use_one_hot_lookup,
         'use_pooler_layer': use_pooler_layer
      }

     config_cls = collections.namedtuple('Config', config_dict.keys())
     self._config = config_cls(**config_dict)

  def call(self,
           word_ids: tf.Tensor,
           segment_ids: Optional[tf.Tensor] = None,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           patch_embeddings: Optional[tf.Tensor] = None,
           training: Optional[bool] = None):
    """Calls MmtEncoder layer.

      Args:
        word_ids: <int32>[batch_size, seq_len] Tensor of word piece ids.
        segment_ids: <int32>[batch_size, seq_len] Optional Tensor of segment
          ids. By default we just fill all elements with segment id 1.
        att_mask: <int32>[batch_size, seq_len,  seq_len].
        relative_att_ids: <int32>[batch_size, seq_len, seq_len].
        training: For Keras, optional boolean scalar tensor or Python boolean
          indicating whether the call is meant for training or inference.

      Returns: A dictionary of encoder outputs. 
        sequence_output: <float32>[batch_size, seq_len, hidden_size].

    """

    if segment_ids is None:
      segment_ids = tf.ones_like(word_ids, dtype=tf.int32)

    word_embeddings = self._word_embedding_layer(word_ids)
    segment_embeddings = self._segment_embedding_layer(segment_ids)

    word_embeddings = self._embedding_norm_layer(word_embeddings)
    word_embeddings = self._embedding_dropout_layer(word_embeddings,
                                                    training=training)

    embeddings = word_embeddings + segment_embeddings

    if self._position_embedding_layer is not None:
      position_embeddings = self._position_embedding_layer(word_embeddings)
      embeddings += position_embeddings

    if patch_embeddings is not None:
      seq_len = tf_utils.get_shape_list(word_embeddings)[1]
      patch_seq_len = tf_utils.get_shape_list(patch_embeddings)[1]

      patch_embeddings = self._patch_embedding_projection(patch_embeddings)

      # Make patch_embeddings and word_embeddings have the same shape.
      # 2 is for CLS and [PATCH]
      prefix_pad_len = 2
      suffix_pad_len = seq_len - 2 - patch_seq_len 
      patch_embeddings = tf.pad(
          patch_embeddings,
          paddings=[[0, 0], [prefix_pad_len, suffix_pad_len], [0, 0]])
      embeddings += patch_embeddings

    encoder_output = self._transformer_layers(
        inputs=embeddings,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=training)

    outputs = dict()
    outputs['sequence_output'] = encoder_output

    if hasattr(self, '_pooler_layer'):
      batch_size, _, hidden_size = tf_utils.get_shape_list(encoder_output)
      first_token_tensor = tf.slice(
          encoder_output, [0, 0, 0], [batch_size, 1, hidden_size])
      first_token_tensor = tf.squeeze(first_token_tensor, axis=1)
      cls_output = self._pooler_layer(first_token_tensor)
      outputs[pooled_output] = cls_output

    return outputs

  def get_word_embedding_table(self):
    """Returns the token embedding table, but only if the model is built."""
    if not hasattr(self._word_embedding_layer, 'embedding_table'):
      raise ValueError(
          'Cannot call `get_token_embedding_table()` until the model has been '
          'called so that all variables are built.')
    return self._word_embedding_layer.embedding_table

  def get_word_embedding_layer(self):
    return self._word_embedding_layer

  def get_config(self):
    return dict(self._config._asdict())

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    if hasattr(self, '_pooler_layer'):
      return self._pooler_layer
    else:
      raise ValueError('pooler layers is not initialized.')

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)
