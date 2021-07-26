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

from typing import Optional

import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from official.modeling import tf_utils

from etcmodel.models import etc_modeling
from etcmodel import layers as etc_layers
from etcmodel import feature_utils as etc_feature_utils

_NUM_OTHER_RELATIVE_IDS = 3


class EtcModel(tf.keras.layers.Layer):
  """ETC model."""

  def __init__(self,
               config: modeling.EtcConfig,
               is_training: Optional[bool] = None,
               use_one_hot_embeddings=False,
               use_one_hot_relative_embeddings=False,
               name: str = "etc",
               **kwargs):
    """Constructor for `EtcModel`.
    Args:
      config: `EtcConfig` instance.
      is_training: Optional bool. True for training model, False for eval model.
        The None default will defer to the typical Keras `training` argument in
        `call` instead. When `is_training` is specified here, the `training`
        argument from `call` must not be used.
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.nn.embedding_lookup() for the word embeddings.
      use_one_hot_relative_embeddings: (optional) bool. Whether to use one-hot
        word embeddings or tf.nn.embedding_lookup() for the relative position
        embeddings.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.
    Raises:
      ValueError: The config is invalid.
    """
    super(EtcModel, self).__init__(name=name, **kwargs)

    config = copy.deepcopy(config)
    if is_training is not None and not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    self.config = config
    self.is_training = is_training
    self.use_one_hot_embeddings = use_one_hot_embeddings
    self.use_one_hot_relative_embeddings = use_one_hot_relative_embeddings

    if config.relative_vocab_size is None:
      if config.relative_pos_max_distance != 0:
        raise ValueError(
            '`relative_pos_max_distance` must be 0 when `relative_vocab_size` '
            'is None.')
    elif config.relative_vocab_size < (etc_feature_utils.RelativePositionGenerator(
        config.relative_pos_max_distance).relative_vocab_size +
                                       _NUM_OTHER_RELATIVE_IDS):
      raise ValueError(
          f'`relative_vocab_size` ({config.relative_vocab_size}) too small for '
          f'`relative_pos_max_distance` ({config.relative_pos_max_distance}')
    if config.embedding_size is None:
      config.embedding_size = config.hidden_size

    self.patch_embedding_transform = tf.keras.layers.Dense(
        units=config.embedding_size)

    self.token_embedding = etc_layers.EmbeddingLookup(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        projection_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name='token_emb_lookup')

    self.token_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='long_emb_layer_norm')
    self.token_embedding_dropout = tf.keras.layers.Dropout(
        rate=config.hidden_dropout_prob)

    self.segment_embedding = etc_layers.EmbeddingLookup(
        vocab_size=config.segment_vocab_size,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=True,
        name='segment_emb_lookup')

    if config.max_absolute_position_embeddings != 0:
      self.position_embedding = etc_layers.EmbeddingLookup(
          vocab_size=config.max_absolute_position_embeddings,
          embedding_size=config.hidden_size,
          initializer_range=config.initializer_range,
          use_one_hot_lookup=use_one_hot_embeddings,
          name='position_emb_lookup_long')
      # Call layers to force variable initialization.
      self.position_embedding(tf.ones([1, 1], tf.int32))
    else:
      self.position_embedding = None

    self.relative_transformer = etc_layers.RelativeTransformerLayers(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=tf_utils.get_activation(config.hidden_act),
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        relative_vocab_size=config.relative_vocab_size,
        use_pre_activation_order=config.use_pre_activation_order,
        use_one_hot_lookup=use_one_hot_relative_embeddings)

  def call(self,
           token_ids: tf.Tensor,
           segment_ids: Optional[tf.Tensor] = None,
           position_ids: Optional[tf.Tensor] = None,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           long_embedding_adder: Optional[tf.Tensor] = None,
           image_data: Optional[tf.Tensor] = None,
           training=None):
    """Calls the layer.
    Args:
      token_ids: <int32>[batch_size, long_seq_len] Tensor of token ids.
      segment_ids: <int32>[batch_size, long_seq_len] Optional Tensor of segment
        ids. By default we just fill all elements with segment id 1.
      position_ids: <int32>[batch_size, long_seq_len] Optional Tensor of
        absolute position ids. By default we use `tf.range(long_seq_len)` if
        `max_absolute_position_embeddings` is nonzero. The maximum position id
        must not be larger than `max_absolute_position_embeddings`.
      att_mask: <int32>[batch_size, long_seq_len,  long_seq_len].
      relative_att_ids: <int32>[batch_size, long_seq_len, long_seq_len].
      long_embedding_adder: <float32>[batch_size, long_seq_len, hidden_size]
        Tensor of additional values to add to the long input embedding before
        layer normalization of the embeddings. By default this is skipped.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference. Must
        be None if `is_training` was not None in `__init__`.
    Returns:
      A list of Tensors, [long_output, global_output]:
        long_output: <float32>[batch_size, long_seq_len, hidden_size]
    """
    if self.is_training is not None:
      if training is not None:
        raise ValueError(
            "`training` must be None when `is_training` is given in `__init__`."
        )
      training = self.is_training

    if segment_ids is None:
      segment_ids = tf.ones_like(token_ids)

    if self.config.max_absolute_position_embeddings == 0 and (
        position_ids is not None):
      raise ValueError(
          "Cannot specify `position_ids` or `global_position_ids` arguments "
          "when `max_absolute_position_embeddings` is 0.")

    token_ids = token_ids.to_tensor()
    long_input = self.token_embedding(token_ids)
    long_input += self.segment_embedding(segment_ids)
    if self.position_embedding is not None:
      if position_ids is None:
        long_input += self.position_embedding.embedding_table[
            tf.newaxis, :long_input.shape[1], :]
      else:
        long_input += self.position_embedding(position_ids)
    if long_embedding_adder is not None:
      long_input += long_embedding_adder

    if image_data is not None:

      # Create padding to make image_data and long_input have the same shape.
      batch_size, seq_len, hidden_size = tf_utils.get_shape_list(long_input)
      _, image_seq_len, _ = tf_utils.get_shape_list(image_data)
      prefix_pad = tf.zeros(
          shape=(batch_size, 2, hidden_size),
          dtype=tf.float32)
      suffix_pad = tf.zeros(
          shape=(batch_size, seq_len-2-image_seq_len, hidden_size),
          dtype=tf.float32)
      image_data = self.patch_embedding_transform(image_data)
      
      long_input += tf.concat([prefix_pad, image_data, suffix_pad], axis=1)

    long_input = self.token_embedding_norm(long_input)
    long_input = self.token_embedding_dropout(long_input, training=training)

    return self.relative_transformer(
        inputs=long_input,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=training)

  def get_token_embedding_table(self):
    """Returns the token embedding table, but only if the model is built."""
    if not hasattr(self.token_embedding, "embedding_table"):
      raise ValueError(
          "Cannot call `get_token_embedding_table()` until the model has been "
          "called so that all variables are built.")
    return self.token_embedding.embedding_table


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return contrib_layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
