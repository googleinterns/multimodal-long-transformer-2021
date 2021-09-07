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

"""Transformer Encoders.
Includes configurations and factory methods.
"""
from typing import Optional

import dataclasses
import gin
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import networks

from modeling import models


@dataclasses.dataclass
class BertEncoderConfig(hyperparams.Config):
  """BERT encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None
  output_range: Optional[int] = None
  return_all_encoder_outputs: bool = False
  # Pre/Post-LN Transformer
  norm_first: bool = False


@dataclasses.dataclass
class MmtEncoderConfig(hyperparams.Config):
  """Mmt encoder configuration."""

  # Vocabulary size of `token_ids`.
  vocab_size: int = 30522

  # Vocabulary size of `segment_ids`.
  segment_vocab_size: int = 16

  # Size of `token_ids` embeddings.
  # The default of `None` makes this equal to `hidden_size` like original
  # BERT, but it can be set to a smaller value (e.g. 128) like ALBERT. Must
  # be positive.
  embedding_size: int = None

  # Size of the encoder layers and the pooler layer.
  hidden_size: int = 768

  # Number of hidden layers in the Transformer encoder.
  num_hidden_layers: int = 12

  # Number of attention heads for each attention layer in the Transformer
  # encoder
  num_attention_heads: int = 12

  # Maximum distance to use for relative position representations.
  # All larger distances will be clipped to this value.
  relative_pos_max_distance: int = 12

  # The total vocabulary size for relative positions.
  # This must be at least `2 * relative_pos_max_distance + 1`.
  relative_vocab_size: int = 32

  # If larger than 0, use `MmtRelativePositionGenerator` instead of
  # `RelativeTransformerSideInputs`.
  relative_att_num_core_layers: int = 0

  # The maximum sequence length that this model might ever be used with;
  # used for absolute position embeddings like BERT.
  # If set to 0 (the default), we skip absolute position embeddings entirely.
  # If nonzero, inputs larger than this value are not allowed.
  max_absolute_position_embeddings: int = None

  # The size of the "intermediate" (i.e., feed-forward) layer in the 
  # Transformer encoder.
  intermediate_size: int = 3072

  # The non-linear activation function (function or string) in the encoder
  # and pooler.
  hidden_activation: str = 'gelu'

  #  The dropout probability for all fully connected layers in the embeddings,
  # encoder, and pooler.
  hidden_dropout_prob: float = 0.1

  # The dropout ratio for the attention probabilities.
  attention_probs_dropout_prob: float = 0.1

  # The stdev of the truncated_normal_initializer for initializing all weight
  # matrices.
  initializer_range: float = 0.02

  # If True, use "pre-activation" order for residual blocks.
  use_pre_activation_order: bool = False

  # If True, use one-hot lookup for embedding layers with samller vocab.
  use_one_hot_lookup: bool = True

  # If True, use pooler layer to get CLS token embedding.
  use_pooler_layer: bool = False


@dataclasses.dataclass
class EncoderConfig(hyperparams.OneOfConfig):
  """Encoder configuration."""
  type: Optional[str] = "bert"
  bert: BertEncoderConfig = BertEncoderConfig()
  mmt: MmtEncoderConfig = MmtEncoderConfig()


@gin.configurable
def build_encoder(config: EncoderConfig,
                  embedding_layer: Optional[tf.keras.layers.Layer] = None,
                  encoder_cls=None,
                  bypass_config: bool = False):
  """Instantiate a Transformer encoder network from EncoderConfig.

  Args:
    config: the one of encoder config, which provides encoder parameters of a
      chosen encoder.
    embedding_layer: an external embedding layer passed to the encoder.
    encoder_cls: an external encoder cls not included in the supported encoders,
      usually used by gin.configurable.
    bypass_config: whether to ignore config instance to create the object with
      `encoder_cls`.

  Returns:
    An encoder instance.

  """
  if bypass_config:
    return encoder_cls()
  encoder_type = config.type
  encoder_cfg = config.get()

  if encoder_type == "mmt":
    return models.MmtEncoder(
        vocab_size=encoder_cfg.vocab_size,
        segment_vocab_size=encoder_cfg.segment_vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        num_hidden_layers=encoder_cfg.num_hidden_layers,
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        inner_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
        hidden_dropout_prob=encoder_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=encoder_cfg.attention_probs_dropout_prob,
        max_absolute_position_embeddings=encoder_cfg.max_absolute_position_embeddings,
        relative_vocab_size=encoder_cfg.relative_vocab_size,
        relative_pos_max_distance=encoder_cfg.relative_pos_max_distance,
        initializer_range=encoder_cfg.initializer_range,
        use_pre_activation_order=encoder_cfg.use_pre_activation_order,
        use_one_hot_lookup=encoder_cfg.use_one_hot_lookup,
        use_pooler_layer=encoder_cfg.use_pooler_layer,
    )

  # Uses the default BERTEncoder configuration schema to create the encoder.
  return networks.BertEncoder(
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      intermediate_size=encoder_cfg.intermediate_size,
      activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      dropout_rate=encoder_cfg.dropout_rate,
      attention_dropout_rate=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      embedding_layer=embedding_layer,
      return_all_encoder_outputs=encoder_cfg.return_all_encoder_outputs,
      dict_outputs=True,
      norm_first=encoder_cfg.norm_first)
