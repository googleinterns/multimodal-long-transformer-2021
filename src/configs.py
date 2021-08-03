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

import json

import attr
import tensorflow as tf

@attr.s
class MmtConfig(object):
  """Configuration for `MmtModel`."""

  # Vocabulary size of `token_ids`.
  vocab_size = attr.ib()

  # Vocabulary size of `segment_ids`.
  segment_vocab_size = attr.ib(default=16)

  # Size of `token_ids` embeddings.
  # The default of `None` makes this equal to `hidden_size` like original
  # BERT, but it can be set to a smaller value (e.g. 128) like ALBERT. Must
  # be positive.
  embedding_size = attr.ib(default=None)

  # Size of the encoder layers and the pooler layer.
  hidden_size = attr.ib(default=768)

  # Number of hidden layers in the Transformer encoder.
  num_hidden_layers = attr.ib(default=12)

  # Number of attention heads for each attention layer in the Transformer
  # encoder
  num_attention_heads = attr.ib(default=12)

  # Maximum distance to use for relative position representations.
  # All larger distances will be clipped to this value.
  relative_pos_max_distance = attr.ib(default=12)

  # The total vocabulary size for relative positions.
  # This must be at least `2 * relative_pos_max_distance + 1`.
  relative_vocab_size = attr.ib(default=32)

  # The maximum sequence length that this model might ever be used with;
  # used for absolute position embeddings like BERT.
  # If set to 0 (the default), we skip absolute position embeddings entirely.
  # If nonzero, inputs larger than this value are not allowed.
  max_absolute_position_embeddings = attr.ib(default=None)

  # The size of the "intermediate" (i.e., feed-forward) layer in the 
  # Transformer encoder.
  intermediate_size = attr.ib(default=3072)

  # The non-linear activation function (function or string) in the encoder
  # and pooler.
  hidden_act = attr.ib(default='gelu')

  #  The dropout probability for all fully connected layers in the embeddings,
  # encoder, and pooler.
  hidden_dropout_prob = attr.ib(default=0.1)
  
  # The dropout ratio for the attention probabilities.
  attention_probs_dropout_prob = attr.ib(default=0.1)

  # The stdev of the truncated_normal_initializer for initializing all weight
  # matrices.
  initializer_range = attr.ib(default=0.02)

  # If True, use "pre-activation" order for residual blocks.
  use_pre_activation_order = attr.ib(default=False)

  # If True, use one-hot lookup for embedding layers with samller vocab
  use_one_hot_lookup = attr.ib(default=True)
  
  @classmethod
  def from_dict(cls, json_object):
    """Constructs `MmtConfig` from Python dictionary of parameters."""
    return cls(**json_object)

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs `MmtConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    return attr.asdict(self)

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
