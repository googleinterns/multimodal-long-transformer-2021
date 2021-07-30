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

import tensor_utils


class MaskedPP(tf.keras.layers.Layer):
  """Masked patch prediction network head for image modeling.

  This layer implements a masked patch prediction based on the provided
  transformer based encoder.

  Args:
    output_num_classes: The number of output classes for patch prediction.
      it should be (2 ** output_channel_bits ) ** 3.
    activation: The activation, if any, for the dense layer.
    initializer: The initializer for the dense layer. Defaults to a Glorot
      uniform initializer.
    output: The output style for this layer. Can be either 'logits' or
      'predictions'.
      
  Inputs:
    sequence_data: <tf.Tensor>[batch, sequence_length, embedding_size]
      encoder sequence output.
    masked_positions: <tf.Tensor>[batch, mask_sequence_length]
      the indices of masked tokens.
    
  Outputs: <tf.Tensor>[batch, mask_sequence_length, output_num_classes] 
    the logits or predictions.
  """


  def __init__(self,
               output_num_classes: int,
               activation: Optional[str]=None,
               initializer: str = 'glorot_uniform',
               output: str = 'logits',
               name: Optional[str] = None,
               **kwargs):
    super(MaskedPP, self).__init__(name=name, **kwargs)
    self.activation = activation
    self.initializer = tf.keras.initializers.get(initializer)

    if output not in ('predictions', 'logits'):
      raise ValueError(
          (f'Unknown `output` value "{output}". `output` can be either "logits"'
           f'or "predictions"')
    self._output_type = output
    self._output_num_classes = output_num_classes

  def build(self, input_shape):
    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='transform/LayerNorm')
    self.dense = tf.keras.layers.Dense(
        self._output_num_classes,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name='transform/dense')
    self.bias = self.add_weight(
        'output_bias/bias',
        shape=(self._output_num_classes,),
        initializer='zeros',
        trainable=True)

    super(MaskedPP, self).build(input_shape)

  def call(self, sequence_data, masked_positions):
    masked_pp_input = tensor_utils.gather_indexes(sequence_data,
                                                  masked_positions)
    pp_data = self.layer_norm(masked_pp_input)
    pp_data = self.dense(pp_data)
    logits = tf.nn.bias_add(pp_data, self.bias)
    masked_positions_length = tf_utils.get_shape_list(masked_positions)[1]
    logits = tf.reshape(logits,
                        [-1, masked_positions_length, self._output_num_classes])
    if self._output_type == 'logits':
      return logits
    return tf.nn.log_softmax(logits)
