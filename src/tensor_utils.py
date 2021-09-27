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


def full(shape, dtype, value):
  return tf.ones(shape=shape, dtype=dtype) * value


def ragged_full(shape, dtype, value):
  return tf.RaggedTensor.from_tensor(full(shape, dtype, value))


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch.
  Reference: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_pretraining.py#L308
  
  """

  sequence_shape = tf_utils.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def pad_to_max_seq_len(tensor, max_seq_len, axis=0):
  """Pads id tensor to the match the `max_seq_len`.

  Args:
    tensor: <int32>[..., seq_len, ...]. A tensor with arbitrary shape.
    max_seq_len: The maximum sequence length we want to append.
    axis: the axis to be padded.

  Returns:
    <int32>[..., max_seq_len, ...].
  
  """
  shape = tf_utils.get_shape_list(tensor)
  lack_seq_len = max_seq_len - shape[axis]
  lack_seq_len = tf.convert_to_tensor([lack_seq_len])

  indices = tf.constant([[axis]])
  updates = tf.pad(lack_seq_len, paddings=[[1, 0]])
  updates = tf.expand_dims(updates, axis=0)
  shape = tf.constant([len(shape), 2])
  paddings = tf.scatter_nd(indices, updates, shape)
  tensor = tf.pad(tensor, paddings=paddings)
  return tensor
