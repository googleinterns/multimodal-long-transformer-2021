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

import functools

import tensorflow as tf
from official.modeling import tf_utils

import modeling
import input_utils
import tensor_utils


def model_fn_builder(model_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step, learning_rate_schedule):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    raise NotImplementedError

  return model_fn


def metric_fn(lm_example_loss,
              masked_lm_log_probs,
              masked_lm_ids,
              masked_lm_weights,
              is_train=False):
  """Computes the loss and accuracy of the model."""
  raise NotImplementedError


def get_mpp_output(model_config, input_tensor, output_num_classes, positions,
                   label_ids, label_weights):

  """Get loss and log probs for the masked patch prediction (MPP)."""
  input_tensor = tensor_utils.gather_indexes(input_tensor, positions)

  # NOTE (roylu): Do we need layer norm before the linear projection? 
  input_tensor = modeling.layer_norm(input_tensor)

  # We apply one more non-linear transformation before the output layer.
  # This matrix is not used after pre-training.
  logits = tf.compat.v1.layers.dense(
      input_tensor,
      units=output_num_classes,
      activation=tf_utils.get_activation(model_config.hidden_act),
      kernel_initializer=modeling.create_initializer(
          model_config.initializer_range))
  log_probs = tf.nn.log_softmax(logits, axis=-1)

  label_ids = tf.reshape(label_ids, [-1])
  label_weights = tf.reshape(label_weights, [-1])

  one_hot_labels = tf.one_hot(
      label_ids, depth=output_num_classes, dtype=tf.float32)

  per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
  numerator = tf.reduce_sum(label_weights * per_example_loss)
  denominator = tf.reduce_sum(label_weights) + 1e-5
  loss = numerator / denominator
  return (loss, per_example_loss, log_probs)


def get_mlm_output(model_config, input_tensor, output_weights, positions,
                   label_ids, label_weights):
  """Get loss and log probs for the masked language model (MLM)."""
  input_tensor = tensor_utils.gather_indexes(input_tensor, positions)

  input_tensor = tf.compat.v1.layers.dense(
      input_tensor,
      units=(model_config.embedding_size
             if model_config.embedding_size is not None else
             model_config.hidden_size),
      activation=tf_utils.get_activation(model_config.hidden_act),
      kernel_initializer=modeling.create_initializer(
          model_config.initializer_range))
  input_tensor = modeling.layer_norm(input_tensor)

  # The output weights are the same as the input embeddings, but there is
  # an output-only bias for each token.
  output_bias = tf.compat.v1.get_variable(
      "output_bias",
      shape=[model_config.vocab_size],
      initializer=tf.zeros_initializer())
  logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  log_probs = tf.nn.log_softmax(logits, axis=-1)

  label_ids = tf.reshape(label_ids, [-1])
  label_weights = tf.reshape(label_weights, [-1])

  one_hot_labels = tf.one_hot(
      label_ids, depth=model_config.vocab_size, dtype=tf.float32)

  per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
  numerator = tf.reduce_sum(label_weights * per_example_loss)
  denominator = tf.reduce_sum(label_weights) + 1e-5
  loss = numerator / denominator
  return (loss, per_example_loss, log_probs)
