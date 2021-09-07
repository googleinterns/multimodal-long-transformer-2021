# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def weighted_sparse_categorical_crossentropy_loss(logits,
                                                  labels,
                                                  label_weights,
                                                  metrics,
                                                  name):
  with tf.name_scope(name):
    metrics = dict([(metric.name, metric) for metric in metrics])
    prediction_losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        tf.cast(logits, tf.float32),
        from_logits=True)
    label_weights = tf.cast(label_weights, prediction_losses.dtype) 
    numerator_loss = tf.reduce_sum(prediction_losses * label_weights)
    denominator_loss = tf.reduce_sum(label_weights)
    loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
    metrics[f'{name}_loss'].update_state(loss)
  return loss
