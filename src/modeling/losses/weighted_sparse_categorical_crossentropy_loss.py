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

def weighted_sparse_categorical_crossentropy_loss(logits, labels, label_weights, metrics, name, pos_weights=None):
  with tf.name_scope(name):
    metrics = dict([(metric.name, metric) for metric in metrics])
    unweighted_losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        tf.cast(logits, tf.float32),
        from_logits=True)

    if pos_weights is None:
      pos_weights = tf.ones_like(unweighted_losses,
                                 dtype=unweighted_losses.dtype)
    else:
      pos_weights = tf.cast(pos_weights, unweighted_losses.dtype) 

    # Weights losses by thier positive weights.
    losses = pos_weights * unweighted_losses

    # Weights losses by label_weights. label_weights is used to ignore some
    # examples when we don't want to compute their losses.
    label_weights = tf.cast(label_weights, unweighted_losses.dtype) 
    losses = label_weights * losses

    numerator_loss = tf.reduce_sum(losses)
    denominator_loss = tf.reduce_sum(label_weights)
    loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
    metrics[f'{name}_loss'].update_state(loss)
  return loss
