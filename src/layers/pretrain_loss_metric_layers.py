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

def get_loss(output_logits, label_ids, label_weights):
  label_weights = tf.cast(label_weights, tf.float32)
  output_logits = tf.cast(output_logits, tf.float32)

  prediction_losses = tf.keras.losses.sparse_categorical_crossentropy(
      label_ids, output_logits, from_logits=True)
  numerator_loss = tf.reduce_sum(prediction_losses * label_weights)
  denominator_loss = tf.reduce_sum(label_weights)
  mask_loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
  return mask_loss


class MmtPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, vocab_size, mpp_output_num_classes, **kwargs):
    super(MmtPretrainLossAndMetricLayer, self).__init__(**kwargs)
    self._vocab_size = vocab_size
    self.config = {
        'vocab_size': vocab_size,
        'mpp_output_num_classes': mpp_output_num_classes,
    }

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_example_loss, name):
    """Adds metrics."""
    lm_label_weights = tf.cast(lm_label_weights, tf.float32)
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    numerator = tf.reduce_sum(masked_lm_accuracy * lm_label_weights)
    denominator = tf.reduce_sum(lm_label_weights) + 1e-5
    masked_lm_accuracy = numerator / denominator
    self.add_metric(masked_lm_accuracy,
                    name=f'{name}_masked_lm_accuracy', aggregation='mean')

    self.add_metric(lm_example_loss,
                    name=f'{name}_example_loss', aggregation='mean')

  def call(self,
           mlm_logits,
           mlm_label_ids,
           mlm_label_weights,
           mpp_logits,
           mpp_label_ids,
           mpp_label_weights):
    # MLM
    mlm_loss = get_loss(mlm_logits, mlm_label_ids, mlm_label_weights)
    mpp_loss = get_loss(mpp_logits, mpp_label_ids, mpp_label_weights)

    loss = mlm_loss + mpp_loss

    batch_shape = tf.slice(tf.shape(mlm_label_ids), [0], [1])
    final_loss = tf.fill(batch_shape, loss)

    self._add_metrics(mlm_logits, mlm_label_ids, mlm_label_weights,
                      mlm_loss, name='mlm')
    self._add_metrics(mpp_logits, mpp_label_ids, mpp_label_weights,
                      mpp_loss, name='mpp')

    return final_los
