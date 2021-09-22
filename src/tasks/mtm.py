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

"""Pretraining Task."""

from typing import Mapping

import dataclasses
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.nlp.data import data_loader_factory
from official.nlp.modeling import layers

from configs import encoders
import input_utils
from modeling import losses
from configs import mmt
from modeling import models


@dataclasses.dataclass
class PretrainingTaskConfig(cfg.TaskConfig):
  """The pretraining task config."""
  model: mmt.PretrainModelConfig = mmt.PretrainModelConfig()
  scale_loss: bool = False
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(PretrainingTaskConfig)
class PretrainingTask(base_task.Task):
  """Task object for pretraining.

  Pretraining tasks include Masked Language Modeling (MLM), Masked Patch 
  Prediction (MPP), and Image-Text Matching (ITM).

  """

  def _build_encoder(self, encoder_cfg):
    return encoders.build_encoder(encoder_cfg)

  def build_model(self, params=None):
    config = params or self.task_config.model
    encoder_cfg = config.encoder
    encoder = self._build_encoder(encoder_cfg)

    data_cfg = self.task_config.train_data
    mpp_output_num_classes = (2 ** data_cfg.output_channel_bits) ** 3

    encoder_inputs = input_utils.create_mmt_encoder_inputs(data_cfg)
    encoder(**encoder_inputs)

    inputs, labels = input_utils.create_mtm_inputs_and_labels(data_cfg)

    cls_heads = []
    for cfg in config.cls_heads:
      cls_heads.append(layers.ClassificationHead(**cfg.as_dict()))
    cls_labels = input_utils.create_cls_heads_labels(config.cls_heads)

    model = models.MmtPretrainingModel(
        encoder=encoder,
        mpp_output_num_classes=mpp_output_num_classes,
        mlm_activation=tf_utils.get_activation(config.mlm_activation),
        mlm_initializer=config.mlm_initializer,
        mpp_activation=tf_utils.get_activation(config.mpp_activation),
        mpp_initializer=config.mpp_initializer,
        classification_heads=cls_heads)

    inputs.update(encoder_inputs)
    model(**inputs)

    return model

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> tf.Tensor:

    if 'itm_label_weights' in labels:
      # Masks out mlm and mpp losses on negative examples.
      itm_label_ids = labels['itm_label_ids']
      itm_label_ids = tf.expand_dims(itm_label_ids, axis=1)
      itm_label_ids = tf.cast(itm_label_ids, tf.float32)
      mlm_label_weights = labels['mlm_label_weights'] * itm_label_ids
      mpp_label_weights = labels['mpp_label_weights'] * itm_label_ids
    else:
      mlm_label_weights = labels['mlm_label_weights']
      mpp_label_weights = labels['mpp_label_weights']

    mlm_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        model_outputs['mlm_logits'],
        labels['mlm_label_ids'],
        mlm_label_weights,
        metrics,
        name='mlm')

    mpp_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        model_outputs['mpp_logits'],
        labels['mpp_label_ids'],
        mpp_label_weights,
        metrics,
        name='mpp')

    total_loss = mlm_loss + mpp_loss

    if 'itm_label_weights' in labels:
      itm_loss = losses.weighted_sparse_categorical_crossentropy_loss(
          model_outputs['itm_logits'],
          labels['itm_label_ids'],
          labels['itm_label_weights'],
          metrics,
          name='itm')
      total_loss += itm_loss

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for pretraining."""
    if params.input_path == 'dummy':

      def dummy_data(_):
        num_patch_per_row = params.image_size // params.patch_size
        num_patches = num_patch_per_row ** 2
        # batch size = 1
        dummy_ids = tf.zeros((1, params.max_seq_len), dtype=tf.int32)
        dummy_att_mask = tf.zeros((1, params.max_seq_len, params.max_seq_len),
                                  dtype=tf.int32)
        dummy_patch_embeddings = tf.ones((1, num_patches, 768),
                                         dtype=tf.float32)
        dummy_mlm = tf.zeros((1, params.mlm_max_selections_per_seq),
                             dtype=tf.int32)
        dummy_mpp = tf.zeros((1, params.mpp_max_selections_per_seq),
                             dtype=tf.int32)
        return dict(
            word_ids=dummy_ids,
            segment_ids=dummy_ids,
            att_mask=dummy_att_mask,
            relative_att_ids=dummy_att_mask,
            patch_embeddings=dummy_patch_embeddings,
            mlm_positions=dummy_mlm,
            mlm_label_ids=dummy_mlm,
            mlm_label_weights=tf.cast(dummy_mlm, dtype=tf.float32),
            mpp_positions=dummy_mpp,
            mpp_label_ids=dummy_mpp,
            mpp_label_weights=tf.cast(dummy_mpp, dtype=tf.float32),
            itm_label_ids=tf.zeros((1,), dtype=tf.int32),
            itm_label_weights=tf.ones((1,), dtype=tf.float32)
        )

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='mlm_accuracy'),
        tf.keras.metrics.Mean(name='mlm_loss'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='mpp_accuracy'),
        tf.keras.metrics.Mean(name='mpp_loss')
    ]
    for cfg in self.task_config.model.cls_heads:
      name = cfg.name
      metrics.append(
          tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_accuracy'))
      metrics.append(tf.keras.metrics.Mean(name=f'{name}_loss'))
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    with tf.name_scope('PretrainingTask/process_metrics'):
      if 'itm_label_weights' in labels:
        # Masks out mlm and mpp losses on negative examples of itm.
        itm_label_ids = labels['itm_label_ids']
        itm_label_ids = tf.expand_dims(itm_label_ids, axis=1)
        itm_label_ids = tf.cast(itm_label_ids, tf.float32)
        mlm_label_weights = labels['mlm_label_weights'] * itm_label_ids
        mpp_label_weights = labels['mpp_label_weights'] * itm_label_ids
      else:
        mlm_label_weights = labels['mlm_label_weights']
        mpp_label_weights = labels['mpp_label_weights']
      metrics = dict([(metric.name, metric) for metric in metrics])
      if 'mlm_accuracy' in metrics:
        metrics['mlm_accuracy'].update_state(
            labels['mlm_label_ids'], model_outputs['mlm_logits'],
            mlm_label_weights)
      if 'mpp_accuracy' in metrics:
        metrics['mpp_accuracy'].update_state(
            labels['mpp_label_ids'], model_outputs['mpp_logits'],
            mpp_label_weights)
      if 'itm_accuracy' in metrics:
        metrics['itm_accuracy'].update_state(
            labels['itm_label_ids'], model_outputs['itm_logits'],
            labels['itm_label_weights'])

  def train_step(self,
                 inputs: Mapping[str, tf.Tensor],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: tf.keras.metrics.Metric):
    """Does forward and backward pass.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.

    """
    inputs, labels = inputs
    with tf.GradientTape() as tape:
      outputs = model(**inputs, training=True)
      # Computes per-replica loss.
      loss = self.build_losses(
          labels=labels,
          model_outputs=outputs,
          metrics=metrics,
          aux_losses=model.losses)
      if self.task_config.scale_loss:
        # Scales loss as the default gradients allreduce performs sum inside the
        # optimizer.
        scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
    tvars = model.trainable_variables
    if self.task_config.scale_loss:
      grads = tape.gradient(scaled_loss, tvars)
    else:
      grads = tape.gradient(loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(metrics, labels, outputs)
    return {self.loss: loss}

  def validation_step(self,
                      inputs: Mapping[str, tf.Tensor],
                      model: tf.keras.Model,
                      metrics: tf.keras.metrics.Metric):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.

    """
    inputs, labels = inputs
    outputs = self.inference_step(inputs, model)
    loss = self.build_losses(
        labels=labels,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    self.process_metrics(metrics, labels, outputs)
    return {self.loss: loss}

  def inference_step(self, inputs, model: tf.keras.Model):
    """Performs the forward step.

    With distribution strategies, this method runs on devices.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.

    Returns:
      Model outputs.

    """
    return model(**inputs, training=False)
