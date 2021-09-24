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

"""Classification task."""
import collections
import dataclasses
from typing import List, Union, Optional

from absl import logging
import numpy as np
import orbit
from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.nlp.data import data_loader_factory
from official.nlp.modeling import layers
import tensorflow as tf

import input_utils
from configs import mmt
from configs import encoders
from modeling import models
from modeling import losses


METRIC_TYPES = frozenset(['accuracy', 'auc'])


@dataclasses.dataclass
class ClassificationConfig(cfg.TaskConfig):
  """The classification model config."""

  model: mmt.ClassificationModelConfig = mmt.ClassificationModelConfig()
  scale_loss: bool = False

  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()

  init_checkpoint: str = ''
  init_cls_pooler: bool = False
  metric_type: str = 'accuracy'


@task_factory.register_task_cls(ClassificationConfig)
class ClassificationTask(base_task.Task):
  """Task object for classification."""

  def __init__(self,
               params: cfg.TaskConfig,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    super().__init__(params, logging_dir, name=name)
    if params.metric_type not in METRIC_TYPES:
      raise ValueError(f'Invalid metric_type: {params.metric_type}')
    self.metric_type = params.metric_type
    self.label_field = params.train_data.label_field or 'label_ids'
    self.logits_field = params.train_data.logits_field or 'logits'
    self.label_weights_field = (params.train_data.label_weights_field or
                                'label_weights')
    self.pos_weights_field = (params.train_data.pos_weights_field or
                              'pos_weights')

    self.task_name = 'classification'

  def _build_encoder(self, encoder_cfg):
    return encoders.build_encoder(encoder_cfg)

  def build_model(self):
    config = self.task_config.model
    encoder_cfg = config.encoder
    encoder = self._build_encoder(encoder_cfg)

    data_cfg = self.task_config.train_data

    cls_heads = []
    for cfg in config.cls_heads:
      cls_head = layers.ClassificationHead(**cfg.as_dict())
      cls_heads.append(cls_head)

    model = models.MmtClassificationModel(
        encoder=encoder,
        classification_heads=cls_heads)

    inputs = input_utils.create_mmt_encoder_inputs(data_cfg)
    model(**inputs)

    return model

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> tf.Tensor:

    label_ids = labels[self.label_field]
    logits = model_outputs[self.logits_field]

    label_weights = labels[self.label_weights_field]
    pos_weights = labels[self.pos_weights_field]

    if self.task_config.model.num_classes == 1:
      loss_fn = losses.weighted_binary_crossentropy_loss
      logits = tf.reshape(logits, (-1,))
    else:
      loss_fn = losses.weighted_sparse_categorical_crossentropy_loss

    loss = loss_fn(logits, label_ids, label_weights,
                   metrics, self.task_name, pos_weights=pos_weights)

    total_loss = loss

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for classification task."""
    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    if self.task_config.model.num_classes == 1:
      metrics = [
          tf.keras.metrics.AUC(name='auc', curve='PR')
      ]
    elif self.task_config.model.num_classes == 2:
      metrics = [
          tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
          tf.keras.metrics.AUC(name='auc', curve='PR'),
      ]
    else:
      metrics = [
          tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
      ]
    metrics.append(tf.keras.metrics.Mean(name=f'{self.task_name}_loss'))
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    label_ids = labels[self.label_field]
    logits = model_outputs[self.logits_field]
    label_weights = labels[self.label_weights_field]

    with tf.name_scope('ClassificationTask/process_metrics'):
      for metric in metrics:
        if metric.name == 'auc':
          if self.task_config.model.num_classes == 1:
            logits = tf.reshape(logits, (-1,))
            probs = tf.sigmoid(logits)
          elif self.task_config.model.num_classes == 2:
            # Converts the logit to probability and extract the prob of True.
            probs = tf.nn.softmax(logits)[:, 1]
          else:
            raise ValueError('auc requires # classes either 1 or 2.')

          metric.update_state(label_ids, probs, label_weights)

        if metric.name == 'cls_accuracy':
          metric.update_state(label_ids, logits, label_weights)

  def train_step(self,
                 inputs: Tuple[Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Mapping[str, tf.keras.metrics.Metric]):
    """Does forward and backward pass.

    Args:
      inputs: a pair of dictionaries of input and label tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.

    """

    inputs, labels = inputs

    with tf.GradientTape() as tape:
      # Computes per-replica loss.
      outputs = model(**inputs, training=True)
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

  def validation_step(self, inputs, model: tf.keras.Model, metrics):
    inputs, labels = inputs
    outputs = self.inference_step(inputs, model)
    loss = self.build_losses(
        labels=labels,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    logs = {self.loss: loss}
    self.process_metrics(metrics, labels, outputs)
    return logs

  def inference_step(self, inputs, model: tf.keras.Model):
    return model(**inputs, training=False)

  def initialize(self, model):
    """Loads a pretrained checkpoint (if exists)."""

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if not ckpt_dir_or_file:
      logging.info('task_config.init_checkpoint is empty. Train from stratch.')
      return
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    pretrain2finetune_mapping = {
        'encoder': model.checkpoint_items['encoder'],
    }

    # Restores pretrained cls_heads from the checkpoint if the cls_heads exist in
    # the finetuning model.
    if self.task_config.model.cls_heads:
      for cls_head in self.task_config.model.cls_heads:
        for key, item in model.checkpoint_items.items():
          if cls_head.name in key:
            pretrain2finetune_mapping[key] = item
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info(f'Finished loading pretrained checkpoint from {ckpt_dir_or_file}.')


def predict(task: ClassificationTask,
            params: cfg.DataConfig,
            model: tf.keras.Model) -> List[Union[int, float]]:
  """Predicts on the input data.

  Args:
    task: A `ClassificationTask` object.
    params: A `cfg.DataConfig` object.
    model: A keras.Model.

  Returns:
    A list of predictions with length of `num_examples`.

  """

  RawResult = collections.namedtuple('RawResult',
                                     ['image_index',
                                      'text_index',
                                      'gt_image_index',
                                      'output'])

  strategy = tf.distribute.get_strategy()

  def get_raw_results(predictions):
    """Converts multi-replica predictions to RawResult."""
    for img_idx, txt_idx, gt_img_idx, logits in zip(predictions['image_index'],
                                                    predictions['text_index'],
                                                    predictions['gt_image_index'],
                                                    predictions['logits']):

      if task.task_config.model.num_classes == 1:
        outputs = tf.sigmoid(logits)
      elif task.task_config.model.num_classes == 2:
        # Gets the scores of the positive class (num_classes == 2).
        outputs = tf.nn.softmax(logits, axis=1)[:, 1]
      else:
        # Gets the classes with maximum scores (num_classes >= 2).
        outputs = tf.argmax(logits, axis=1)
  
      for values in zip(img_idx.numpy(), txt_idx.numpy(),
                        gt_img_idx.numpy(), outputs.numpy()):
        yield RawResult(image_index=values[0],
                        text_index=values[1],
                        gt_image_index=values[2],
                        output = values[3].tolist())

  @tf.function
  def predict_step(batch):
    """Replicates prediction calculation."""

    def _replicated_step(inputs):
      inputs, labels = inputs
      image_index = inputs.pop('image_index')
      text_index = inputs.pop('text_index')
      gt_image_index = inputs.pop('gt_image_index')
      outputs = task.inference_step(inputs, model)
      return dict(image_index=image_index,
                  text_index=text_index,
                  gt_image_index=gt_image_index,
                  logits=outputs['logits'])

    outputs = strategy.run(_replicated_step, args=(batch,))
    return tf.nest.map_structure(strategy.experimental_local_results, outputs)

  dataset = orbit.utils.make_distributed_dataset(strategy,
                                                 task.build_inputs,
                                                 params)
  dataset = iter(dataset)

  results = []
  for step, batch in enumerate(dataset, start=1):
    predictions = predict_step(batch)
    results.extend(list(get_raw_results(predictions)))

    if step % 5 == 0:
      logging.info(f'Made predictions for {len(results)} examples.')

  logging.info(f'Finished predictions for {len(results)} examples.')
  return results
