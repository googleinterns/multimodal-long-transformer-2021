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

"""Finetuning experiment configurations."""
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization

from data import classification_dataloader
from data import retrieval_dataloader
from tasks import classification


@exp_factory.register_config_factory('mmt/classification')
def mmt_classification() -> cfg.ExperimentConfig:
  """Mmt Classification."""
  config = cfg.ExperimentConfig(
      task=classification.ClassificationConfig(
          train_data=classification_dataloader.MmtClassificationDataConfig(),
          validation_data=classification_dataloader.MmtClassificationDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 3e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  config.task.model.encoder.type = 'mmt'
  return config


@exp_factory.register_config_factory('mmt/retrieval')
def mmt_retrieval() -> cfg.ExperimentConfig:
  """Mmt Retrieval."""
  config = cfg.ExperimentConfig(
      task=classification.ClassificationConfig(
          train_data=retrieval_dataloader.MmtRetrievalDataConfig(),
          validation_data=retrieval_dataloader.MmtRetrievalDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate':
                          0.01,
                      'exclude_from_weight_decay':
                          ['LayerNorm', 'layer_norm', 'bias'],
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 3e-5,
                      'end_learning_rate': 0.0,
                  }
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  config.task.model.encoder.type = 'mmt'
  return config
