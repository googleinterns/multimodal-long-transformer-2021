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

"""Pretraining Mmt experiment configurations."""
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization

from tasks import pretraining
from data import pretrain_dataloader


_TRAINER = cfg.TrainerConfig(
    train_steps=1000000,
    optimizer_config=optimization.OptimizationConfig({
        'optimizer': {
            'type': 'adamw',
            'adamw': {
                'weight_decay_rate':
                    0.01,
                'exclude_from_weight_decay': [
                    'LayerNorm', 'layer_norm', 'bias'
                ],
            }
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 1e-4,
                'end_learning_rate': 0.0,
            }
        },
        'warmup': {
            'type': 'polynomial',
        }
    }))


@exp_factory.register_config_factory('mmt/pretraining')
def mmt_pretraining() -> cfg.ExperimentConfig:
  """mmt pretraining experiment."""
  config = cfg.ExperimentConfig(
      task=pretraining.PretrainingTaskConfig(
          train_data=pretrain_dataloader.MmtPretrainDataConfig(),
          validation_data=pretrain_dataloader.MmtPretrainDataConfig(
              is_training=False)),
      trainer=_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
