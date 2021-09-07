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

"""Mmt model configurations."""
from typing import List, Optional

import dataclasses
from official.modeling.hyperparams import base_config

from configs import encoders


@dataclasses.dataclass
class ClsHeadConfig(base_config.Config):
  inner_dim: int = 0
  num_classes: int = 2
  activation: Optional[str] = "tanh"
  dropout_rate: float = 0.0
  cls_token_idx: int = 0
  name: Optional[str] = None


@dataclasses.dataclass
class PretrainModelConfig(base_config.Config):
  """Pretrain model configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  mlm_activation: str = 'gelu'
  mlm_initializer: str = 'glorot_uniform'
  mpp_activation: str = 'gelu'
  mpp_initializer: str = 'glorot_uniform'
  cls_heads: List[ClsHeadConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ClassificationModelConfig(base_config.Config):
  """A classifier/regressor configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  num_classes: int = 0
  cls_heads: List[ClsHeadConfig] = dataclasses.field(default_factory=list)
