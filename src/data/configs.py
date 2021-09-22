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

import dataclasses

from official.core import config_definitions as cfg


@dataclasses.dataclass
class MmtDataConfig(cfg.DataConfig):
  """Data config for all Mmt tasks."""

  seed: int = 128
  input_path: str = ''
  num_examples: int = 0
  vocab_filename: str = ''
  is_training: bool = True
  global_batch_size: int = 256

  use_image_text_matching_label: bool = True
  image_key: str = 'image_data'
  text_key_dict: str = ('{"caption_attribution_description": "[ATT]",'
                        ' "caption_reference_description":"[REF]"}')
  tasks: str = ''
  image_size: int = 224
  patch_size: int = 16
  patch_order: int = 'raster_scan'
  max_pixel_val: int = 256
  max_seq_len: int = 512

  #TODO(roylu): should be defined in the model config only.
  relative_pos_max_distance: int = 12
  relative_att_num_core_layers: int = 0

  label_field: str = None 
  label_weights_field: str = None
  logits_field: str = None
  pos_weights_field: str = None
