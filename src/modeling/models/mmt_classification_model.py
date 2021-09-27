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

from typing import Optional, List

import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.keras_nlp import layers


class MmtClassificationModel(tf.keras.Model):
  """Multimodal Transformer-based classification model."""

  def __init__(self,
               encoder: tf.keras.Model,
               classification_heads: List[tf.keras.layers.Layer],
               name: str = 'mmt_classification_model',
               **kwargs):
    """Instantiates MmtClassificationModel.

    Args:
      encoder: A transformer network. This network should output a sequence
        output.
      classification_heads: A list of head layers to transform on encoder
        sequence outputs.
      name: The name of the model.

    """

    super(MmtClassificationModel, self).__init__(name=name, **kwargs)
    self._config = {
        'encoder': encoder,
        'classification_heads': classification_heads,
        'name': name,
    }
    self.encoder = encoder
    self.classification_heads = classification_heads
    if len(set([cls.name for cls in self.classification_heads])) != len(
        self.classification_heads):
      raise ValueError('Classification heads should have unique names.')

  def call(self,
           word_ids: tf.Tensor,
           segment_ids: Optional[tf.Tensor] = None,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           patch_embeddings: Optional[tf.Tensor] = None,
           training: Optional[bool] = None):

    outputs = dict()
    
    encoder_outputs = self.encoder(
        word_ids=word_ids,
        segment_ids=segment_ids,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        patch_embeddings=patch_embeddings,
        training=training)
    outputs.update(encoder_outputs)
    sequence_output = outputs['sequence_output']

    for cls_head in self.classification_heads:
      cls_outputs = cls_head(sequence_output)
      outputs[f'{cls_head.name}_logits'] = cls_outputs

    return outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        encoder=self.encoder,
        classification_heads=self.classification_heads
    )
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
