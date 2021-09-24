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

from typing import Optional, List, Callable, Union

import tensorflow as tf
from official.nlp.keras_nlp import layers

from modeling import layers as mmt_layers


class MmtPretrainingModel(tf.keras.Model):
  """Multimodal Transformer-based pretraining model.

  Adds the Masked Language Modeling (MLM), Masked Patch Prediction (MPP) and
  optional classification heads upon the transformer encoder.

  """

  def __init__(self,
               encoder: tf.keras.Model,
               mpp_output_num_classes: Optional[int] = None,
               mlm_activation: Optional[Union[Callable, str]] = None,
               mlm_initializer: str = 'glorot_uniform',
               mpp_activation: Optional[Union[Callable, str]] = None,
               mpp_initializer: str ='glorot_uniform',
               classification_heads: Optional[List[tf.keras.layers.Layer]] = None,
               bind_word_embedding_table: bool = True,
               name: str = 'mmt_pretraining_model',
               **kwargs):
    """instantiates `MmtPretrainingModel`.

    Args:
      encoder: A transformer network. This network should output a sequence of 
        contextualized word embeddings (sequence_output).
      mlm_activation: The activation (if any) to use in the masked LM network. If
        None, no activation will be used.
      mlm_initializer: The initializer (if any) to use in the masked LM. Default
        to a Glorot uniform initializer.
      mpp_activation: The activation (if any) to use in the masked PP network. If
        None, no activation will be used.
      mpp_initializer: The initializer (if any) to use in the masked PP. Default
        to a Glorot uniform initializer.
      classification_heads: A list of optional head layers to transform on encoder
        sequence outputs.
      bind_word_embedding_table: Bind input word embedding layer and output layer
        of MLM.
      name: The name of the model.

    """

    super(MmtPretrainingModel, self).__init__(name=name, **kwargs)
    self._config = {
        'encoder': encoder,
        'mpp_output_num_classes': mpp_output_num_classes,
        'mlm_initializer': mlm_initializer,
        'mpp_initializer': mpp_initializer,
        'classification_heads': classification_heads,
        'name': name,
    }
    self.encoder = encoder
    self.classification_heads = classification_heads or []
    if len(set([cls.name for cls in self.classification_heads])) != len(
        self.classification_heads):
      raise ValueError('Classification heads should have unique names.')

    embedding_layer = self.encoder.get_word_embedding_layer()
    if bind_word_embedding_table:
      embedding_table = embedding_layer.embedding_table
    else:
      self.mlm_embedding_table = embedding_table = self.add_weight(
        name='mlm_embedding_table',
        shape=[embedding_layer.vocab_size,
               embedding_layer.embedding_size],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=embedding_layer.initializer_range),
        trainable=True)

    self.masked_lm = layers.MaskedLM(
        embedding_table=embedding_table,
        activation=mlm_activation,
        initializer=mlm_initializer,
        output='logits',
        name='mlm')

    self.masked_pp = mmt_layers.MaskedPP(
        output_num_classes=mpp_output_num_classes,
        activation=mpp_activation,
        initializer=mpp_initializer,
        output='logits',
        name='mpp')

  def call(self,
           word_ids: tf.Tensor,
           segment_ids: Optional[tf.Tensor] = None,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           patch_embeddings: Optional[tf.Tensor] = None,
           mlm_positions: Optional[tf.Tensor] = None,
           mpp_positions: Optional[tf.Tensor] = None,
           training: Optional[bool] = None):
    """Calls MmtPretrainingModel.

    Args: Inputs defined by the encoder network, plus `mlm_positions`
      and `mpp_positions` as a dictionary.
      mlm_positions: <tf.int32>[batch_size, masked_text_seq_len].
      mpp_positions: <tf.int32>[batch_size, masked_patch_seq_len].
  
    Returns: A dictionary of `mlm_logits`, `mpp_logits`, classification head 
      outputs keyed by head names, and also outputs from `encoder_network`,
      keyed by `sequence_output`.
      
    """

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

    # Inference may not have mlm_positions or mpp_position
    # Thus, mlm_logits and mpp_logits are not needed.
    if mlm_positions is not None:
      outputs['mlm_logits'] = self.masked_lm(
          sequence_output, masked_positions=mlm_positions)

    if mpp_positions is not None:
      outputs['mpp_logits'] = self.masked_pp(
          sequence_output, masked_positions=mpp_positions)

    for cls_head in self.classification_heads:
      cls_outputs = cls_head(sequence_output)
      outputs[f'{cls_head.name}_logits'] = cls_outputs

    return outputs

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        encoder=self.encoder,
        masked_lm=self.masked_lm,
        masked_pp=self.masked_pp,
    )
    for head in self.classification_heads:
      for key, item in head.checkpoint_items.items():
        items[f'{head.name}.{key}'] = item
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
