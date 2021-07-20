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

import attr
from typing import Text, Mapping, List

import tensorflow as tf
import tensorflow_text as tf_text

from google_research.etcmodel.models.modeling import EtcConfig 
from input_utils import PretrainInputConfig, get_pretrain_example_decode_fn


def input_fn_builder(tokenizer: tf_text.BertTokenizer,
                     input_filenames: List,
                     input_config: PretrainInputConfig,
                     model_config: EtcConfig,
                     is_training: bool,
                     num_cpu_threads: int = 4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  vocab = tokenizer._wordpiece_tokenizer._get_vocab_and_ids()[0]
  vocab = vocab.numpy().tolist()
  
  # Unused tokens are used as special tokens that indicate the types of 
  # subsequences. For example, [unused0] is for [PATCH] which will be 
  # added prior to the sequence of patch tokens.
  unselectable_tokens = [b'[CLS]', b'[SEP]', b'[unused0]']
  unselectable_tokens += [f'[unused{i}]'.encode() 
                          for i in range(1, len(input_config.text_keys)+1)]
  unselectable_ids = list(map(vocab.index, unselectable_tokens))
  mask_token_id = vocab.index(b'[MASK]')

  text_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=input_config.mlm_max_selections_per_batch,
      selection_rate=input_config.mlm_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  image_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=input_config.mpp_max_selections_per_batch,
      selection_rate=input_config.mpp_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  mask_values_chooser = tf_text.MaskValuesChooser(len(vocab),
                                                  mask_token_id, 0.8)
  num_patch_per_row = input_config.image_size // input_config.patch_size

  def make_masked_lm_features(features: Mapping[Text, tf.Tensor]):
    """Randomly mask out tokens
    Args:
      features: A dictionary of Tensor features, crucially including
          `image_input_ids`, `text_input_ids`.

    Returns:
      A new `features` dictionary with global-local transformer side inputs.
    """

    features = dict(features)

    # image_input_ids: <tf.RaggedTensor>[batch, (patches), (1)].
    image_input_ids = features['image_input_ids']
    image_features = tf_text.mask_language_model(
        image_input_ids, image_item_selector, mask_values_chooser, axis=1)

    # text_input_ids: <tf.RaggedTensor>[batch, (words), (wordpieces)].
    text_input_ids = features['text_input_ids']
    if not input_config.mlm_use_whole_word:
      text_input_ids = text_input_ids.merge_dims(-2, -1)
    text_features = tf_text.mask_language_model(
        text_input_ids, text_item_selector, mask_values_chooser, axis=1)

    # Offset text positions after image ones ([CLS] [PATCH] P1 p2 ... P196).
    text_positions = text_features[1] + 2 + num_patch_per_row ** 2
    masked_input_ids = tf.concat([image_features[0], text_features[0]], axis=0)
    masked_positions = tf.concat([image_features[1], text_positions], axis=0)
    masked_ids = tf.concat([image_features[2], text_features[2]], axis=0)

    features['token_ids'] = masked_input_ids
    features['masked_lm_positions'] = masked_positions
    features['masked_lm_ids'] = masked_ids
    features['masked_lm_weights'] = tf.where(masked_input_ids == mask_token_id,
                                             1, 0) 
    return features

  def input_fn(params):
    """The actual input function."""

    batch_size = params['batch_size']

    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_filenames))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_filenames))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_filenames))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_filenames)
      # Since we evaluate for a fixed number of steps we don't want to
      # encounter out-of-range exceptions.
      d = d.repeat()

    # roylu TODO
    # Filter out examples with empty texts 

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we'd prefer *not* to drop remainder
    # so we don't lose examples, but since we typically run eval on TPUs also,
    # we set drop_remainder=True.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            get_pretrain_example_decode_fn(
                tokenizer,
                input_config,
                model_config,
                is_training),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(make_masked_lm_features, tf.data.experimental.AUTOTUNE)
    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn
