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

import functools
from typing import Mapping, List

import attr
from einops import reduce, rearrange
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.ops import math_ops
from official.modeling import tf_utils

import input_utils
from etcmodel.models import modeling


def input_fn_builder(tokenizer: tf_text.BertTokenizer,
                     input_filenames: List,
                     input_config: input_utils.PretrainInputConfig,
                     model_config: modeling.EtcConfig,
                     is_training: bool,
                     num_cpu_threads: int = 4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  patch_size = input_config.patch_size
  image_size = input_config.image_size
  max_pixel_val = input_config.max_pixel_val
  output_channel_bits = input_config.output_channel_bits

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
  num_patch_per_row = image_size // patch_size

  def make_masked_patch_prediction_target(patch_tokens):
    """Make taget labels for masked patch prediction
  
    Args:
      patch_tokens: <tf.Tensor>[batch, sequence_length, patch_token_size]
  
    Returns:
      <tf.Tensor>[batch, sequence_length]
    """

    channels = tf_utils.get_shape_list(patch_tokens)[-1] // (patch_size**2)
    bin_size = max_pixel_val // (2 ** output_channel_bits)
  
    patch_tokens = patch_tokens * (max_pixel_val - 1)
    avg_target = reduce(
        patch_tokens,
        'batch seq_len (p2 channels) -> batch (seq_len channels)',
        'mean',
        channels=channels,
        p2=patch_size**2)

    channel_bins = list(range(bin_size, max_pixel_val, bin_size))

    discretized_target = math_ops._bucketize(avg_target, channel_bins)
    discretized_target = tf.cast(discretized_target, tf.int32)
    discretized_target = rearrange(
        discretized_target,
        'batch (seq_len channels) -> batch seq_len channels',
        channels=channels)
    
    bin_mask = (2 ** output_channel_bits) ** tf.range(0, channels)
    bin_mask = rearrange(bin_mask, 'channels -> () () channels')
    target_label = bin_mask * discretized_target
    target_label = reduce(
        target_label,
        'batch seq_len channels -> batch seq_len',
        'sum')
    return target_label

  def get_masked_weights(masked_token_ids: tf.Tensor,
                         masked_seq_len: int) -> tf.Tensor:
    """ Get mask for real predictions
    The `positions` tensor might be zero-padded (if the sequence is too short 
    to have the maximum number of predictions).
    The `masked_weights` tensor has a value of 1.0 for every real prediction 
    and 0.0 for the padding predictions.
    Args:
      masked_token_ids: A tensor that contains mask token ids
    Returns:
      A mask that indicate the positions of real predictions
    """
    helper = tf.where(
        masked_token_ids == mask_token_id, 1.0, 0.0).to_tensor()
    num_masked_tokens = tf.math.count_nonzero(
        helper, axis=1, keepdims=True, dtype=tf.int32) 
    helper = tf.range(masked_seq_len, dtype=tf.int32)
    return tf.where(helper < num_masked_tokens, 1.0, 0.0) 

  def make_masked_language_model_and_masked_patch_prediction_features(
    features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Randomly mask out text and patch tokens
    Args:
      features: A dictionary of Tensor features, crucially including
          `image_input_ids`, `text_input_ids`.
    Returns:
      A new `features` dictionary with global-local transformer side inputs.
    """

    features = dict(features)

    # Image: create features for masked patch prediction (mpp)
    # image_input_ids: <tf.RaggedTensor>[batch, (patches), (1)].
    (image_masked_token_ids,
     image_masked_positions, image_masked_ids) = tf_text.mask_language_model(
        features['image_input_ids'],
        image_item_selector,
        mask_values_chooser,
        axis=1)

    # Offset -2 position indices: [CLS] and [PATCH] tokens.
    shifted_image_masked_positions = (image_masked_positions - 2).to_tensor()
    batch_size, image_masked_seq_len = tf_utils.get_shape_list(
        shifted_image_masked_positions)
    shifted_image_masked_positions = tf.cast(shifted_image_masked_positions,
                                             tf.int32)
    patch_token_size = tf_utils.get_shape_list(features['image_data'])[2]
    masked_patch_tensor = gather_indexes(features['image_data'],
                                         shifted_image_masked_positions)
    masked_patch_tensor = tf.reshape(masked_patch_tensor,
                                     [batch_size,
                                      image_masked_seq_len,
                                      patch_token_size])
    
    features['image_masked_positions'] = tf.cast(image_masked_positions,
                                                 tf.int32).to_tensor()
    # Create label for masked patche prediction
    features['image_masked_target'] = make_masked_patch_prediction_target(
        masked_patch_tensor)

    features['image_masked_weights'] = get_masked_weights(
        image_masked_token_ids, image_masked_seq_len)

    # Text: create features for masked language model (mlm)
    # text_input_ids: <tf.RaggedTensor>[batch, (words), (wordpieces)].
    text_input_ids = features['text_input_ids']
    if not input_config.mlm_use_whole_word:
      text_input_ids = text_input_ids.merge_dims(-2, -1)
    (text_masked_token_ids,
     text_masked_positions, text_masked_ids) = tf_text.mask_language_model(
        text_input_ids, text_item_selector, mask_values_chooser, axis=1)
      
    # Offset text positions because we append text tokens after patch tokens
    # ([CLS] [PATCH] P1 p2 ... P196).
    text_masked_positions = text_masked_positions + 2 + num_patch_per_row ** 2
    text_masked_positions = tf.cast(text_masked_positions, tf.int32).to_tensor()
    features['text_masked_positions'] = text_masked_positions
    features['text_masked_target'] = text_masked_ids.to_tensor()

    # Join text and image token_ids
    features['token_ids'] = tf.concat(
        [image_masked_token_ids, text_masked_token_ids], axis=1)

    text_masked_seq_len = tf_utils.get_shape_list(text_masked_positions)[1]
    features['text_masked_weights'] = get_masked_weights(text_masked_token_ids,
                                                         text_masked_seq_len)
        
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
            input_utils.get_pretrain_example_decode_fn(
                tokenizer,
                input_config,
                model_config,
                is_training),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(
        make_masked_language_model_and_masked_patch_prediction_features, 
        tf.data.experimental.AUTOTUNE)

    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(
        functools.partial(input_utils.add_side_input_features,
                          input_config, model_config),
        tf.data.experimental.AUTOTUNE)
    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn
