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

from typing import List, Optional, Mapping

import attr
import tensorflow as tf
import tensorflow_text as tf_text
from einops import reduce, rearrange
from official.modeling import tf_utils
from tensorflow.python.ops import math_ops

import configs
import tensor_utils
from etcmodel import feature_utils as etc_feature_utils


_PATCH_START_UNUSED_INDEX = 99


@attr.s
class PretrainInputConfig(object):
  """Config options for pretraining model input."""

  # Images of image_size * imate_size pixels.
  image_size = attr.ib(default=224)

  # Patches of patch_size * patch_size pixels.
  patch_size = attr.ib(default=16)

  # The order of Patches to feed into transformer.
  patch_order = attr.ib(default='raster_scan')

  # Maximum pixel value.
  max_pixel_val = attr.ib(default=256) 

  # The names of text fields we want to use in the input.
  text_keys = attr.ib(factory=List)

  # Whole word masking for masked language modeling.
  mlm_use_whole_word = attr.ib(default=False)

  # The fraction of tokens to mask for masked language model (mlm) loss.
  mlm_fraction_to_mask = attr.ib(default=0.15)

  # The fraction of tokens to mask for masked patch prediction (mpp) loss.
  mpp_fraction_to_mask = attr.ib(default=0.50)

  # Maximum number of masked text tokens per sequence.
  mlm_max_selections_per_sequence = attr.ib(default=47)
  
  # Maximum number of masked patch tokens per sequence.
  mpp_max_selections_per_sequence = attr.ib(default=98)

  # Output channel bits in masked patch prediction.
  output_channel_bits = attr.ib(default=3)

  # Number of channels of input images.
  input_channels = attr.ib(default=3)

  # Maximum input sequence length (image+text) after WordPiece tokenization.
  max_seq_len = attr.ib(default=512)


@attr.s
class RelativeTransformerSideInputs(object):
  """RelativeTransformer side inputs ("att_mask" and "relative_att_ids").

  See `RelativeTransformerLayers.call()` for a description of these side
  inputs.

  """

  att_mask = attr.ib()  # type: Optional[tf.Tensor]
  relative_att_ids = attr.ib()  # type: Optional[tf.Tensor]

  def to_dict(self):
    """Returns attributes in a Python dictionary."""
    return attr.asdict(self, filter=lambda a, v: v is not None)


def get_decode_fn(tokenizer: tf_text.BertTokenizer,
                  input_config: PretrainInputConfig,
                  model_config: configs.MmtConfig,
                  is_training: bool):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  patch_size = input_config.patch_size
  max_seq_len = input_config.max_seq_len
  num_patch_per_row = image_size // patch_size
  num_patches = num_patch_per_row ** 2
  vocab = tokenizer._wordpiece_tokenizer._get_vocab_and_ids()[0]
  vocab = vocab.numpy().tolist()

  special_token_to_ragged_tensor = {
      'cls': 
          tensor_utils.ragged_full((1, 1, 1), tf.int32, vocab.index(b'[CLS]')),
      'sep':
          tensor_utils.ragged_full((1, 1, 1), tf.int32, vocab.index(b'[SEP]')),
  }

  for i, key in enumerate(['patch'] + input_config.text_keys):
    unused_token = f'[unused{i}]'.encode()
    unused_token_idx = vocab.index(unused_token)
    special_token_to_ragged_tensor[key] = tensor_utils.ragged_full(
        (1, 1, 1), tf.int32, unused_token_idx)

  # We start from unused99 to make special tokens from unused1 to 98 flexible.
  # The large index of unused tokens is 993.
  # Make sure the number of patches is lower than 895 (993 - 99 + 1).
  patch_start_token = f'[unused{_PATCH_START_UNUSED_INDEX}]'.encode()
  patch_start_idx = vocab.index(patch_start_token)

  patch_ids_tensor = tf.expand_dims(
      tf.range(patch_start_idx, num_patches + patch_start_idx),
      axis=1)
  patch_ids_tensor = tf.reshape(patch_ids_tensor, (1, num_patches, 1))
  patch_ids_tensor = tf.RaggedTensor.from_tensor(patch_ids_tensor)

  # -2 is for [CLS] and [PATCH]; -1 is for [SEP] at the end of the sequence.
  max_text_seq_len = (max_seq_len - 2 - num_patches -
                      len(input_config.text_keys) - 1)
  trimmer = tf_text.RoundRobinTrimmer(max_seq_length=[max_text_seq_len])

  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature(
        [], tf.string, default_value='')

  def convert_image_to_patches(im):
    """Converts an image to patches (token embeddings).

    Args:
      im: <float32>[height, width, num_channels].

    Returns:
      <float32>[num_patch_per_row, num_patch_per_row, 3*(patch_size**2)].

    """
    im = tf.expand_dims(im, axis=0)
    im = tf.image.extract_patches(im,
                                  sizes=[1, patch_size, patch_size, 1],
                                  strides=[1, patch_size, patch_size, 1],
                                  rates=[1, 1, 1, 1],
                                  padding='VALID')
    im = tf.squeeze(im, axis=0)
    return im

  def reorder_patches(im, mode='raster_scan'):
    """Reorders the patch order of a image.

    Args:
      im: <float32>[num_patch_per_row, num_patch_per_row, 3*(patch_size**2)].
      mode: Mode of reordering.

    Returns:
      <float32>[num_patches, 3*(patch_size**2)].

    """
    if mode == 'raster_scan':
      return tf.reshape(im, [num_patches, (patch_size**2)*3])
    else:
      raise ValueError(f'Reordering mode ({mode}) is not available.')

  def get_num_wordpieces(t):
    return tf_utils.get_shape_list(t.merge_dims(-2, -1))[0]

  def _decode_fn(record):
    example = tf.io.parse_single_example(record, name_to_features)

    # Image
    # We follow the implementation of ViT.
    im = tf.io.decode_image(example.pop('image_data'), dtype=tf.float32)
    if is_training:
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.85, 1.0),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      im.set_shape([None, None, channels])
      im = tf.image.resize(im, [image_size, image_size])
      if tf.random.uniform(shape=[]) > 0.5:
        im = tf.image.flip_left_right(im)
    else:
      im = tf.image.resize(im, [image_size, image_size])

    im = convert_image_to_patches(im)
    im = reorder_patches(im, mode=input_config.patch_order)
    example['patch_embeddings'] = im

    # Concatenate all input token ids together by the following ordering:
    # [CLS] [PATCH] patch1 patch2 ... [ATTRIBUTION] w_ATT1 w_ATT2 ...
    # [REFERENCE] w_REF1 w_REF2 ... [ALT_TEXT] w_ALT1 w_ALT2 ... [SEP].
    patch_input_ids = [special_token_to_ragged_tensor['cls'],
                       special_token_to_ragged_tensor['patch'],
                       patch_ids_tensor]
    patch_input_ids = tf.squeeze(tf.concat(patch_input_ids, axis=1), axis=0)
    patch_input_ids = tf.RaggedTensor.from_tensor(patch_input_ids)
    example['patch_input_ids'] = patch_input_ids
    
    # Text
    for k in input_config.text_keys:
      example[k] = tokenizer.tokenize(example[k])

    text_input_ids = []
    for k in input_config.text_keys:
      text_input_ids.append(example[k])
      example.pop(k)
    text_input_ids = trimmer.trim(text_input_ids)
    
    # Create text_input_ids.
    # Firstly, insert a special token prior to each text source.
    for i, k in enumerate(input_config.text_keys):
      s = special_token_to_ragged_tensor[k]
      text_input_ids.insert(i*2, s)
    text_input_ids.append(special_token_to_ragged_tensor['sep'])
    text_input_ids = tf.squeeze(tf.concat(text_input_ids, axis=1), axis=0)
    example['text_input_ids'] = text_input_ids

    # Total sequence length: image + text
    example['num_image_wordpieces'] = 2 + num_patches
    example['num_text_wordpieces'] = get_num_wordpieces(text_input_ids)

    return example

  return _decode_fn


def make_relative_transformer_side_inputs(
  long_breakpoints: tf.Tensor,
  relative_pos_max_distance: int,
  name: Optional[str] = None) -> RelativeTransformerSideInputs:
  """Makes relative transformer side input tensors.

  Args:
    long_breakpoints: <int32>[batch_size, long_seq_len] Tensor of ending
      breakpoints separating different packed examples.
    relative_pos_max_distance: Maximum distance to use for relative position
      representations. All larger distances will be clipped to this value. Use
      0 to skip relative position representations entirely.
    name: A name for the operation (optional).

  Returns:
    A `RelativeTransformerSideInputs` with all relevant tensors set.
  """

  with tf.name_scope(name or 'make_relative_transformer_side_inputs'):
    long_breakpoints = tf.convert_to_tensor(long_breakpoints)
    long_example_ids = tf.cumsum(long_breakpoints, axis=-1, reverse=True)
    long_seq_len = tf_utils.get_shape_list(long_example_ids)[1]
    att_mask = etc_feature_utils.make_segmented_att_mask(long_example_ids)
    batch_size = tf_utils.get_shape_list(long_example_ids)[0]
  
    relative_att_ids = None
    if relative_pos_max_distance > 0:
      relative_pos_generator = etc_feature_utils.RelativePositionGenerator(
          relative_pos_max_distance)
      relative_att_ids = relative_pos_generator.make_relative_att_ids(
          seq_len=long_seq_len,
          batch_size=batch_size)
  
    return RelativeTransformerSideInputs(
        att_mask=att_mask,
        relative_att_ids=relative_att_ids)


def add_side_input_features(
  input_config: PretrainInputConfig,
  model_config: configs.MmtConfig,
  features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Replaces raw input features with derived ETC side inputs.

  This function is meant to be called as part of a Dataset pipeline.

  Args:
    model_config: A MmtConfig.
    features: A dictionary of Tensor features, crucially including
      `long_breakpoints`, `num_image_wordpieces`, `num_text_wordpieces`.

  Returns:
    A new `features` dictionary with side inputs.

  """

  max_seq_len = input_config.max_seq_len

  img_wp = features.pop('num_image_wordpieces')
  txt_wp = features.pop('num_text_wordpieces')
  seq_len = img_wp + txt_wp
  batch_size = tf_utils.get_shape_list(img_wp)[0]

  img_wp = img_wp[:, tf.newaxis]
  txt_wp = txt_wp[:, tf.newaxis]
  position = tf.range(max_seq_len, dtype=tf.int32)
  img_segment = tf.where(position < img_wp, 1, 0)

  txt_segment_mask = (position > img_wp) & (position < img_wp + txt_wp)
  txt_segment = tf.where(txt_segment_mask, 2, 0)

  segment_ids = img_segment + txt_segment
  features['segment_ids'] = segment_ids
  
  # seq_len-1 because we need the indices
  features['long_breakpoints'] = tf.one_hot(
      seq_len-1, depth=max_seq_len,
      on_value=1, off_value=0, dtype=tf.int32)

  side_inputs = make_relative_transformer_side_inputs(
      long_breakpoints=features.pop('long_breakpoints'),
      relative_pos_max_distance=model_config.relative_pos_max_distance)

  features.update(side_inputs.to_dict())

  return features


def add_fake_labels(features):
  """Adds dummy labels."""

  batch_size = tf_utils.get_shape_list(features['word_ids'])[0]
  return (features, tf.zeros((batch_size, 1), tf.int32))


def get_masking_fn(tokenizer: tf_text.BertTokenizer,
                   input_config: PretrainInputConfig):
  """Creates a masking_fn (`make_mlm_and_mpp_features`)."""

  text_keys = input_config.text_keys
  max_seq_len = input_config.max_seq_len
  patch_size = input_config.patch_size
  image_size = input_config.image_size
  max_pixel_val = input_config.max_pixel_val
  channels = input_config.input_channels
  output_channel_bits = input_config.output_channel_bits

  mlm_fraction_to_mask = input_config.mlm_fraction_to_mask
  mpp_fraction_to_mask = input_config.mpp_fraction_to_mask
  mlm_max_selections_per_sequence = input_config.mlm_max_selections_per_sequence  
  mpp_max_selections_per_sequence = input_config.mpp_max_selections_per_sequence  

  vocab = tokenizer._wordpiece_tokenizer._get_vocab_and_ids()[0]
  vocab = vocab.numpy().tolist()
  
  # Unused tokens are used as special tokens that indicate the types of 
  # subsequences. For example, [unused0] is for [PATCH] which will be 
  # added prior to the sequence of patch tokens.
  unselectable_tokens = [b'[CLS]', b'[SEP]', b'[unused0]']
  unselectable_tokens += [f'[unused{i}]'.encode() 
                          for i in range(1, len(text_keys)+1)]
  unselectable_ids = list(map(vocab.index, unselectable_tokens))
  mask_token_id = vocab.index(b'[MASK]')

  text_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mlm_max_selections_per_sequence,
      selection_rate=mlm_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  patch_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mpp_max_selections_per_sequence,
      selection_rate=mpp_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  mask_values_chooser = tf_text.MaskValuesChooser(len(vocab),
                                                  mask_token_id, 0.8)
  num_patch_per_row = image_size // patch_size
  num_patches = num_patch_per_row ** 2

  def make_masked_patch_label_ids(masked_patch_embeddings):
    """Makes taget labels for masked patch prediction.
  
    Args:
      masked_patch_embeddings:
        <tf.Tensor>[batch, sequence_length, patch_embedding_size].
  
    Returns:
      <tf.Tensor>[batch, sequence_length].

    """

    bin_size = max_pixel_val // (2 ** output_channel_bits)
  
    # Scale from 0-1 to 0-255
    masked_patch_embeddings = masked_patch_embeddings * (max_pixel_val - 1)
    avg_target = reduce(
        masked_patch_embeddings,
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
    label_ids = bin_mask * discretized_target
    label_ids = reduce(
        label_ids,
        'batch seq_len channels -> batch seq_len',
        'sum')
    return label_ids

  def get_masked_weights(masked_token_ids: tf.Tensor,
                         masked_seq_len: int) -> tf.Tensor:
    """Gets mask for real predictions.

    The `positions` tensor might be zero-padded (if the sequence is too short 
    to have the maximum number of predictions).
    The `masked_weights` tensor has a value of 1.0 for every real prediction 
    and 0.0 for the padding predictions.

    Args:
      masked_token_ids: A tensor that contains mask token ids.
    Returns:
      A mask that indicate the positions of real predictions.

    """
    mask_position = tf.equal(masked_token_ids, mask_token_id)
    mask_position = tf.cast(mask_position, tf.int32)
    num_real_masked_tokens = tf.reduce_sum(mask_position, axis=1, keepdims=True)
    position = tf.range(masked_seq_len, dtype=tf.int32)
    return tf.where(position < num_real_masked_tokens, 1.0, 0.0)
    
  def pad_to_max_seq_len(tensor, max_seq_len):
    seq_len = tf_utils.get_shape_list(tensor)[1]
    lack_seq_len = max_seq_len - seq_len
    tensor = tf.pad(tensor, paddings=[[0, 0], [0, lack_seq_len]])
    return tensor

  def make_mlm_and_mpp_features(
    features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Makes features for masked language model and masked patch prediction.

    Args:
      features: A dictionary of Tensor features, crucially including
          `patch_input_ids`, `text_input_ids`. The two tensors will be popped
          out since we don't need them during training.

    Returns:
      A new `features` dictionary with `word_ids`, `masked_patch_positions`,
      `masked_patch_label_ids`, `masked_patch_label_weights`, 
      `masked_text_positions`, `masked_text_label_ids`,
      `masked_text_label_weights`.

    """

    # Patch: create features for masked patch prediction (mpp)
    # patch_input_ids: <tf.RaggedTensor>[batch, (patches), (1)].
    (masked_patch_token_ids,
     masked_patch_positions, _) = tf_text.mask_language_model(
        features.pop('patch_input_ids'),
        patch_item_selector,
        mask_values_chooser,
        axis=1)

    # Offset -2 position indices: [CLS] and [PATCH] tokens.
    # We want to select from patch embeddings.
    shifted_patch_masked_positions = (masked_patch_positions - 2).to_tensor()
    shifted_patch_masked_positions = tf.cast(shifted_patch_masked_positions,
                                             tf.int32)
    batch_size, _, patch_embedding_size = tf_utils.get_shape_list(
        features['patch_embeddings'])
    masked_patch_embeddings = tensor_utils.gather_indexes(
      features['patch_embeddings'], shifted_patch_masked_positions)
    masked_patch_embeddings = rearrange(masked_patch_embeddings,
                                        '(b s) h -> b s h',
                                        b=batch_size)
  
    masked_patch_positions = masked_patch_positions.to_tensor()
    masked_patch_positions = tf.cast(masked_patch_positions, tf.int32)
    masked_patch_positions = pad_to_max_seq_len(
        masked_patch_positions, mpp_max_selections_per_sequence)
    features['masked_patch_positions'] = masked_patch_positions
                                    
    # Create label for masked patche prediction.
    masked_patch_label_ids = make_masked_patch_label_ids(
        masked_patch_embeddings)
    masked_patch_seq_len = tf_utils.get_shape_list(masked_patch_positions)[1]
    features['masked_patch_label_weights'] = get_masked_weights(
        masked_patch_token_ids, masked_patch_seq_len)
    features['masked_patch_label_ids'] = masked_patch_label_ids

    # Zeros out patch embeddings that are masked.
    # The 1st and 2nd token ids in masked_patch_token_ids are [CLS] and [PATCH].
    sliced_masked_patch_token_ids = tf.slice(
        masked_patch_token_ids.to_tensor(),
        begin=[0, 2],
        size=[batch_size, num_patches])
    non_masked_token_position = tf.not_equal(sliced_masked_patch_token_ids,
                                             mask_token_id)
    non_masked_token_position = tf.cast(non_masked_token_position, tf.float32)
    non_masked_token_position = tf.expand_dims(non_masked_token_position,
                                               axis=-1)
    features['patch_embeddings'] *= non_masked_token_position

    # Text: create features for masked language model (mlm).
    # text_input_ids: <tf.RaggedTensor>[batch, (words), (wordpieces)].
    text_input_ids = features.pop('text_input_ids')
    if not input_config.mlm_use_whole_word:
      text_input_ids = text_input_ids.merge_dims(-2, -1)
    (masked_text_token_ids,
     masked_text_positions, masked_text_label_ids) = tf_text.mask_language_model(
        text_input_ids, text_item_selector, mask_values_chooser, axis=1)
        
    # Offset text positions because we append text tokens after patch tokens
    # ([CLS] [PATCH] P1 p2 ... P196).
    masked_text_positions = masked_text_positions + 2 + num_patches
    masked_text_positions = tf.cast(masked_text_positions, tf.int32).to_tensor()
    masked_text_positions = pad_to_max_seq_len(
        masked_text_positions, mlm_max_selections_per_sequence)
    features['masked_text_positions'] = masked_text_positions
    
    masked_text_label_ids = masked_text_label_ids.to_tensor()
    masked_text_label_ids = pad_to_max_seq_len(
        masked_text_label_ids, mlm_max_selections_per_sequence)
    features['masked_text_label_ids'] = masked_text_label_ids

    text_masked_seq_len = tf_utils.get_shape_list(masked_text_positions)[1]
    masked_text_label_weights = get_masked_weights(
        masked_text_token_ids, text_masked_seq_len)
    masked_text_label_weights = pad_to_max_seq_len(
        masked_text_label_weights, mlm_max_selections_per_sequence)
    features['masked_text_label_weights'] = masked_text_label_weights 
        
    # Join text and image token_ids.
    word_ids = tf.concat(
        [masked_patch_token_ids, masked_text_token_ids], axis=1)
    word_ids = word_ids.to_tensor()
    word_ids = pad_to_max_seq_len(word_ids, max_seq_len)
    features['word_ids'] = word_ids
  
    return features
  
  return make_mlm_and_mpp_features
