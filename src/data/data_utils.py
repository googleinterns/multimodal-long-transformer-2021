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

import json
from typing import List, Optional, Mapping

import attr
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.ops import math_ops
from official.modeling import tf_utils
from official.vision.image_classification import augment

from data import configs
from etcmodel import feature_utils as etc_feature_utils
import feature_utils
import tensor_utils


PATCH_START_UNUSED_INDEX = 104

# We copied numbers from here.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def check_input_patterns(input_patterns):
  for input_pattern in input_patterns:
    if not tf.io.gfile.glob(input_pattern):
      raise ValueError(f'{input_pattern} does not match any files.')


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


def get_split_features_fn(input_keys, label_keys):
  def split_features_fn(features):
    """Splits features into two dictionaries, inputs and labels."""
    inputs = {k: features[k] for k in input_keys}
    labels = {k: features[k] for k in label_keys if k in features}
    return inputs, labels
  return split_features_fn


def get_decode_fn(name_to_features: Mapping[str, tf.io.FixedLenFeature],
                  tokenizer: tf_text.BertTokenizer,
                  input_config: configs.MmtDataConfig,
                  merge_dims: bool,
                  is_training: bool,
                  decode_image: bool = True,
                  decode_text: bool = True):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  patch_size = input_config.patch_size
  max_seq_len = input_config.max_seq_len
  max_pixel_val = input_config.max_pixel_val
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
  text_special_token_field_dict = json.loads(
      input_config.text_special_token_field_dict)
  text_special_token_field_dict = {
      k: v.encode() for k, v in text_special_token_field_dict.items()}
  patch_key_dict = {'patch': b"[PATCH]"}

  for key, token in {**patch_key_dict, **text_special_token_field_dict}.items():
    token_idx = vocab.index(token)
    special_token_to_ragged_tensor[key] = tensor_utils.ragged_full(
        (1, 1, 1), tf.int32, token_idx)

  # We start from unused99 (index 104).
  # The large index of unused tokens is 993.
  # Make sure the number of patches is lower than 895 (993 - 99 + 1).
  patch_ids_tensor = tf.expand_dims(
      tf.range(PATCH_START_UNUSED_INDEX,
               num_patches + PATCH_START_UNUSED_INDEX), axis=1)
  patch_ids_tensor = tf.reshape(patch_ids_tensor, (1, num_patches, 1))
  patch_ids_tensor = tf.RaggedTensor.from_tensor(patch_ids_tensor)

  # -2 is for [CLS] and [PATCH]; -1 is for [SEP] at the end of the sequence.
  max_text_seq_len = (max_seq_len - len(text_special_token_field_dict) - 1)
  # 2 is for [CLS] and [PATCH].
  max_text_seq_len -= (2 + num_patches)
  trimmer = tf_text.RoundRobinTrimmer(max_seq_length=[max_text_seq_len])

  # Remaining part: [SEP] + Text + [SEP].
  max_remaining_seq_len = max_seq_len - num_patches - 2
  if use_rand_aug:
    rand_aug = augment.RandAugment(num_layers=1)
    # We don't use Invert and Cutout in RandAug.
    #'Invert': Color change might hurts image-text retrieval.
    #'Cutout': We might cut out important objects.
    rand_aug.available_ops = [
        'AutoContrast',
        'Equalize',
        'Rotate',
        'Posterize',
        'Solarize',
        'Color',
        'Contrast',
        'Brightness',
        'Sharpness',
        'ShearX',
        'ShearY',
        'TranslateX',
        'TranslateY',
        'SolarizeAdd'
    ]

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

  def decode_fn(record):
    example = tf.io.parse_single_example(record, name_to_features)

    if decode_image:
      # TODO(roylu): Unify the name of index during preprocessing.
      if 'index' in example:
        example['image_index'] = tf.cast(example.pop('index'), tf.int32)
      if 'image_index' in example:
        example['image_index'] = tf.cast(example.pop('image_index'), tf.int32)

      im = tf.io.decode_image(example.pop('image_data'),
                              dtype=tf.float32, 
                              expand_animations=False)

      if is_training and use_rand_aug:
        im = tf.image.convert_image_dtype(im, dtype=tf.uint8)
        im = rand_aug.distort(im)
        im = tf.image.convert_image_dtype(im, dtype=tf.float32)

      norm_im = (im - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_MEAN
      norm_im = tf.image.resize(norm_im, [image_size, image_size])

      im = tf.image.resize(im, [image_size, image_size])

      if is_training and tf.random.uniform(shape=[]) > 0.5:
        im = tf.image.flip_left_right(im)
        norm_im = tf.image.flip_left_right(norm_im)

      im = convert_image_to_patches(im)
      im = reorder_patches(im, mode=input_config.patch_order)

      # For generating MPP labels.
      if keep_unnormalized_patch_embeddings:
        example['unnormalized_patch_embeddings'] = im

      example['patch_embeddings'] = norm_im

      # Concatenate all input token ids together by the following ordering:
      # [CLS] [PATCH] patch1 patch2 ... [ATTRIBUTION] w_ATT1 w_ATT2 ...
      # [REFERENCE] w_REF1 w_REF2 ... [ALT_TEXT] w_ALT1 w_ALT2 ... [SEP].
      patch_token_ids = [special_token_to_ragged_tensor['cls'],
                         special_token_to_ragged_tensor['patch'],
                         patch_ids_tensor]
      patch_token_ids = tf.squeeze(tf.concat(patch_token_ids, axis=1), axis=0)
      patch_token_ids = tf.RaggedTensor.from_tensor(patch_token_ids)

      # If we apply masking function later, merge_dims should be False here.
      if merge_dims:
        example['patch_token_ids'] = patch_token_ids.merge_dims(-2, -1)
      else:
        example['patch_token_ids'] = patch_token_ids

      example['num_image_wordpieces'] = tf.constant(2 + num_patches)

    if decode_text:

      # TODO(roylu): Unify the name of index during preprocessing.
      if 'index' in example:
        example['text_index'] = tf.cast(example.pop('index'), tf.int32)
      if 'text_index' in example:
        example['text_index'] = tf.cast(example.pop('text_index'), tf.int32)

      if 'gt_image_index' in example:
        example['gt_image_index'] = tf.cast(example.pop('gt_image_index'),
                                            tf.int32)

      for k in text_special_token_field_dict.keys():
        example[k] = tokenizer.tokenize(example[k])

      text_token_ids = []
      for k in text_special_token_field_dict.keys():
        text_token_ids.append(example.pop(k))
      text_token_ids = trimmer.trim(text_token_ids)

      # Create text_token_ids.
      # Firstly, insert a special token prior to each text source.
      for i, k in enumerate(text_special_token_field_dict.keys()):
        s = special_token_to_ragged_tensor[k]
        text_token_ids.insert(i*2, s)
      text_token_ids.append(special_token_to_ragged_tensor['sep'])
      text_token_ids = tf.squeeze(tf.concat(text_token_ids, axis=1), axis=0)

      example['num_text_wordpieces'] = get_num_wordpieces(text_token_ids)

      # If we apply masking function later, merge_dims should be False here.
      if merge_dims:
        text_token_ids = text_token_ids.merge_dims(-2, -1)
        text_token_ids = tensor_utils.pad_to_max_seq_len(
            text_token_ids, max_remaining_seq_len)
        example['text_token_ids'] = text_token_ids
      else:
        example['text_token_ids'] = text_token_ids

    return example

  return decode_fn


def get_add_side_input_features_fn(input_config: configs.MmtDataConfig,
                                   relative_pos_max_distance,
                                   relative_att_num_core_layers):

  if relative_att_num_core_layers > 0:
    image_size = input_config.image_size
    patch_size = input_config.patch_size
    num_patch_per_row = image_size // patch_size
    num_patches = num_patch_per_row ** 2

    relative_pos_generator = feature_utils.MmtRelativePositionGenerator(
      num_patch_per_row,
      relative_att_num_core_layers,
      relative_pos_max_distance)
  else:
    relative_pos_generator = etc_feature_utils.RelativePositionGenerator(
      relative_pos_max_distance)

  max_seq_len = input_config.max_seq_len

  def make_relative_transformer_side_inputs(
    long_breakpoints: tf.Tensor,
    name: Optional[str] = None) -> RelativeTransformerSideInputs:
    """Makes relative transformer side input tensors.

    Args:
      long_breakpoints: <int32>[batch_size, long_seq_len] Tensor of ending
        breakpoints separating different packed examples.
      name: A name for the operation (optional).
  
    Returns:
      A `RelativeTransformerSideInputs` with all relevant tensors set.
  
    """
    with tf.name_scope(name or 'make_relative_transformer_side_inputs'):
      long_breakpoints = tf.convert_to_tensor(long_breakpoints)
      long_example_ids = tf.cumsum(long_breakpoints, axis=-1, reverse=True)
      att_mask = etc_feature_utils.make_segmented_att_mask(long_example_ids)
  
      batch_size, long_seq_len = tf_utils.get_shape_list(long_example_ids)
      relative_att_ids = None
      if relative_pos_max_distance > 0:
        relative_att_ids = relative_pos_generator.make_relative_att_ids(
            seq_len=long_seq_len,
            batch_size=batch_size)
    
      return RelativeTransformerSideInputs(att_mask=att_mask,
                                           relative_att_ids=relative_att_ids)


  def add_side_input_features(
    features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Replaces raw input features with derived ETC side inputs.
  
    This function is meant to be called as part of a Dataset pipeline.
  
    Args:
      features: A dictionary of Tensor features, crucially including
        `long_breakpoints`, `num_image_wordpieces`, `num_text_wordpieces`.
  
    Returns:
      A new `features` dictionary with side inputs.
  
    """
  
    img_wp = features['num_image_wordpieces']
    txt_wp = features['num_text_wordpieces']
    seq_len = img_wp + txt_wp
  
    position = tf.range(max_seq_len, dtype=tf.int32)
    img_segment = tf.where(position < img_wp, 1, 0)
  
    txt_segment_mask = (position > img_wp) & (position < img_wp + txt_wp)
    txt_segment = tf.where(txt_segment_mask, 2, 0)
  
    segment_ids = img_segment + txt_segment
    features['segment_ids'] = segment_ids
    
    # seq_len-1 because we need the indices
    long_breakpoints = tf.one_hot(
        seq_len-1, depth=max_seq_len,
        on_value=1, off_value=0, dtype=tf.int32)
  
    long_breakpoints = tf.expand_dims(long_breakpoints, axis=0)
  
    side_inputs = make_relative_transformer_side_inputs(
        long_breakpoints=long_breakpoints)
  
    side_inputs.relative_att_ids = tf.squeeze(
        side_inputs.relative_att_ids, axis=0)
    side_inputs.att_mask = tf.squeeze(side_inputs.att_mask, axis=0)
  
    features.update(side_inputs.to_dict())
  
    return features
  return add_side_input_features


def get_masking_fn(tokenizer: tf_text.BertTokenizer,
                   input_config: configs.MmtDataConfig):
  """Creates a masking_fn (`make_mlm_and_mpp_features`)."""

  text_special_token_field_dict = json.loads(
      input_config.text_special_token_field_dict)
  max_seq_len = input_config.max_seq_len
  patch_size = input_config.patch_size
  image_size = input_config.image_size
  max_pixel_val = input_config.max_pixel_val
  channels = input_config.input_channels
  output_channel_bits = input_config.output_channel_bits

  mlm_fraction_to_mask = input_config.mlm_fraction_to_mask
  mpp_fraction_to_mask = input_config.mpp_fraction_to_mask
  mlm_max_selections_per_seq = min(input_config.mlm_max_selections_per_seq,
                                   max_seq_len)
  mpp_max_selections_per_seq = input_config.mpp_max_selections_per_seq  

  vocab = tokenizer._wordpiece_tokenizer._get_vocab_and_ids()[0]
  vocab = vocab.numpy().tolist()
  
  unselectable_tokens = [b'[CLS]', b'[SEP]', b'[PATCH]']
  unselectable_tokens += [t.encode()
                          for t in text_special_token_field_dict.values()]
  unselectable_ids = list(map(vocab.index, unselectable_tokens))
  mask_token_id = vocab.index(b'[MASK]')

  if input_config.use_patch_mask_token_id:
    patch_mask_token_id = vocab.index(b'[PATCH_MASK]')
  else:
    patch_mask_token_id = mask_token_id

  text_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mlm_max_selections_per_seq,
      selection_rate=mlm_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  text_dummy_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mlm_max_selections_per_seq,
      selection_rate=0,
      unselectable_ids=unselectable_ids)

  patch_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mpp_max_selections_per_seq,
      selection_rate=mpp_fraction_to_mask,
      unselectable_ids=unselectable_ids)

  patch_dummy_item_selector = tf_text.RandomItemSelector(
      max_selections_per_batch=mpp_max_selections_per_seq,
      selection_rate=0,
      unselectable_ids=unselectable_ids)

  text_mask_values_chooser = tf_text.MaskValuesChooser(
      len(vocab), mask_token_id, 0.8)

  patch_mask_values_chooser = tf_text.MaskValuesChooser(
      len(vocab), patch_mask_token_id, 0.8)

  num_patch_per_row = image_size // patch_size
  num_patches = num_patch_per_row ** 2
  # [CLS] + [PATCH] + Image + [SEP] + Text + [SEP].
  # -2: [CLS] and [PATCH].
  max_remaining_seq_len = max_seq_len - num_patches - 2

  def make_mpp_label_ids(mpp_embeddings, masked_seq_len):
    """Makes taget labels for masked patch prediction.
  
    Args:
      mpp_embeddings:
        <tf.Tensor>[sequence_length, patch_embedding_size].
  
    Returns:
      <tf.Tensor>[sequence_length].

    """

    bin_size = max_pixel_val // (2 ** output_channel_bits)

    masked_seq_len = tf_utils.get_shape_list(mpp_embeddings)[0]
  
    # Scale from 0-1 to 0-255
    mpp_embeddings = mpp_embeddings * (max_pixel_val - 1)
    mpp_embeddings = tf.reshape(
        mpp_embeddings, [masked_seq_len, patch_size**2, channels])
    avg_target = tf.reduce_mean(mpp_embeddings, axis=1)

    channel_bins = list(range(bin_size, max_pixel_val, bin_size))

    discretized_target = math_ops._bucketize(avg_target, channel_bins)
    discretized_target = tf.cast(discretized_target, tf.int32)
    discretized_target = tf.reshape(
        discretized_target, [masked_seq_len, channels])
    
    bin_mask = (2 ** output_channel_bits) ** tf.range(0, channels)
    bin_mask = tf.expand_dims(bin_mask, axis=0)
    label_ids = bin_mask * discretized_target
    label_ids = tf.reduce_sum(label_ids, axis=1)
    return label_ids

  def get_masked_weights(masked_token_ids: tf.Tensor,
                         masked_seq_len: int,
                         mask_token_id: int) -> tf.Tensor:
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
    num_real_masked_tokens = tf.reduce_sum(mask_position, axis=0)
    position = tf.range(masked_seq_len, dtype=tf.int32)
    return tf.where(position < num_real_masked_tokens, 1.0, 0.0)
    
  def make_mlm_and_mpp_features(
    features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Makes features for masked language model and masked patch prediction.

    Args:
      features: A dictionary of Tensor features, crucially including
          `patch_token_ids`, `text_token_ids`. The two tensors will be popped
          out since we don't need them during training.

    Returns:
      A new `features` dictionary with `word_ids`, `mpp_positions`,
      `mpp_label_ids`, `mpp_label_weights`, 
      `mlm_positions`, `mlm_label_ids`,
      `mlm_label_weights`.

    """
    # Patch: create features for masked patch prediction (mpp)
    # patch_token_ids: <tf.RaggedTensor>[batch, (patches), (1)].
    patch_token_ids = features['patch_token_ids']

    # Extend axis 0 as batch dim for tf_text.mask_language_model.
    # patch_token_ids: <tf.RaggedTensor>[1, (patches), (1)].
    patch_token_ids = tf.expand_dims(patch_token_ids, axis=0)
    
    (mpp_token_ids,
     mpp_positions, _) = tf_text.mask_language_model(
        patch_token_ids,
        patch_item_selector,
        patch_mask_values_chooser,
        axis=1)

    # Squeeze extended axis 0.
    mpp_token_ids = tf.squeeze(mpp_token_ids, axis=0)
    mpp_positions = tf.squeeze(mpp_positions, axis=0)
    mpp_positions = tf.cast(mpp_positions, tf.int32)

    # Offset -2 position indices: [CLS] and [PATCH] tokens.
    # We want to select from patch embeddings.
    shifted_patch_masked_positions = mpp_positions - 2
    shifted_patch_masked_positions = tf.cast(shifted_patch_masked_positions,
                                             tf.int32)
    unnorm_patch_embeddings = features.pop('unnormalized_patch_embeddings')
    patch_masked_seq_len = tf_utils.get_shape_list(
        shifted_patch_masked_positions)[0]
    patch_embedding_size = tf_utils.get_shape_list(
        unnorm_patch_embeddings)[1]

    mpp_embeddings = tf.gather(
      unnorm_patch_embeddings, shifted_patch_masked_positions)
    mpp_embeddings = tf.reshape(
        mpp_embeddings,
        [patch_masked_seq_len, patch_embedding_size])

    mpp_positions = tensor_utils.pad_to_max_seq_len(
        mpp_positions, mpp_max_selections_per_seq)

    # Create label for masked patch prediction.
    mpp_label_ids = make_mpp_label_ids(
        mpp_embeddings, patch_masked_seq_len)
    mpp_label_weights = get_masked_weights(
        mpp_token_ids, patch_masked_seq_len, patch_mask_token_id)

    mpp_label_ids = tensor_utils.pad_to_max_seq_len(
        mpp_label_ids, mpp_max_selections_per_seq)
    mpp_label_weights = tensor_utils.pad_to_max_seq_len(
        mpp_label_weights, mpp_max_selections_per_seq)

    # Zero out the masked patch embeddings.
    sliced_mpp_token_ids = tf.slice(
        mpp_token_ids,
        begin=[2],
        size=[num_patches])
    non_masked_token_position = tf.not_equal(sliced_mpp_token_ids,
                                             patch_mask_token_id)
    non_masked_token_position = tf.cast(non_masked_token_position, tf.float32)
    non_masked_token_position = tf.expand_dims(non_masked_token_position,
                                               axis=-1)
    features['patch_embeddings'] *= non_masked_token_position

    features['mpp_positions'] = mpp_positions
    features['mpp_label_weights'] = mpp_label_weights
    features['mpp_label_ids'] = mpp_label_ids
    features['patch_token_ids'] = mpp_token_ids

    # Text: create features for Masked Language Modeling (MLM).
    # text_token_ids: <tf.RaggedTensor>[(words), (wordpieces)].
    text_token_ids = features['text_token_ids']

    # text_token_ids: <tf.RaggedTensor>[1, (words), (wordpieces)].
    # Expands a dummy dim for batch.
    text_token_ids = tf.expand_dims(text_token_ids, axis=0)

    if not input_config.mlm_use_whole_word:
      # text_token_ids: <tf.RaggedTensor>[batch, (wordpieces)].
      text_token_ids = text_token_ids.merge_dims(-2, -1)

    # MLM
    (mlm_token_ids,
     mlm_positions,
     mlm_label_ids) = tf_text.mask_language_model(
        text_token_ids,
        text_item_selector,
        text_mask_values_chooser,
        axis=1)

    # Squeezes the dummpy batch dim.
    mlm_token_ids = tf.squeeze(mlm_token_ids, axis=0)
    mlm_positions = tf.squeeze(mlm_positions, axis=0)
    mlm_label_ids = tf.squeeze(mlm_label_ids, axis=0)
    mlm_positions = tf.cast(mlm_positions, tf.int32)

    # Offsets text positions because we append text tokens after patch tokens
    # ([CLS] [PATCH] P1 p2 ... P196).
    mlm_positions = mlm_positions + 2 + num_patches
    mlm_positions = tensor_utils.pad_to_max_seq_len(
        mlm_positions, mlm_max_selections_per_seq)
    
    mlm_label_ids = tensor_utils.pad_to_max_seq_len(
        mlm_label_ids, mlm_max_selections_per_seq)

    text_masked_seq_len = tf_utils.get_shape_list(mlm_positions)[0]
    mlm_label_weights = get_masked_weights(
        mlm_token_ids, text_masked_seq_len, mask_token_id)
    mlm_label_weights = tensor_utils.pad_to_max_seq_len(
        mlm_label_weights, mlm_max_selections_per_seq)

    features['mlm_positions'] = mlm_positions
    features['mlm_label_ids'] = mlm_label_ids
    features['mlm_label_weights'] = mlm_label_weights 
    features['text_token_ids'] = tensor_utils.pad_to_max_seq_len(
        mlm_token_ids, max_remaining_seq_len)
    return features
  
  return make_mlm_and_mpp_features


def get_matching_fn(input_config,
                    batch_size,
                    negative_positive_ratio=1,
                    min_shift=5):

  assert batch_size > (negative_positive_ratio + 1 + min_shift)
  assert negative_positive_ratio > 0
  image_key_field = input_config.image_key_field

  def make_matching_features(features):

    # Sorts by image index to put the same images together.
    _, in_batch_image_idx = tf.unique(features.pop(image_key_field))
    sort_order = tf.argsort(in_batch_image_idx)
    sort_order = tf.expand_dims(sort_order, axis=-1)
    for k, v in features.items():
      features[k] = tf.gather_nd(v, sort_order)

    total_num_copy = negative_positive_ratio + 1

    patch_embeddings = features['patch_embeddings']
    patch_token_ids = features['patch_token_ids']
    features['patch_token_ids'] = tf.tile(
        patch_token_ids, [total_num_copy, 1])
    multiply = [total_num_copy, 1, 1]
    features['patch_embeddings'] = tf.tile(patch_embeddings, multiply)
    num_image_wordpieces = features['num_image_wordpieces']
    features['num_image_wordpieces'] = tf.tile(
        num_image_wordpieces, [total_num_copy])

    text_token_ids = features['text_token_ids']
    num_text_wordpieces = features['num_text_wordpieces']

    # Creates permutation indices for texts.
    # 1. creates `negative_positive_ratio + 1` times copies.
    # 2. shifts 1 index for each new copy. 
    # For example, if negative_positive_ratio = 1, min_shift = 1 and the text
    # indices for 4 examples are [0, 1, 2, 3], then [0, 1, 2, 3] will be shifted 
    # to [2, 3, 0, 1].
    permutations = []
    permutation = tf.range(batch_size)
    permutations.append(permutation)
    for i in range(1, negative_positive_ratio+1):
      permutation = tf.range(batch_size)
      permutation = tf.roll(permutation, shift=min_shift+i, axis=0)
      permutations.append(permutation)
    permutations = tf.concat(permutations, axis=0)
    permutations = tf.expand_dims(permutations, axis=-1)

    # Permutates text because the whole sequence length is determined by text.
    features['text_token_ids'] = tf.gather_nd(text_token_ids, permutations)
    features['num_text_wordpieces'] = tf.gather_nd(
        num_text_wordpieces, permutations)

    pos_idxs = tf.expand_dims(tf.range(batch_size), axis=-1)
    updates = tf.ones((batch_size,))
    shape = tf.constant([batch_size*total_num_copy])
    itm_label_ids = tf.scatter_nd(pos_idxs, updates, shape)
    features['itm_label_ids'] = tf.cast(itm_label_ids, tf.int32)
    itm_label_weights_base = tf.ones_like(itm_label_ids, tf.float32)
    itm_label_pos_weights = itm_label_ids * (negative_positive_ratio - 1)
    features['itm_label_weights'] = itm_label_weights_base
    features['itm_pos_weights'] = itm_label_weights_base + itm_label_pos_weights

    for k in ['mlm_positions', 'mlm_label_ids', 'mlm_label_weights',
              'mpp_positions', 'mpp_label_ids', 'mpp_label_weights']:
      if k in features:
        features[k] = tf.gather_nd(features[k], permutations)
  
    return features
  return make_matching_features


def get_pop_fn(keep_keys):
  """Pops out unused features."""

  keep_keys = set(keep_keys)

  def pop_unused_features(features):
    for k in list(features.keys()):
      if k not in keep_keys:
        features.pop(k)
    return features
  return pop_unused_features


def get_word_ids_fn(input_config):
  """Merges patch and text token ids as word ids."""

  max_seq_len = input_config.max_seq_len

  def make_word_ids_features(features):
    patch_token_ids = features['patch_token_ids']
    text_token_ids = features['text_token_ids']
    word_ids = tf.concat([patch_token_ids, text_token_ids], axis=0)
    word_ids = tensor_utils.pad_to_max_seq_len(word_ids, max_seq_len)
    features['word_ids'] = word_ids
    return features

  return make_word_ids_features


def get_retrieval_label_fn(pos_weight):
  """Gets retrieval labels given image and ground-truth indices."""

  def get_label(features):
  
    label = tf.where(
        features['image_index'] == features['gt_image_index'], 1, 0)
    label = tf.cast(label, tf.int32)
    features['label_ids'] = label
  
    # Adds weights on positive examples.
    label_weights = tf.cast(label, dtype=tf.float32)
    label_weights = label_weights * (pos_weight - 1)
    label_weights = label_weights + 1
    features['label_weights'] = label_weights
  
    return features
  return get_label


def combine_image_text(x, y):
  ds = tf.data.Dataset.range(1)
  ds = ds.map(lambda i: {**x, **y})
  return ds
