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
from typing import Text, List

import tensorflow as tf
import tensorflow_text as tf_text

from google_research.etcmodel.models.modeling import EtcConfig 
from tensor_utils import full, ragged_full


_PATCH_START_UNUSED_INDEX = 99


@attr.s
class PretrainInputConfig(object):
  """Config options for pretraining model input."""

  # Images of "image_size * imate_size" pixels.
  image_size = attr.ib(default=224)

  # The names of text fields we want to use in the input
  text_keys = attr.ib(factory=List)

  # Patches of "patch_size * patch_size" pixels.
  patch_size = attr.ib(default=16)

  # The order of Patches to feed into transformer.
  patch_order = attr.ib(default='raster_scan')

  # Whole word masking for masked language modeling.
  mlm_use_whole_word = attr.ib(default=False)

  # The fraction of tokens to mask for masked language model loss.
  mlm_fraction_to_mask = attr.ib(default=0.15)

  # The fraction of tokens to mask for masked patch prediction loss.
  mpp_fraction_to_mask = attr.ib(default=0.50)


def get_pretrain_example_decode_fn(tokenizer: tf_text.BertTokenizer,
                                   input_config: PretrainInputConfig,
                                   model_config: EtcConfig,
                                   is_training: bool):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  patch_size = input_config.patch_size
  num_patch_per_row = image_size // patch_size
  vocab = tokenizer._wordpiece_tokenizer._get_vocab_and_ids()[0]
  vocab = vocab.numpy().tolist()

  special_token_to_ragged_tensor = {
      'cls': ragged_full((1, 1, 1), tf.int32, vocab.index(b'[CLS]')),
      'sep': ragged_full((1, 1, 1), tf.int32, vocab.index(b'[SEP]')),
  }

  for i, key in enumerate(['patch'] + input_config.text_keys):
    unused_token = f'[unused{i}]'.encode()
    unused_token_idx = vocab.index(unused_token)
    special_token_to_ragged_tensor[key] = ragged_full(
        (1, 1, 1), tf.int32, unused_token_idx)

  # We start from unused99 to make unused1 to unused98 flexible
  # The large index of unused toekns is 993
  # Make sure the number of patches is lower than 895
  patch_start_token = f'[unused{_PATCH_START_UNUSED_INDEX}]'.encode()
  patch_start_idx = vocab.index(patch_start_token)

  patch_ids_tensor = tf.expand_dims(
      tf.range(patch_start_idx, num_patch_per_row**2+patch_start_idx), axis=1)
  patch_ids_tensor = tf.reshape(patch_ids_tensor, (1, num_patch_per_row**2, 1))
  patch_ids_tensor = tf.RaggedTensor.from_tensor(patch_ids_tensor)

  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature(
        [], tf.string, default_value='')

  def convert_image_to_patches(im):
    """Convert an image to patches (token embeddings).

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
    """Reorder the patch order of a iamge.

    Args:
      im: <float32>[num_patch_per_row, num_patch_per_row, 3*(patch_size**2)].
      mode: Mode of reordering.

    Returns:
      <float32>[num_patch_per_row**2, 3*(patch_size**2)].
    """
    if mode == 'raster_scan':
      return tf.reshape(im, [num_patch_per_row**2, (patch_size**2)*3])
    else:
      raise ValueError(f'Reordering mode ({mode}) is not available.')

  def _decode_fn(record):
    example = tf.io.parse_single_example(record, name_to_features)

    # Image
    # We follow the implementation of ViT
    im = tf.io.decode_image(example['image_data'], dtype=tf.float32)
    if is_training:
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.05, 1.0),
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
    example['image_data'] = im

    # Text
    text_len = 0
    for k in input_config.text_keys:
      example[k] = tokenizer.tokenize(example[k])

    # Concatenate all input token ids together by the following ordering:
    # [CLS] [PATCH] patch1 patch2 ... [ATTRIBUTION] w_ATT1 w_ATT2 ...
    # [REFERENCE] w_REF1 w_REF2 ... [ALT_TEXT] w_ALT1 w_ALT2 ... [SEP]
    image_input_ids = [special_token_to_ragged_tensor['cls'],
                       special_token_to_ragged_tensor['patch'],
                       patch_ids_tensor]
    image_input_ids = tf.squeeze(tf.concat(image_input_ids, axis=1), axis=0)
    image_input_ids = tf.RaggedTensor.from_tensor(image_input_ids)
    example['image_input_ids'] = image_input_ids

    text_input_ids = []
    for k in input_config.text_keys:
      text_input_ids.append(special_token_to_ragged_tensor[k])
      text_input_ids.append(example[k])
    text_input_ids.append(special_token_to_ragged_tensor['sep'])
    text_input_ids = tf.squeeze(tf.concat(text_input_ids, axis=1), axis=0)
    example['text_input_ids'] = text_input_ids

    return example

  return _decode_fn
