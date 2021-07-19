import attr
from typing import List

import tensorflow as tf
import tensorflow_text as tf_text

from google_research.etcmodel.models.modeling import EtcConfig 


@attr.s
class PretrainInputConfig(object):
  """Config options for pretraining model input."""

  image_size = attr.ib(default=224)
  text_keys = attr.ib(factory=List)
  patch_size = attr.ib(default=16)
  patch_order = attr.ib(default='raster_scan')
  mlm_use_whole_word = attr.ib()  # type: bool


def get_pretrain_example_decode_fn(tokenizer: tf_text.BertTokenizer,
                                   input_config: PretrainInputConfig,
                                   model_config: EtcConfig,
                                   is_training: bool):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  patch_size = input_config.patch_size
  num_patch_per_row = image_size // patch_size

  image_token_ids = tf.range(start=IMAGE_START_UNUSED_ID,
                             limit=IMAGE_START_UNUSED_ID+num_patch_per_row**2)

  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature([], tf.string, default_value='')


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
                                 padding="VALID")
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
    for k in input_config.text_keys:
      ids = tokenizer.tokenize(example[k]).merge_dims(-2, -1)
      example[k] = ids 
      example[f'len_{k}'] = tf.size(ids)

    # (roylu) TODO
    # Get MLM features
    
    return example

  return _decode_fn
