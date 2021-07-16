import attr
import tensorflow as tf
import tensorflow_text as tf_text

from typing import List

from google_research.etcmodel.models.modeling import EtcConfig 


@attr.s
class PretrainInputConfig(object):
  """Config options for pretraining model input."""

  image_size = attr.ib(default=224)
  text_keys = attr.ib(factory=List)
  patch_size = attr.ib(default=16)
  patch_order = attr.ib(default='snake')
  mlm_use_whole_word = attr.ib()  # type: bool


def get_pretrain_example_decode_fn(tokenizer: tf_text.BertTokenizer,
                                   input_config: PretrainInputConfig,
                                   model_config: EtcConfig,
                                   is_training: bool):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  patch_size = input_config.patch_size
  num_patch_per_row = image_size // patch_size

  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature([], tf.string, default_value='')


  def patch_reorder(im, mode='snake'):
    if mode == 'snake':
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
      if tf.random.uniform(shape=[]) > 0.0:
        im = tf.image.flip_left_right(im)
    else:
      im = tf.image.resize(im, [image_size, image_size])
    
    # Create a sequence of patches (token)
    im = tf.expand_dims(im, axis=0)
    im = tf.image.extract_patches(im,
                                  sizes=[1, patch_size, patch_size, 1],
                                  strides=[1, patch_size, patch_size, 1],
                                  rates=[1, 1, 1, 1],
                                  padding = "VALID")
    im = tf.squeeze(im, axis=0)
    im = patch_reorder(im, mode=input_config.patch_order)
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
