import attr
import tensorflow as tf

from typing import List

from google_research.etcmodel.models.tokenization import FullTokenizer


@attr.s
class PretrainInputConfig(object):
  
  """Config options for pretraining model input."""

  image_size = attr.ib(default=224)
  text_keys = attr.ib(factory=List)


def get_pretrain_example_decode_fn(tokenizer: FullTokenizer,
                                   input_config: PretrainInputConfig,
                                   is_training: bool):
  """Returns a decode function to parse a single example into Tensors."""

  image_size = input_config.image_size
  name_to_features = {'image_data': tf.io.FixedLenFeature([], tf.string)}
  for k in input_config.text_keys:
    name_to_features[k] = tf.io.FixedLenFeature([], tf.string, default_value='')

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
    example['image_data'] = im

    # Text
    for k in input_config.text_keys:
      txt = example[k].numpy().decode('utf-8')
      tokens = tokenizer.tokenize(txt)
      ids = tokenizer.convert_tokens_to_ids(tokens)
      example[k] = ids

    return example

  return _decode_fn
