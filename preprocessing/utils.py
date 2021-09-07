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

import tensorflow as tf


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
  # Returns a bytes_list from a string / byte.
  if isinstance(value, type(tf.constant(0))):
      # BytesList won't unpack a string from an EagerTensor.
      value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  # Returns a float_list from a float / double.
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  # Returns an int64_list from a bool / enum / int / uint.
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    
def image_example(image_string, string_dict, int_dict=None):
  image_shape = tf.io.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image_data': _bytes_feature(image_string),
  }
  for k, v in string_dict.items():
    feature[k] = _bytes_feature(v)

  if int_dict is not None:
    for k, v in int_dict.items():
      feature[k] = _int64_feature(v)
  return tf.train.Example(features=tf.train.Features(feature=feature))


def get_txt_info(txt_info_filename, description_key='description'):
  """Gets metadata of each image_id (image file).
  
  Returns:
  txt_info: A dictionary. key is the image_id and value is a dictionary of the
  corresponding metadata.
  
  """
  txt_info = dict()
  with tf.io.gfile.GFile(txt_info_filename, 'r') as f:
    for i, line in enumerate(f, start=1):
      line = line.split('\x01')
  
      image_main_id = line[0]
      image_id = line[1]
      category = line[2]
      sub_category = line[4]
      description = line[6] 
  
      txt_info[image_id] = {
        'image_main_id': image_main_id.encode(),
        'image_id': image_id.encode(),
        'category': category.encode(),
        'sub_category': sub_category.encode(),
        description_key: description.encode(),
      }
  
      if i % 10000 == 0:
        print(f'Read txt info: {i}')
  return txt_info
