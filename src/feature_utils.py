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


"""Feature utils.

Includes 2D relative position.

"""

from typing import Union

import tensorflow as tf

from etcmodel import feature_utils as etc_feature_utils


class MmtRelativePositionGenerator(object):
  """Generates `relative_att_ids` for image and text."""

  def __init__(self,
               num_patch_per_row: int,
               num_core_layers: int,
               text_relative_pos_max_distance: int):

    if num_patch_per_row <= 0:
      raise ValueError('`num_patch_per_row` must be positive.')
    if num_core_layers <= 0:
      raise ValueError('`num_core_layers` must be positive.')
    if text_relative_pos_max_distance < 0:
      raise ValueError('`text_relative_pos_max_distance` must be positive.')

    self._num_patch_per_row = num_patch_per_row
    # The number of core layers (radius from the top layer to the center) of
    # fine-grained position ids. The minimum is 1 which will has 9 ids.
    self._num_core_layers = num_core_layers
    # core_layer_diameter will be 3 in the case shown above.
    self._core_layer_diameter = num_core_layers * 2 + 1

    # 1D relative position IDs for text.
    text_max_id = text_relative_pos_max_distance * 2 + 1

    # Gives the same IDs for all patches when using 1D text relative IDs.
    self._image_part_id = (self._num_patch_per_row ** 2 +
                           len(self.direction_config) + text_max_id)

    # Gives the same IDs for all text when using 2D patch relative IDs.
    self._text_part_id = self._image_part_id + 1

    self._base_tensor = self.create_base_tensor()

    self._text_relative_generator = etc_feature_utils.RelativePositionGenerator(
        text_relative_pos_max_distance)

  def create_base_tensor(self):
    """Creates the base tensor for all patches.

    We use a kernel sliding on the base tensor to get the 2d relative position 
    ids.

    """
    r = self._num_core_layers
    d = self._core_layer_diameter
    n = self._num_patch_per_row - self._num_core_layers

    num_center_ids = d ** 2
    center = tf.range(num_center_ids)
    center = tf.roll(center, shift=d*r+r, axis=0)
    center = tf.reshape(center, (d, d))
    center = tf.pad(center, paddings=[[n, n], [n, n]])

    base_tensor = center
    for idx, dn in enumerate(self.direction_config.values(), start=d*d):
      dn_tensor = tf.fill(dn['fill'], idx)
      dn_tensor = tf.pad(dn_tensor, paddings=dn['paddings'])
      base_tensor += dn_tensor

    return base_tensor

  def make_relative_att_ids(self,
                            seq_len: Union[int, tf.Tensor],
                            batch_size: int):

    """Makes relative attention ids.

    Includes 1D for text and 2D for image.


    For image 2D relative IDs, we use the base tensor as the auxiliary tensor.
    Let's use the base_tensor as a toy example shown below.

      base_tensor = tf.Tensor(
          [[16  9  9  9 10]
           [15  5  6  7 11]
           [15  8  0  1 11]
           [15  2  3  4 11]
           [14 13 13 13 12]], shape=(5, 5), dtype=int32)

    If the image has 9 (3x3) patches A-I.

      A B C
      D E F
      G H I

    We position each patch at 0 of the base tensor and crop the region of the 
    base tensor that corresponds to the rest of the patches' positions. 

    For example, the 2D relative position attention ids of the patch A will be:

       0  1 11
       3  4 11
      13 13 12

    The 2D relative position attention ids of the patch B will be:

       8  0  1
       2  3  4
      13 13 13

    The 2D relative position attention ids of the patch H will be:
       9  9  9
       5  6  7
       8  0  1

    """
    image_seq_len = self._num_patch_per_row ** 2
    text_seq_len = seq_len - image_seq_len

    image_relative_att_ids = []
    for x in range(self._num_patch_per_row):
      for y in range(self._num_patch_per_row):
        begin = [self._num_patch_per_row - x, self._num_patch_per_row - y]
        size = [self._num_patch_per_row, self._num_patch_per_row]
        ids = tf.slice(self._base_tensor, begin, size)
        ids = tf.reshape(ids, (-1,))
        image_relative_att_ids.append(ids)

    image_relative_att_ids = tf.stack(image_relative_att_ids)
    image_relative_att_ids = tf.pad(image_relative_att_ids,
                                    paddings=[[0, 0], [0, text_seq_len]],
                                    constant_values=self._text_part_id)
    image_relative_att_ids = tf.expand_dims(image_relative_att_ids, axis=0)

    text_relative_att_ids = self._text_relative_generator.make_relative_att_ids(
        text_seq_len,
        batch_size=batch_size)
    text_relative_att_ids = tf.pad(text_relative_att_ids,
                                   paddings=[[0, 0], [0, 0], [image_seq_len, 0]],
                                   constant_values=self._image_part_id)
    return tf.concat([image_relative_att_ids, text_relative_att_ids], axis=1)

  @property
  def direction_config(self):
    """Creates direction configurations for 8 directions.

    Toy example:

      base_tensor = tf.Tensor(
          [[16  9  9  9 10]
           [15  5  6  7 11]
           [15  8  0  1 11]
           [15  2  3  4 11]
           [14 13 13 13 12]], shape=(5, 5), dtype=int32)

      1. Fine-grained IDs.
        5 6 7   ^
        8 0 1   |  d = core_layer_diameter
        2 3 4   v

      2. Coarse-grained IDs.
        The other 8 directions in total are as follows.

        top: 9.
        top-right: 10.
        right: 11.
        bottom-right 12.
        bottom: 13.
        bottom-left: 14.
        left: 15.
        top-left: 16.

    """
    d = self._core_layer_diameter
    m = self._num_patch_per_row + self._num_core_layers + 1
    n = self._num_patch_per_row - self._num_core_layers

    direction_config = {
        'top': {
            'fill': [n, d],
            'paddings': [[0, m], [n, n]],
        },
        'top_right': {
            'fill': [n, n],
            'paddings': [[0, m], [m, 0]],
        },
        'right': {
            'fill': [d, n],
            'paddings': [[n, n], [m, 0]],
        },
        'right_bottom': {
            'fill': [n, n],
            'paddings': [[m, 0], [m, 0]],
        },
        'bottom': {
            'fill': [n, d],
            'paddings': [[m, 0], [n, n]],
        },
        'bottom_left': {
            'fill': [n, n],
            'paddings': [[m, 0], [0, m]],
        },
        'left': {
            'fill': [d, n],
            'paddings': [[n, n], [0, m]],
        },
        'top_left': {
            'fill': [n, n],
            'paddings': [[0, m], [0, m]],
        }
    }
    return direction_config
