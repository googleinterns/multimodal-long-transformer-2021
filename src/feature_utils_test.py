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

"""Tests for feature utils."""

from absl.testing import parameterized
import tensorflow as tf

import feature_utils


class TensorUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_relative_position_generator_init(self):
    relative_pos_gen = feature_utils.MmtRelativePositionGenerator(
        num_patch_per_row=2,
        num_core_layers=1,
        text_relative_pos_max_distance=3)

    self.assertEqual(2, relative_pos_gen._num_patch_per_row)
    self.assertEqual(1, relative_pos_gen._num_core_layers)
    self.assertEqual(3, relative_pos_gen._core_layer_diameter)
    self.assertEqual(19, relative_pos_gen._image_part_id)
    self.assertEqual(20, relative_pos_gen._text_part_id)

  def test_relative_position_generator_init_invalid_arguments(self):
    with self.assertRaises(ValueError):
      feature_utils.MmtRelativePositionGenerator(num_patch_per_row=0,
                                                 num_core_layers=1,
                                                 text_relative_pos_max_distance=2)
      feature_utils.MmtRelativePositionGenerator(num_patch_per_row=1,
                                                 num_core_layers=0,
                                                 text_relative_pos_max_distance=2)
      feature_utils.MmtRelativePositionGenerator(num_patch_per_row=1,
                                                 num_core_layers=1,
                                                 text_relative_pos_max_distance=-1)

  def test_make_relative_att_ids_smaller_case(self):
    relative_pos_gen = feature_utils.MmtRelativePositionGenerator(
        num_patch_per_row=2,
        num_core_layers=1,
        text_relative_pos_max_distance=3)

    """
    base_tensor = tf.Tensor(
        [[16  9  9  9 10]
         [15  5  6  7 11]
         [15  8  0  1 11]
         [15  2  3  4 11]
         [14 13 13 13 12]], shape=(5, 5), dtype=int32)
    """

    expected = [[
        [ 0,  1,  3,  4, 20, 20, 20],
        [ 8,  0,  2,  3, 20, 20, 20],
        [ 6,  7,  0,  1, 20, 20, 20],
        [ 5,  6,  8,  0, 20, 20, 20],
        [19, 19, 19, 19,  0,  1,  2],
        [19, 19, 19, 19,  4,  0,  1],
        [19, 19, 19, 19,  5,  4,  0],
    ]]

    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(7, 1))

  def test_make_relative_att_ids_larger_case(self):
    relative_pos_gen = feature_utils.MmtRelativePositionGenerator(
        num_patch_per_row=3,
        num_core_layers=2,
        text_relative_pos_max_distance=9)

    """
    base_tensor = tf.Tensor(
        [[32 32 25 25 25 25 25 26 26]
         [32 32 25 25 25 25 25 26 26]
         [31 31 13 14 15 16 17 27 27]
         [31 31 18 19 20 21 22 27 27]
         [31 31 23 24  0  1  2 27 27]
         [31 31  3  4  5  6  7 27 27]
         [31 31  8  9 10 11 12 27 27]
         [30 30 29 29 29 29 29 28 28]
         [30 30 29 29 29 29 29 28 28]], shape=(9, 9), dtype=int32)
    """

    expected = [[
        [ 0,  1,  2,  5,  6,  7, 10, 11, 12, 37, 37, 37],
        [24,  0,  1,  4,  5,  6,  9, 10, 11, 37, 37, 37],
        [23, 24,  0,  3,  4,  5,  8,  9, 10, 37, 37, 37],
        [20, 21, 22,  0,  1,  2,  5,  6,  7, 37, 37, 37],
        [19, 20, 21, 24,  0,  1,  4,  5,  6, 37, 37, 37],
        [18, 19, 20, 23, 24,  0,  3,  4,  5, 37, 37, 37],
        [15, 16, 17, 20, 21, 22,  0,  1,  2, 37, 37, 37],
        [14, 15, 16, 19, 20, 21, 24,  0,  1, 37, 37, 37],
        [13, 14, 15, 18, 19, 20, 23, 24,  0, 37, 37, 37],
        [36, 36, 36, 36, 36, 36, 36, 36, 36,  0,  1,  2],
        [36, 36, 36, 36, 36, 36, 36, 36, 36, 10,  0,  1],
        [36, 36, 36, 36, 36, 36, 36, 36, 36, 11, 10,  0],
    ]]

    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(12, 1))


if __name__ == '__main__':
  tf.test.main()
