# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs prediction."""

import json
import pprint

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

# Imports registered experiment configs.
from official.core import exp_factory
from official.core import task_factory
from official.modeling.hyperparams import params_dict

import distribute_utils
import prediction_helper
import registry_imports


# Device configs.
flags.DEFINE_string(
    'distribution_strategy',
    default='tpu',
    help='The Distribution Strategy to use for prediction.')

flags.DEFINE_string(
    'tpu',
    default='',
    help='The Cloud TPU to use for training.')

flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='zone of TPUs.')

flags.DEFINE_integer(
    'predict_global_batch_size',
    default=2048,
    help='batch size while prediction.')

flags.DEFINE_string(
    'test_output_dir',
    default=None,
    help='The dir to the test output data.')

flags.DEFINE_string(
    'input_meta_data_path',
    default=None,
    help='Path to file that contains metadata about input file.')

flags.DEFINE_string(
    'init_checkpoint',
    default=None,
    help='Initial checkpoint from a pre-trained BERT model.')

flags.DEFINE_multi_string(
    'config_file',
    default=None,
    help='File to specify the `ExperimentConfig` directly.')

flags.DEFINE_string(
    'predict_split',
    default='val',
    help='split that is used for prediction.')

FLAGS = flags.FLAGS

EXPERIMENT_TYPE = 'mmt/classification'


def _override_exp_config_by_file(exp_config, exp_config_files):
  """Overrides an `ExperimentConfig` object by files."""
  for exp_config_file in exp_config_files:
    if not tf.io.gfile.exists(exp_config_file):
      raise ValueError('%s does not exist.' % exp_config_file)
    params_dict.override_params_dict(
        exp_config, exp_config_file, is_strict=True)

  return exp_config


def _get_exp_config(input_meta_data, exp_config_files):
  """Gets an `ExperimentConfig` object."""
  exp_config = exp_factory.get_exp_config(EXPERIMENT_TYPE)

  logging.info('Loading `ExperimentConfig` from file.')
  exp_config = _override_exp_config_by_file(exp_config, exp_config_files)

  exp_config.validate()
  exp_config.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s',
               pp.pformat(exp_config.as_dict()))

  return exp_config


def _check_path_exists(flag_path, flag_name):
  if not tf.io.gfile.exists(flag_path):
    raise ValueError('Flag `%s` at %s does not exist.' %
                     (flag_name, flag_path))


def _validate_path(flag_path, flag_name):
  if not flag_path:
    raise ValueError(f'Flag `{flag_name}` must be provided')
  _check_path_exists(flag_path, flag_name)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  _validate_path(FLAGS.input_meta_data_path, 'input_meta_data_path')

  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      tpu_address=FLAGS.tpu,
      zone=FLAGS.tpu_zone)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  with distribution_strategy.scope():
    logging.info('Starting predict...')
    exp_config_files = FLAGS.config_file
    exp_config = _get_exp_config(input_meta_data=input_meta_data,
                                 exp_config_files=exp_config_files)
    task = task_factory.get_task(exp_config.task)
    prediction_helper.write_results(task, input_meta_data, FLAGS)


if __name__ == '__main__':
  app.run(main)
