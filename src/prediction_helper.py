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

"""The helper for prediction."""
import collections
import json
import os
import pprint

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from data import retrieval_dataloader
from tasks import classification


def get_recall_at_k_from_dataframe(df, topks=[1, 3, 5, 10]):
  """Gets recall@k from a dataframe.

  The dataframe should contains 4 columns: image_index, text_index,
  gt_image_index, and output.

  """
  score_matrix = df.pivot_table(values='output',
                                index='image_index',
                                columns='text_index').values
  # If inference examples do not share the same pool, there will be missing
  # values in the score_matrix. Since all scores are positive, we replace the
  # missing values with -1 to ignore them when sorting.
  score_matrix = np.nan_to_num(score_matrix, nan=-1)

  df['positive'] = (df['image_index'] == df['gt_image_index']).astype(int)
  gt_matrix = df.pivot_table(values='positive',
                             index='image_index',
                             columns='text_index').values

  # If inference examples do not share the same pool, there will be missing
  # values in the gt_matrix. In gt_matrix, 1 means the image-text pair is a
  # matched pair. Oppositely, 0 means the pair is not matched. Missing values
  # must be negatives. Thus, we replace the missing values with 0s to ignore them
  # when counting scores.
  gt_matrix = np.nan_to_num(gt_matrix, nan=0)

  def rank(x, axis=-1):
   return np.argsort(np.argsort(x, axis=axis), axis=axis)

  m, n = score_matrix.shape
  
  # Reverses ranking indices: -n and -m and * -1.
  i2t_rank = (rank(score_matrix, axis=1) - n) * -1
  t2i_rank = (rank(score_matrix, axis=0) - m) * -1
  
  recall_dict = collections.OrderedDict()
  # image-to-text retrieval.
  for k in topks:
    rank_at_gt = (i2t_rank * gt_matrix)
    match = ((rank_at_gt <= k) & (rank_at_gt > 0)).astype(int)
    match = np.sum(match, axis=1).astype(float)
    match = np.clip(match, 0, 1)

    num_valid_gt = np.clip(np.sum(gt_matrix, axis=1), 0, 1)
    recall = np.divide(np.sum(match), np.sum(num_valid_gt), out=np.zeros(1))
    recall_dict[f'i2t @ {k:>2}'] = f'{np.mean(recall):.4f}'
  
  # text-to-image retrieval.
  for k in topks:
    rank_at_gt = (t2i_rank * gt_matrix)
    match = (rank_at_gt <= k) & (rank_at_gt > 0)
    match = np.sum(match, axis=0).astype(float)
    match = np.clip(match, 0, 1)

    num_valid_gt = np.clip(np.sum(gt_matrix, axis=0), 0, 1)
    recall = np.divide(np.sum(match), np.sum(num_valid_gt), out=np.zeros(1))
    recall_dict[f't2i @ {k:>2}'] = f'{np.mean(recall):.4f}'

  return recall_dict


def _write_results(task, model, data_config, output_dir):
  """Makes predictions and writes to output file.

  There are two output files. 

  1. results.csv: prediction results that contains matching scores of image-text pairs.
  2. recall.json: recall numbers of this prediction.

  """

  results = classification.predict(task, data_config, model)

  result_path = os.path.join(output_dir, 'results.csv')
  with tf.io.gfile.GFile(result_path, 'w') as f:
    # Converts list of RawResult to a dataframe.
    results = list(map(lambda x: dict(x._asdict()), results))
    df = pd.DataFrame(results)
    df['output'] = df['output'].clip(upper=1.0, lower=0.0)
    df.to_csv(f, index=False, float_format='%.8f')

  recall_dict = get_recall_at_k_from_dataframe(df)
  result_path = os.path.join(output_dir, 'recall.json')
  with tf.io.gfile.GFile(result_path, 'w') as f:
    json.dump(recall_dict, f, indent=4)

  pp = pprint.PrettyPrinter()
  logging.info(f'Results: {pp.pformat(recall_dict)}')


def write_results(task, input_meta_data, flags_obj):

  params = task.task_config.train_data
  vocab_filename = params.vocab_filename
  relative_pos_max_distance = params.relative_pos_max_distance
  relative_att_num_core_layers = params.relative_att_num_core_layers
  text_special_token_field_dict = params.text_special_token_field_dict

  predict_split = flags_obj.predict_split
  predict_global_batch_size = flags_obj.predict_global_batch_size
  test_output_dir = flags_obj.test_output_dir

  def get_retrieval_data_config():

    input_path = input_meta_data.get(f'{predict_split}_input_path', None)
    num_examples = input_meta_data.get(f'{predict_split}_num_examples', None)
    seq_length = input_meta_data['max_seq_length']

    if input_path is None:
      # Enumerates all combination of examples from image and text records.
      image_input_path = input_meta_data[f'{predict_split}_image_input_path']
      text_input_path = input_meta_data[f'{predict_split}_text_input_path']
      num_image_examples = input_meta_data[f'{predict_split}_num_image_examples']
      num_text_examples = input_meta_data[f'{predict_split}_num_text_examples']
  
      logging.info(f'Predicting {num_image_examples} images from {image_input_path}')
      logging.info(f'Predicting {num_text_examples} texts from {text_input_path}')
      logging.info(f'Predicting {num_text_examples * num_image_examples} in total.')
  
      data_config = retrieval_dataloader.MmtRetrievalDataConfig(
          global_batch_size=predict_global_batch_size,
          vocab_filename=vocab_filename,
          text_special_token_field_dict=text_special_token_field_dict,
          is_training=False,
          image_input_path=image_input_path,
          text_input_path=text_input_path,
          num_image_examples=num_image_examples,
          num_text_examples=num_text_examples,
          max_seq_len=seq_length,
          drop_remainder=False,
          include_image_text_index=True,
          relative_pos_max_distance=relative_pos_max_distance,
          relative_att_num_core_layers=relative_att_num_core_layers)
    else:
      # Image-text records.
      logging.info(f'Predicting {num_examples} examples from {input_path}.')
      data_config = retrieval_dataloader.MmtRetrievalDataConfig(
          global_batch_size=predict_global_batch_size,
          vocab_filename=vocab_filename,
          text_special_token_field_dict=text_special_token_field_dict,
          is_training=False,
          input_path=input_path,
          num_examples=num_examples,
          max_seq_len=seq_length,
          drop_remainder=False,
          include_image_text_index=True,
          relative_pos_max_distance=relative_pos_max_distance,
          relative_att_num_core_layers=relative_att_num_core_layers)

    return data_config

  data_config = get_retrieval_data_config()

  tf.io.gfile.makedirs(test_output_dir)
  
  ckpt_file = flags_obj.init_checkpoint
  if not ckpt_file:
    raise ValueError('No checkpoint assigned for prediction mode.')

  model = task.build_model()
  logging.info(f'Restoring checkpoint from {ckpt_file}.')
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.read(ckpt_file).expect_partial()
  status.expect_partial().assert_existing_objects_matched()
  logging.info(f'Finished loading pretrained checkpoint from {ckpt_file}.')

  _write_results(task, model, data_config, test_output_dir)
