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

import collections
import json
import os

import tensorflow as tf

import utils

# `all` configurations.
INPUT_FILES = 'gs://tabletalk-wit/wit/split/wit_v1.ai.{}.en.recordio-*'
EVAL_DATA_DIR='gs://mmt/wit/inference_data/all'

# `old_infernce` configurations.
# In order to have faster iteration of evaluating models,
# We take fewer number of examples.
INPUT_FILES = 'gs://tabletalk-wit/wit/split/wit_v1.ai.{}.en.recordio-0000*'
EVAL_DATA_DIR='gs://mmt/wit/inference_data/old_inference'


def serialize_example(feature):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

text_feature_keys = set([
    'canonical_doc_id',
    'caption_attribution_description',
    'caption_reference_description',
    'caption_alt_text_description',
    'page_title',
    'context_page_description'])

image_feature_keys = set([
    'image_data', 
    'canonical_doc_id'])

name_to_features = {'canonical_doc_id': tf.io.FixedLenFeature([], tf.string)}

input_meta_data = {'max_seq_length': 512}

for split in ['val', 'test']:
  print(f'Start to process {split}.')
  record_filenames = tf.io.gfile.glob(INPUT_FILES.format(split))

  id_to_image_feature = collections.OrderedDict()
  id_to_text_features = collections.defaultdict(list)
  for record_filename in record_filenames:

    basename = os.path.basename(record_filename)
    raw_dataset = tf.data.TFRecordDataset([record_filename])
    for record in raw_dataset:
      example = tf.io.parse_single_example(record, name_to_features)
      canonical_doc_id = example['canonical_doc_id'].numpy().decode('utf-8')

      example = tf.train.Example()
      example.ParseFromString(record.numpy())
      features = dict(example.features.feature)

      image_features = {k: v for k, v in features.items() if k in image_feature_keys}
      text_features = {k: v for k, v in features.items() if k in text_feature_keys}

      image_features['source'] = utils._bytes_feature(basename.encode())

      text_features['source'] = utils._bytes_feature(basename.encode())

      if canonical_doc_id not in id_to_image_feature:
        id_to_image_feature[canonical_doc_id] = image_features

      if canonical_doc_id in id_to_text_features:
        if text_features in id_to_text_features[canonical_doc_id]:
          print(f'duplicate txt found! file: {basename}')
          print(text_features, id_to_text_features[canonical_doc_id])
          continue
      id_to_text_features[canonical_doc_id].append(text_features)

  image_serialized_examples = []
  img_id_to_img_idx = {}
  for img_idx, (img_id, img_feat) in enumerate(id_to_image_feature.items()):
    img_feat['image_index'] = utils._int64_feature(img_idx)
    image_serialized_examples.append(serialize_example(img_feat))
    img_id_to_img_idx[img_id] = img_idx

  text_serialized_examples = []
  txt_idx = 0
  img_id_to_txt_idxs = collections.defaultdict(list)
  for img_id, txt_feats in id_to_text_features.items():

    img_idx = img_id_to_img_idx[img_id]
    for txt_feat in txt_feats:
      txt_feat['text_index'] = utils._int64_feature(txt_idx)
      txt_feat['gt_image_index'] = utils._int64_feature(img_idx)
      text_serialized_examples.append(serialize_example(txt_feat))
      img_id_to_txt_idxs[img_id].append(txt_idx)
      txt_idx += 1

  img_to_txts = {}
  txt_to_img = {}
  img_idx_to_img_id = {}
  positive_pairs = {}
  for img_id, img_idx in img_id_to_img_idx.items():
    txt_idxs = img_id_to_txt_idxs[img_id]
    img_to_txts[img_idx] = txt_idxs
    img_idx_to_img_id[img_idx] = img_id

    for txt_idx in txt_idxs:
      txt_to_img[txt_idx] = img_idx
      positive_pairs[f'{img_idx}_{txt_idx}'] = 1

  print(f'Total number of unique image examples: {len(image_serialized_examples)}')
  print(f'Total number of unique text examples: {len(text_serialized_examples)}')

  input_meta_data.update({
    f'{split}_image_input_path':
      f'{EVAL_DATA_DIR}/wit_v1.ai.{split}.en.recordio.image-00001-of-00001',
    f'{split}_text_input_path': 
      f'{EVAL_DATA_DIR}/wit_v1.ai.{split}.en.recordio.text-00001-of-00001',
    f'{split}_num_image_examples': len(image_serialized_examples),
    f'{split}_num_text_examples': len(text_serialized_examples),
  })

  for domain, examples in [('image', image_serialized_examples),
                           ('text', text_serialized_examples)]:
    filename = os.path.join(
        EVAL_DATA_DIR, f'wit_v1.ai.{split}.en.recordio.{domain}-00001-of-00001')

    # tf.data.Dataset.from_generator needs to take a callable generator.
    def gen():
      for ex in examples:
        yield ex

    serialized_features_dataset = tf.data.Dataset.from_generator(
      gen, output_types=tf.string, output_shapes=())
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

filename = os.path.join(EVAL_DATA_DIR, 'input_meta_data')
with tf.io.gfile.GFile(filename, 'w') as f:
  json.dump(input_meta_data, f, indent=4)
