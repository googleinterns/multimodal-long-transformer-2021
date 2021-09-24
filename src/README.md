# Multimodal Transformer Models

## Environmnet setup

1. Set up python environment.
```bash
pip install requirements.txt
```
2. Clone etcmodel from [here](https://github.com/google-research/google-research/tree/master/etcmodel).
   Put the directory `etcmodel` in `src`.
   
## Pretraining

1. Use `scripts/pretrain.sh` and an yaml file to pretrain your model.
2. Remember to set your `TPU_NAME` and `TPU_ZONE` in `scripts/pretrain.sh`.

```bash
bash scripts/pretrain.sh [config_file] [output_dir]
```

For example,
```bash
bash scripts/pretrain.sh ./exp_yamls/pretrain/wit/mlm_itm.yaml gs://my_buckets/my_exp_dir
```

## Finetuning

1. Use `scripts/finetune.sh` and an yaml file to finetune your model.
2. Remember to set your `TPU_NAME` and `TPU_ZONE` in `scripts/finetune.sh`.

```bash
bash scripts/finetune.sh [config_file] [output_dir]
```

For example,
```bash
bash scripts/finetune.sh ./exp_yamls/finetune/wit/itm.yaml gs://my_buckets/my_exp_dir
```

## Predicting

1. Use `scripts/predict.sh` to predict the output and scores on val or test sets.
2. Use the corresponding finetuning yaml file for `config_file`.
3. Remember to set your `TPU_NAME` and `TPU_ZONE` in `scripts/predict.sh`.

```bash
bash scripts/predict.sh [config_file] [input_meta_data] [output_dir] [predict_split] [ckpt_path]
```

For example,
```bash
bash scripts/predict.sh ./exp_yamls/finetune/wit/itm.yaml gs://my_buckets/my_inference_data/input_meta_data gs://my_buckets/my_exp_dir/results test gs://my_buckets/my_exp_dir/ckpt-999
```

