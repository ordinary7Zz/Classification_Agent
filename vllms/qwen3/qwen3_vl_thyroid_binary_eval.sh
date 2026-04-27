python qwen3_vl_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv qwen3_vl_DDTI_Classification_all_preds.csv \
  --dtype bf16

