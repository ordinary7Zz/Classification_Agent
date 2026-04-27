#!/bin/bash

# Set model path, image directory, and label JSON file
MODEL_DIR="/mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/Llama-3.2-11B-Vision-Instruct"
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images"
LABEL_JSON="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json"
OUTPUT_CSV="llama_DDTI_preds.csv"

# Set additional options
DTYPE="fp16"
THRESHOLD=0.5

# Run inference with the Python script
python llama_thyroid_inference.py \
  --model_dir "$MODEL_DIR" \
  --image_dir "$IMAGE_DIR" \
  --label_json "$LABEL_JSON" \
  --out_csv "$OUTPUT_CSV" \
  --dtype "$DTYPE" \
  --threshold "$THRESHOLD"