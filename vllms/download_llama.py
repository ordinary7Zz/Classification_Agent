import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
model_id = "LLM-Research/Llama-3.2-11B-Vision-Instruct"
model_dir = snapshot_download(model_id, ignore_file_pattern=['*.pth'])

print(model_dir)
