import os
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
from qwen_vl_utils import process_vision_info, smart_resize
from diffusers import DiffusionPipeline
import torch
import torch
import argparse
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import PartialState
import json
import hashlib
import random
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/data/wangzhen/pretrain_model/Qwen-Image-Edit")
parser.add_argument("--num_inference_steps", type=int, default=28)
parser.add_argument("--image_dir", type=str, default="dataset/MSCOCO/images")
parser.add_argument("--save_dir", type=str, default="dataset/MSCOCO")
parser.add_argument("--dataset", type=str, default="dataset/MSCOCO/test_contrastive_strategy.json")
args = parser.parse_args()

def load_models(model_name):

    pipeline = QwenImageEditPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="balanced")
    pipeline.set_progress_bar_config(disable=None)
    return pipeline
    


def generate_image(pipe, prompt, image_dir, num_inference_steps):
    image = Image.open(image_dir).convert("RGB")

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
    }
    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]
    return output_image


image_save_dir = os.path.join(args.save_dir, 'edit_images')
f_w = open(os.path.join(args.save_dir, 'test_image_editing_%i.json'%args.split), 'w', encoding='utf-8')
os.makedirs(image_save_dir, exist_ok=True)
lines = open(args.dataset, 'r').readlines()

prompt_list = []
hash_list = []
image_list = []
target_file_list = []
filter_num = 0
for line in lines:
    data = json.loads(line)

    editing_instruction = data['edit_instruction'].strip()
    if editing_instruction.count(' ') > 100:
        filter_num += 1
        continue
    prompt_list.append(editing_instruction)
    image_dir_i = os.path.join(args.image_dir, data['ori_image'])
    image_list.append(image_dir_i)
    target_file_list.append(os.path.join(args.image_dir, data['image']))


# Initialize with default model
print("Loading model...")
pipe = load_models(args.model)
print("Model loaded successfully!")
print('=====================================================================')
print('Image Editing Number:', len(prompt_list))
print('Filter Number for too long prompt:', filter_num)
assert len(image_list) == len(prompt_list)

for prompt, image, target_file in tqdm(zip(prompt_list, image_list, target_file_list)):
    try:
        if os.path.exists(target_file):
            continue
        gen_image = generate_image(pipe, prompt, image, args.num_inference_steps)
        gen_image.save(target_file)
    except:
        continue


