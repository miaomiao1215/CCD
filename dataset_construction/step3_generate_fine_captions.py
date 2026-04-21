import torch
from modelscope import AutoModel
import os
from tqdm import tqdm
import json
from collections import defaultdict
from PIL import Image
from glob import glob
import time
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import copy
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
from prompts import *
from vllm.assets.image import ImageAsset
import timeout_decorator
import re
import os
import math

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  
        return True, img.size
    except (IOError, SyntaxError) as e:
        return False, None


def generate_vllm(llm, prompts, sampling_params):
    print('===========Prompt Number: %i============='%len(prompts))
    outputs = llm.generate(prompts, sampling_params)
    zh_caption_all_list, caption_all_list, result_list = [], [], []

    for output in outputs:
        generated_text = output.outputs[0].text.split('</think>')[-1]
        try:
            captions = generated_text[generated_text.rindex('English Image Captions:'): generated_text.rindex('Chinese Image Captions:')].replace('English Image Captions:', '').strip()
            captions_list = captions.split('\n')
            captions_list = [re.sub(r'\d+\.', '', caption).strip() for caption in captions_list]
            
            zh_captions = generated_text[generated_text.rindex('Chinese Image Captions:'): ].replace('Chinese Image Captions:', '').strip()
            zh_captions_list = zh_captions.split('\n')
            zh_captions_list = [re.sub(r'\d+\.', '', caption).strip() for caption in zh_captions_list]
        except:
            captions_list = []
            zh_captions_list = []

        caption_all_list.append(captions_list)
        result_list.append(generated_text)
        zh_caption_all_list.append(zh_captions_list)
    return result_list, caption_all_list, zh_caption_all_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_save_dir", type=str, default='dataset/MSCOCO/images', help="do predict")
    parser.add_argument("--model_path", type=str, default='/path/to/Qwen3-VL-32B-Thinking-FP8', help="do predict")
    parser.add_argument("--dataset_dir", type=str, default='dataset/MSCOCO/test_constrastive_strategy.json', help="do predict")
    args = parser.parse_args()

    min_pixels = 256 * 28 * 28
    max_pixels = 840 * 504

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=40000, n=1)
    prompt_template = PROMPT_FINE_CAPTION
    image_list_all, prompts_list, ori_captions_list = [], [], []
    lines = open(args.dataset_dir, 'r').readlines()
    image_contrast_dict = defaultdict(list)
    image_info_dict = {}
    
    for index, line in enumerate(lines):
        info = json.loads(line)
        if '_' in info['image']:
            ori_image = info['image'].split('_')[0] + '.jpg'
            image_contrast_dict[ori_image].append(info['image'])
        else:
            image_info_dict[info['image']] = info

    total_image_num = 0
    processed_image_info_list = []
    for image_id in tqdm(image_info_dict.keys()):
        if image_id not in image_contrast_dict.keys():
            processed_image_info_list.append(image_info_dict[image_id])
            continue

        image_contents_list = []
        for contrast_image in ([image_id] + image_contrast_dict[image_id]):
            image_dir = os.path.join(args.image_save_dir, contrast_image)
            bool_valid, image_size = validate_image(image_dir)
            if not bool_valid:
                print(image_dir)
                continue
            resized_height, resized_width = smart_resize(
                    height=image_size[1],
                    width=image_size[0],
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            image_contents_list.append({"type": "image", "image": image_dir, "resized_height": resized_height, "resized_width": resized_width})

        total_image_num += len(image_contents_list)
        for index, caption in enumerate(image_info_dict[image_id]['captions']):
            prompt_i = copy.deepcopy(prompt_template).format(caption=caption, image_num=len(image_contents_list))
            contents_list = copy.deepcopy(image_contents_list)
            contents_list.append({"type": "text", "text": prompt_i})
            messages = [{"role": "user", "content": contents_list}]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = {
                "prompt": text,
                "multi_modal_data": {
                    "image": image_inputs
                },
            }
            image_list_all.append([image_id] + image_contrast_dict[image_id])
            prompts_list.append(inputs)
            ori_captions_list.append(caption)
    

    print('============================Number: %i============================='%len(prompts_list))
    print('============================total_image_num: %i============================='%total_image_num)
    llm = LLM(model=args.model_path, tensor_parallel_size=4, trust_remote_code=True, enable_expert_parallel=True)

    batch_size = 32
    faile_num = 0
    f_w = open(args.dataset_dir.replace('.json', '_fine_captions.json'), 'w')
    for processed_image_info in processed_image_info_list:
        f_w.writelines(json.dumps(processed_image_info, ensure_ascii=False) + '\n')

    image_captions_dict = defaultdict(list)
    save_image_num = 0
    for index in tqdm(range(0, len(prompts_list), batch_size)):
        try:
            output_batch, captions_batch, zh_captions_batch = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
        except:
            continue
        image_list_batch = image_list_all[index: index+batch_size]
        ori_captions_batch = ori_captions_list[index: index+batch_size]
        
        caption_index = 0
        for output, captions, zh_captions, image_list, ori_caption in zip(output_batch, captions_batch, zh_captions_batch, image_list_batch, ori_captions_batch):
            caption_index += 1
            if len(captions) == len(image_list):
                for caption, image in zip(captions, image_list):
                    image_captions_dict[image].append(caption)

    for image, captions in image_captions_dict.items():
        info = {'image': image, 'captions': captions}
        f_w.writelines(json.dumps(info, ensure_ascii=False) + '\n')
    f_w.close()

