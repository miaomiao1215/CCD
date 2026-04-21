import torch
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
import hashlib


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
    out_list, result_list = [], []

    for output in outputs:
        try:
            generated_text = output.outputs[0].text
            generated_text = generated_text.split('</think>')[-1].replace('**', '')
            if 'Output:' in generated_text:
                generated_text = generated_text[generated_text.index('Output:'): ].replace('Output:', '').strip()
            out_list.append(generated_text)

            contrastive_strategies = []
            max_strategy_num = 10
            stop_flag = False
            for strategy_index in range(max_strategy_num):
                if 'Strategy-%s:'%(strategy_index+2) not in generated_text:
                    strategy_part = generated_text[generated_text.index('Strategy-%s:'%(strategy_index+1)): ]\
                        .replace('Strategy-%s:'%(strategy_index+1), '').strip()
                    stop_flag = True
                else:
                    strategy_part = generated_text[generated_text.index('Strategy-%s:'%(strategy_index+1)): generated_text.index('Strategy-%s:'%(strategy_index+2))]\
                        .replace('Strategy-%s:'%(strategy_index+1), '').strip()
                contrastive_detail = strategy_part[strategy_part.index('Contrastive Detail:'): strategy_part.index('Contrastive Aspect:')]\
                    .replace('Contrastive Detail:', '').strip()
                contrastive_aspect = strategy_part[strategy_part.index('Contrastive Aspect:'): strategy_part.index('Contrastive Strategy:')]\
                    .replace('Contrastive Aspect:', '').strip()
                contrastive_strategy = strategy_part[strategy_part.index('Contrastive Strategy:'): strategy_part.index('Image Editing Instruction:')]\
                    .replace('Contrastive Strategy:', '').strip()
                if 'event element/attribute' in contrastive_strategy.lower():
                    contrastive_strategy = 'Event Element'
                edit_instruction = strategy_part[strategy_part.index('Image Editing Instruction:'): ]\
                    .replace('Image Editing Instruction:', '').strip()
                contrastive_strategies.append({'contrastive_detail': contrastive_detail, 'contrastive_aspect': contrastive_aspect, \
                    'contrastive_strategy':contrastive_strategy, 'edit_instruction': edit_instruction})
                if stop_flag:
                    break

            result_list.append(contrastive_strategies)
        except:
            result_list.append(None)
    return out_list, result_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/path/to/Qwen3-VL-32B-Thinking-FP8', help="do predict")
    parser.add_argument("--dataset_dir", type=str, default='dataset/MSCOCO/test.json', help="do predict")
    parser.add_argument("--image_save_dir", type=str, default='dataset/MSCOCO/images', help="do predict")
    args = parser.parse_args()

    model_path = args.model_path
    min_pixels = 256 * 28 * 28
    max_pixels = 840 * 504
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=10000, n=1)

    prompts_list, image_id_list = [], []
    lines = open(args.dataset_dir, 'r').readlines()
    info_list = [json.loads(line) for line in lines]
    id_images_dict = defaultdict(list)
    for info in info_list:
        image = info['image']
        image_id = image.split('.')[0].split('_')[0]
        
        prompt_i = copy.deepcopy(PROMPT_GENERATE_CONTRAST_VLM)
        content_list = []
        
        image_dir = os.path.join(args.image_save_dir, image)
        bool_valid, image_size = validate_image(image_dir)
        if not bool_valid:
            continue
        resized_height, resized_width = smart_resize(
                height=image_size[1],
                width=image_size[0],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        content_list.append({"type": "image", "image": image_dir, "resized_height": resized_height, "resized_width": resized_width})
        content_list.append({"type": "text", "text": prompt_i})
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

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

        prompts_list.append(inputs)
        image_id_list.append(image_id)


    print('============================Number: %i============================='%len(prompts_list))
    llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True, enable_expert_parallel=True)

    batch_size = 16
    invalid_num = 0
    image_num, contrastive_image_num = 0, 0
    f_w = open(args.dataset_dir.replace('.json', '_contrast_strategy.json'), 'w', encoding='utf-8')
    for index in tqdm(range(0, len(prompts_list), batch_size)):
        out_list_batch, strategies_batch = generate_vllm(llm, prompts_list[index: index+batch_size], sampling_params)
        image_id_batch = image_id_list[index: index+batch_size]
        
        for image_id, strategies in zip(image_id_batch, strategies_batch):
            if strategies == None:
                invalid_num += 1
                continue
            image_num += 1
            for contrast_index, strategy in enumerate(strategies):
                strategy['ori_image'] = image_id + '.jpg'
                name = strategy['ori_image'].replace('.jpg', '_contrast_%s.jpg'%contrast_index)
                strategy['image'] = name
                contrastive_image_num += 1
                f_w.writelines(json.dumps(strategy, ensure_ascii=False) + '\n')
        print('invalid_num: ', invalid_num)

    f_w.close()
    print('invalid_num: ', invalid_num)
    print('image_num: ', image_num)
    print('contrastive_image_num: ', contrastive_image_num)
