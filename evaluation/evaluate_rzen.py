import os
import json
from metrics import cal_recall
import torch
from modelscope import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
import json
from collections import defaultdict
from PIL import Image
from glob import glob
import time
import torch.nn.functional as F
import numpy as np
import argparse
from rzen_embed_inference import RzenEmbed

def compute_CDA(similarity_mat, triple_list, mode='i2t'):
    compare_result_list = []
    for triple in triple_list:
        index, true_index, false_index = triple
        if mode == 'i2t':
            similarity_i = similarity_mat[index, :]
            if similarity_i[true_index] > similarity_i[false_index]:
                compare_result_list.append(True)
            else:
                compare_result_list.append(False)
        else:
            similarity_i = similarity_mat[:, index]
            if similarity_i[true_index] > similarity_i[false_index]:
                compare_result_list.append(True)
            else:
                compare_result_list.append(False)            
    return 100 * compare_result_list.count(False) / len(compare_result_list)


def i2t(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    rank_all = np.argsort(sims, axis=1)[:, ::-1]
    for index in range(npts):
        inds = rank_all[index]

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), ranks, rank_all
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sims, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) or (5N, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    
    # --> (5N(caption), N(image))
    sims = sims.T
    rank_all = np.argsort(sims, axis=1)[:, ::-1]
    for index in range(npts):
        for i in range(5):
            inds = rank_all[5 * index + i]
            rank_all[index] = inds

            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), ranks, rank_all
    else:
        return (r1, r5, r10, medr, meanr)




def extract_embed_bge_vl_t2i(model, data_list, mode='image'):
    embed_all = None
    query_instruction = "Find me an everyday image that matches the given caption: "
    candidate_instruction = "Represent the given image."
    with torch.no_grad():
        if mode == 'text':
            batch_size = 10
            for start_index in tqdm(range(0, len(data_list), batch_size)):
                embed_batch = model.get_fused_embeddings(instruction=query_instruction, texts=data_list[start_index: start_index+batch_size])
                if embed_all == None:
                    embed_all = embed_batch
                else:
                    embed_all = torch.cat([embed_all, embed_batch], dim=0)
        elif mode == 'image':
            batch_size = 10
            for start_index in tqdm(range(0, len(data_list), batch_size)):
                images = [Image.open(_image).resize((512,512)).convert("RGB") for _image in data_list[start_index: start_index+batch_size]]
                embed_batch = model.get_fused_embeddings(instruction=candidate_instruction, images=images)
                if embed_all == None:
                    embed_all = embed_batch
                else:
                    embed_all = torch.cat([embed_all, embed_batch], dim=0)
    return embed_all



def extract_embed_bge_vl_i2t(model, data_list, mode='image'):
    embed_all = None
    query_instruction = "Find an image caption describing the given everyday image."
    with torch.no_grad():
        if mode == 'text':
            batch_size = 10
            for start_index in tqdm(range(0, len(data_list), batch_size)):
                embed_batch = model.get_fused_embeddings(texts=data_list[start_index: start_index+batch_size])
                if embed_all == None:
                    embed_all = embed_batch
                else:
                    embed_all = torch.cat([embed_all, embed_batch], dim=0)
        elif mode == 'image':
            batch_size = 10
            for start_index in tqdm(range(0, len(data_list), batch_size)):
                images = [Image.open(_image).resize((512,512)).convert("RGB") for _image in data_list[start_index: start_index+batch_size]]
                embed_batch = model.get_fused_embeddings(instruction=query_instruction, images=images)
                if embed_all == None:
                    embed_all = embed_batch
                else:
                    embed_all = torch.cat([embed_all, embed_batch], dim=0)
    return embed_all


parser = argparse.ArgumentParser()
parser.add_argument("--bench", type=str, default='flickr30k', help="do predict")
parser.add_argument("--dataset", type=str, default='/path/to/dataset', help="do predict")
parser.add_argument("--image_dir", type=str, default='/path/to/image_dataset', help="do predict")
parser.add_argument("--model_path", type=str, default="/path/to/RzenEmbed-v2-7B", help="do predict")
args = parser.parse_args()

t2i_prompt = 'Identify the image showcasing the described everyday scene.' if args.bench == 'MSCOCO' else 'Find an image that matches the given caption.'
i2t_prompt = 'Find an image caption describing the following image.' if args.bench == 'MSCOCO' else 'Find an image caption describing the following image.'

bench = args.bench
lines = open(args.dataset, 'r').readlines()
data_list = [json.loads(line) for line in lines]
caption_list = []
image_list = []
ori_image_index_list = []
for index, data in enumerate(data_list):
    image = os.path.join(args.image_dir, data['image'])
    if not os.path.exists(image):
        print(image)
        continue
    captions = data['captions']
    if len(captions) != 5:
        print('caption number error: ', image)
        continue
    if '_' not in data['image']:
        ori_image_index_list.append(index)
    image_list.append(image)
    caption_list.extend(captions)

print('Number of images: ', len(image_list))
print('Number of captions: ', len(caption_list))
print('original_image_number:', len(ori_image_index_list))

model_path = args.model_path
model = RzenEmbed(model_path)

print('========================Start evaluating image to text retrieval========================')
start_time = time.time()
print('========================Extract embeds for images========================')
image_embeds = extract_embed_bge_vl_i2t(model, image_list, mode='image')
print('using: ', time.time()-start_time)
print('========================Extract embeds for texts========================')
start_time = time.time()
text_embeds = extract_embed_bge_vl_i2t(model, caption_list, mode='text')
print('using: ', time.time()-start_time)


print("Calculating similarity...")
i2t_similarity = image_embeds @ text_embeds.T  # [N_image, N_text]
# os.makedirs('%s_fine'%bench, exist_ok=True)
# torch.save({'similarity': similarity.cpu().float()}, '%s_fine/%s_i2t.pth'%(bench, os.path.basename(model_path)))
i2t_similarity = i2t_similarity.cpu().float().numpy()

# query: ori dataset, candidates: ori dataset
similarity_select = i2t_similarity[ori_image_index_list]
select_columns_list = []
for index in ori_image_index_list:
    select_columns_list.extend([index*5+i for i in range(5)])
similarity_select = similarity_select[:, select_columns_list]

ori_nocontrast_result_i2t, ori_nocontrast_ranks_i2t, ori_nocontrast_rank_all_i2t = i2t(similarity_select, return_ranks=True)
ori_nocontrast_i2t_r1 = 100.0 * len(np.where(ori_nocontrast_ranks_i2t < 1)[0]) / len(ori_nocontrast_ranks_i2t)
ori_nocontrast_i2t_r5 = 100.0 * len(np.where(ori_nocontrast_ranks_i2t < 5)[0]) / len(ori_nocontrast_ranks_i2t)
ori_nocontrast_i2t_r10 = 100.0 * len(np.where(ori_nocontrast_ranks_i2t < 10)[0]) / len(ori_nocontrast_ranks_i2t)
print("(original dataset without contrast) Image to text: r1 %.1f; r5 %.1f; r10 %.1f" %(ori_nocontrast_i2t_r1, ori_nocontrast_i2t_r5, ori_nocontrast_i2t_r10))
res_dict = {'ori_nocontrast_i2t_r1': ori_nocontrast_i2t_r1, 'ori_nocontrast_i2t_r5': ori_nocontrast_i2t_r5, 'ori_nocontrast_i2t_r10': ori_nocontrast_i2t_r10}

# query: ori + contrast dataset, candidates: ori + contrast dataset
result_i2t, ranks_i2t, rank_all_i2t = i2t(i2t_similarity, return_ranks=True)
i2t_r1 = 100.0 * len(np.where(ranks_i2t < 1)[0]) / len(ranks_i2t)
i2t_r5 = 100.0 * len(np.where(ranks_i2t < 5)[0]) / len(ranks_i2t)
i2t_r10 = 100.0 * len(np.where(ranks_i2t < 10)[0]) / len(ranks_i2t)
print("All dataset Image to text: r1 %.1f; r5 %.1f; r10 %.1f" %(i2t_r1, i2t_r5, i2t_r10))
res_dict['i2t_r1'] = i2t_r1
res_dict['i2t_r5'] = i2t_r5
res_dict['i2t_r10'] = i2t_r10

# query: ori dataset, candidates: ori + contrast dataset; used in paper
ranks_i2t_ori = np.zeros(len(image_list))
ori_index = 0
for index, image in enumerate(image_list):
    if '_' in os.path.basename(image):
        continue
    ranks_i2t_ori[ori_index] = ranks_i2t[index]
    ori_index += 1
ranks_i2t_ori = ranks_i2t_ori[0: ori_index]
print('original image num: ', ori_index)
ori_i2t_r1 = 100.0 * len(np.where(ranks_i2t_ori < 1)[0]) / len(ranks_i2t_ori)
ori_i2t_r5 = 100.0 * len(np.where(ranks_i2t_ori < 5)[0]) / len(ranks_i2t_ori)
ori_i2t_r10 = 100.0 * len(np.where(ranks_i2t_ori < 10)[0]) / len(ranks_i2t_ori)
print("Original Image to text: r1 %.1f; r5 %.1f; r10 %.1f" %(ori_i2t_r1, ori_i2t_r5, ori_i2t_r10))
res_dict['ori_i2t_r1'] = ori_i2t_r1
res_dict['ori_i2t_r5'] = ori_i2t_r5
res_dict['ori_i2t_r10'] = ori_i2t_r10




print('========================Start evaluating text to image retrieval========================')
start_time = time.time()
print('========================Extract embeds for images========================')
image_embeds = extract_embed_bge_vl_t2i(model, image_list, mode='image')
print('using: ', time.time()-start_time)
print('========================Extract embeds for texts========================')
start_time = time.time()
text_embeds = extract_embed_bge_vl_t2i(model, caption_list, mode='text')
print('using: ', time.time()-start_time)

print("Calculating similarity...")
t2i_similarity = image_embeds @ text_embeds.T  # [N_image, N_text]
# os.makedirs('%s_fine'%bench, exist_ok=True)
# torch.save({'similarity': similarity.cpu().float()}, '%s_fine/%s_t2i.pth'%(bench, os.path.basename(model_path)))
t2i_similarity = t2i_similarity.cpu().float().numpy()

# query: ori dataset, candidates: ori dataset
similarity_select = t2i_similarity[ori_image_index_list]
select_columns_list = []
for index in ori_image_index_list:
    select_columns_list.extend([index*5+i for i in range(5)])
similarity_select = similarity_select[:, select_columns_list]


ori_nocontrast_result_t2i, ori_nocontrast_ranks_t2i, ori_nocontrast_rank_all_t2i = t2i(similarity_select, return_ranks=True)
ori_nocontrast_t2i_r1 = 100.0 * len(np.where(ori_nocontrast_ranks_t2i < 1)[0]) / len(ori_nocontrast_ranks_t2i)
ori_nocontrast_t2i_r5 = 100.0 * len(np.where(ori_nocontrast_ranks_t2i < 5)[0]) / len(ori_nocontrast_ranks_t2i)
ori_nocontrast_t2i_r10 = 100.0 * len(np.where(ori_nocontrast_ranks_t2i < 10)[0]) / len(ori_nocontrast_ranks_t2i)
print("(original dataset without contrast) Text to image: r1 %.1f; r5 %.1f; r10 %.1f"%(ori_nocontrast_t2i_r1, ori_nocontrast_t2i_r5, ori_nocontrast_t2i_r10))
res_dict['ori_nocontrast_t2i_r1'] = ori_nocontrast_t2i_r1
res_dict['ori_nocontrast_t2i_r5'] = ori_nocontrast_t2i_r5
res_dict['ori_nocontrast_t2i_r10'] = ori_nocontrast_t2i_r10

# query: ori + contrast dataset, candidates: ori + contrast dataset
result_t2i, ranks_t2i, rank_all_t2i = t2i(t2i_similarity, return_ranks=True)
t2i_r1 = 100.0 * len(np.where(ranks_t2i < 1)[0]) / len(ranks_t2i)
t2i_r5 = 100.0 * len(np.where(ranks_t2i < 5)[0]) / len(ranks_t2i)
t2i_r10 = 100.0 * len(np.where(ranks_t2i < 10)[0]) / len(ranks_t2i)
print("Text to image: r1 %.1f; r5 %.1f; r10 %.1f"%(t2i_r1, t2i_r5, t2i_r10))
res_dict['t2i_r1'] = t2i_r1
res_dict['t2i_r5'] = t2i_r5
res_dict['t2i_r10'] = t2i_r10

# query: ori dataset, candidates: ori + contrast dataset; used in paper
ranks_t2i_ori = np.zeros(len(caption_list))
ori_index = 0
for index, image in enumerate(image_list):
    if '_' in os.path.basename(image):
        continue
    for i in range(5):
        ranks_t2i_ori[ori_index] = ranks_t2i[index*5+i]
        ori_index += 1
ranks_t2i_ori = ranks_t2i_ori[0: ori_index]

ori_t2i_r1 = 100.0 * len(np.where(ranks_t2i_ori < 1)[0]) / len(ranks_t2i_ori)
ori_t2t_r5 = 100.0 * len(np.where(ranks_t2i_ori < 5)[0]) / len(ranks_t2i_ori)
ori_t2i_r10 = 100.0 * len(np.where(ranks_t2i_ori < 10)[0]) / len(ranks_t2i_ori)
print("Original Text to Image: r1 %.1f; r5 %.1f; r10 %.1f" %(ori_t2i_r1, ori_t2t_r5, ori_t2i_r10))
res_dict['ori_t2i_r1'] = ori_t2i_r1
res_dict['ori_t2t_r5'] = ori_t2t_r5
res_dict['ori_t2i_r10'] = ori_t2i_r10

f_w = open('test_%s_%s.json'%(bench, os.path.basename(model_path)), 'w')
f_w.writelines(json.dumps(res_dict, ensure_ascii=False, indent=4))
f_w.close()



image_id_list = [info['image'] for info in data_list]
i2t_type_contrast_pair_dict = defaultdict(list)
t2i_type_contrast_pair_dict = defaultdict(list)
contrastive_aspect_num = defaultdict(int)
for index, data in enumerate(data_list):
    if 'contrastive_aspect' in data.keys():
        contrastive_aspect_num[data['contrastive_aspect']] += 1
        ori_image_id = data['image'].split('_')[0] + '.jpg'
        ori_image_index = image_id_list.index(ori_image_id)
        for id in range(5):
            t2i_type_contrast_pair_dict[data['contrastive_aspect']].append([ori_image_index*5 + id, ori_image_index, index])
        for id in range(5):
            t2i_type_contrast_pair_dict[data['contrastive_aspect']].append([index*5 + id, index, ori_image_index])
        
        for true_id in [ori_image_index*5+id for id in range(5)]:
            for false_id in [index*5+id for id in range(5)]:
                i2t_type_contrast_pair_dict[data['contrastive_aspect']].append([ori_image_index, true_id, false_id])

        for true_id in [index*5+id for id in range(5)]:
            for false_id in [ori_image_index*5+id for id in range(5)]:
                i2t_type_contrast_pair_dict[data['contrastive_aspect']].append([index, true_id, false_id])



contrastive_aspect_dict = {
    "Entity": ['Entity Type', 'Entity Attribute', 'Entity Relationship', 'Entity Emotion'],
    "Scene": ['Scene Type', 'Scene Attribute'],
    "Event": ['Event Category', 'Event Element'],
    "Style and Presentation": ['Style and Presentation'],
}
print(contrastive_aspect_num)
f_w = open('%s_%s_contrast_metric.txt'%(bench, os.path.basename(args.model_path)), 'w')

print('===================================Text to Image Evaluation=============================================')
for category, aspect_list in contrastive_aspect_dict.items():
    triple_list = []
    num = 0
    for contrastive_aspect in aspect_list:
        aspect_num = contrastive_aspect_num[contrastive_aspect]
        cda_acc = compute_CDA(t2i_similarity, t2i_type_contrast_pair_dict[contrastive_aspect], mode='t2i')
        print('Aspect: %s, image number: %i, CDE: %.2f'%(contrastive_aspect, aspect_num, cda_acc))
        f_w.writelines('Text to Image. Aspect: %s, image number: %i, CDE: %.2f\n'%(contrastive_aspect, aspect_num, cda_acc))

        triple_list.extend(t2i_type_contrast_pair_dict[contrastive_aspect])
        num += contrastive_aspect_num[contrastive_aspect]
    cda_acc = compute_CDA(t2i_similarity, triple_list, mode='t2i')
    print('Category: %s, image number: %i, CDE: %.2f'%(category, num, cda_acc))
    f_w.writelines('Text to Image. Category: %s, image number: %i, CDE: %.2f\n'%(category, num, cda_acc))


print('===================================Image to Text Evaluation=============================================')
for category, aspect_list in contrastive_aspect_dict.items():
    triple_list = []
    num = 0
    for contrastive_aspect in aspect_list:
        aspect_num = contrastive_aspect_num[contrastive_aspect]
        cda_acc = compute_CDA(i2t_similarity, i2t_type_contrast_pair_dict[contrastive_aspect], mode='i2t')
        print('Aspect: %s, image number: %i, CDE: %.2f'%(contrastive_aspect, aspect_num, cda_acc))
        f_w.writelines('Image to Text. Aspect: %s, image number: %i, CDE: %.2f\n'%(contrastive_aspect, aspect_num, cda_acc))

        triple_list.extend(i2t_type_contrast_pair_dict[contrastive_aspect])
        num += contrastive_aspect_num[contrastive_aspect]
    cda_acc = compute_CDA(i2t_similarity, triple_list, mode='i2t')
    print('Category: %s, image number: %i, CDE: %.2f'%(category, num, cda_acc))
    f_w.writelines('Image to Text. Category: %s, image number: %i, CDE: %.2f\n'%(category, num, cda_acc))
    

f_w.close()

