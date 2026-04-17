import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from datasets import load_dataset
import re

from scipy.ndimage import label
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.stats import multivariate_normal
from torchvision.utils import save_image

from refine import refine

from scipy.spatial.distance import cdist
DO_PLOT = False
METHOD = "attention" #"grad" attention
INDEX = -9
DO_PRUNE = False
LOAD_DATA = False
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions,multi_image, is15, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.is15 = is15
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.multi_image = multi_image



    def __getitem__(self, index, bbox = None , P = 24, split = False, n = 4, path = None, augmentation = None):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if True: #self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')


        

        if bbox is not None:

            a,b,width,height = bbox

            W = image.size[0]/P 
            H = image.size[1]/P
            
            x_min = int(a * W)
            x_max = int(x_min + width *W) 
            y_min = int(b * H)
            y_max = int(y_min + height *H)

            box_resized = (x_min, y_min, x_max, y_max)
            image = image.crop(box_resized)



        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')


       
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        if self.multi_image == False and self.is15 == False:
            image_tensor = image_tensor[0,:,:,:].unsqueeze(0)
        
        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)



def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions,multi_image,is15, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, multi_image,is15, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader, dataset




def eval_model(args):

 
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device = "cuda", use_flash_attn = False if METHOD == "grad" else False
)   
    model.requires_grad_(False)   # optional, if you never want model grads
    #model.gradient_checkpointing_enable()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    with open("/cluster/scratch/mgroepl/data/textvqa/val.json", "r") as f:
        gts = json.load(f)["data"] #answers
    
    data_loader,dataset = create_data_loader(questions,args.multi_image, args.is15, args.image_folder, tokenizer, image_processor, model.config, batch_size = 1)


    avg_pruning = 0
    index = -1
    datas_len = len(data_loader)
    vis_len = 576#1152


    if LOAD_DATA:
        box_data = load_dataset("jrzhang/TextVQA_GT_bbox")["train"]
        print(len(box_data))
        id_to_index = {ex["dataset_id"]: i for i, ex in enumerate(box_data)}
        ids = box_data["dataset_id"]
        id_to_bbox = {item["dataset_id"]: item["bbox"] for item in box_data}
        with open("/cluster/scratch/mgroepl/data/textvqa/val.json", "r") as f:
            img_to_id = json.load(f)["data"]
        ordered_ids = [item["question_id"] for item in img_to_id]
        new_order = [id_to_index[str(qid)] for qid in ordered_ids if str(qid) in id_to_index]
        index_data = -1
        box_data = box_data.select(new_order)




    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):

        img_id = line["image"]
        
        index += 1
        if LOAD_DATA:
            ordered_id = ordered_ids[index]
            print(ordered_id)
            if str(ordered_id) in id_to_index:
                index_data += 1
                box_dataset_orig = box_data[index_data]["bbox"]
                box_dataset_orig_size = (box_dataset_orig[2]*box_dataset_orig[3]) / (image_sizes[0][0]*image_sizes[0][1])
            else:
                box_dataset_orig_size = -1.0
        else:
            box_dataset_orig_size = -1.0
        print(index)
        if index < INDEX: #96
            continue

        question = line["text"]
        #gt_answer = gts[index]["answers"]




        print(img_id)
        idx = line["question_id"]
        cur_prompt = line["text"]

       
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        x_flat = input_ids.flatten()
        begin_pos_vis = (x_flat == -200).nonzero(as_tuple=True)[0].item()
        #delete_images_in_folder("/cluster/scratch/mgroepl/heatmaps/mean/")
     


        if True:#with torch.inference_mode():

            print("image shape: ", image_tensor.shape)
 


            orig_embeds , att_mask, position_ids= get_embedding(model,input_ids,image_tensor,image_sizes)

            orig_embeds = refine(model,input_ids,image_tensor,image_sizes, orig_embeds , att_mask, position_ids, dataset, index )
           
            attn_layers = model(
                #input_ids = input_ids,    to(torch.float32)
                attention_mask=att_mask,
                #images=[image_tensor.to(dtype=torch.float32)],
                inputs_embeds = orig_embeds,
                image_sizes=None,
                output_attentions=False,
                return_dict=True,
            )

            logits = attn_layers.logits
            last_logits = logits[0, -1]  
            probs = torch.softmax(last_logits, dim=-1)

        
            outputs = model.generate(
                None,
                attention_mask = None,
                inputs_embeds = orig_embeds,
                #images=image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),58
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=100,#args.max_new_tokens,
                
                output_hidden_states=False,
                return_dict_in_generate=True, 
                use_cache=False)

            sequences = outputs.sequences

            text2 = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            print("text2:",text2)     
            
         
        print("box_dataset_orig_size: ", box_dataset_orig_size)
        print("output:",text2 )
        print(img_id)
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": text2 if args.inject else text,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {},
                                   "box_size": box_dataset_orig_size }) + "\n",
                                   )
        # ans_file.flush()
        
    print(f"average pruning: {avg_pruning/len(dataset)}")
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--multi_image", type=bool, default=False)
    parser.add_argument("--inject", type=bool, default=False)
    parser.add_argument("--is15", type=bool, default=False)
    args = parser.parse_args()
    print("args.inject", args.inject)
    print("multi_image", args.multi_image)
    eval_model(args)
