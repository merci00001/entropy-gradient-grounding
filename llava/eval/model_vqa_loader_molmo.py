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
#from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
#from llava.conversation import conv_templates, SeparatorStyle
#from llava.model.builder import load_pretrained_model
#from llava.utils import disable_torch_init
#from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image, ImageDraw
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from torchvision.utils import save_image

from groundingdino.util.inference import load_model, load_image, predict, annotate

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, model, processor):
        self.questions = questions
        self.image_folder = image_folder
        self.model = model
        self.processor = processor




    def __getitem__(self, index, bbox = None , P = 24, split = False, n = 4, path = None, augmentation = None):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
   
        prompt = qs
    
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
            image.save("/cluster/project/cvg/students/mgroepl/LLaVA/llava/eval/debug.png")


        inputs = self.processor.process(
            images=[image],
            text="Describe this image."
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
      
        
        return inputs   #["input_ids"], inputs["images"], inputs["image_input_idx"], inputs["image_masks"], image.size

    def __len__(self):
        return len(self.questions)



def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, model, processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder,model, processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader, dataset


def bbox_from_mask(mask) :
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return x0, y0, width, height






def bbox_from_att_image_adaptive(att_map, image_size, begin_pos_vis, bbox_size=336, vis_len = 576 ):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """
    attn_last_to_vis = att_map[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

    # Step 2: Squeeze query dimension (it's size 1)
    attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
    


    selected = analyze_heads( attn_last_to_vis.detach().cpu())

    selected = selected[:3]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())

    newAtt = torch.zeros((att_map.shape[2],att_map.shape[3]), dtype = float, device = "cuda")

    for x in selected:
        l = x["layer"]
        h = x["head"]
        newAtt += att_map[l,h,:,:]
    
    att_map = newAtt[-1, begin_pos_vis:begin_pos_vis + vis_len].reshape((24,24)).cpu().numpy()
    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2



def elbow_chord(values):
    # Returns threshold value (y), not index
    if len(values) <= 2:
        return min(values) if values else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)  # ascending
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return y[0]
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def binarize_mean_relu(M, ent = None):
    d = 1.0
  
    if False: # M.sum() < 2.4 and ent > 1.5:
        d = 1.5
    m = M.mean() * d
    

    B = np.maximum(M - m, 0.0)
    return (B > 0).astype(np.uint8)


def spatial_entropy(attn_map_2d: torch.Tensor, threshold: float):
    # attn_map_2d: [P, P]
    S = attn_map_2d
    mean_val = torch.mean(S)
    B = torch.relu(S - mean_val*2)
    B_np = B.detach().cpu().to(torch.float32).numpy()
    binary = (B_np > threshold).astype(np.int32)

    from scipy.ndimage import label
    labeled, num = label(binary, structure=np.ones((3, 3)))

    total = float(B.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}

    # Probability mass per component
    probs = []
    for i in range(1, num + 1):
        comp_sum = B_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}

def analyze_heads(attn: torch.Tensor):
    """Analyze heads and return a ranked list.

    attn: [L, H, 1, V]
    meta: includes patch_size (P)
    """
    L, H, _, V = attn.shape
    P = 24 #int(meta.get("patch_size", int(np.sqrt(V))))

    # Criterion 1: head sums over image patches
    sums = []
    for l in range(L):
        for h in range(H):
            s = float(attn[l, h, 0].sum().item())
            sums.append(s)

    thr_val = elbow_chord(sums) #if cfg.logic.threshold.method == "chord" else min(sums)

    # Analyze Criterion 2 only for heads above thr_val (by value)
    results =  []
    idx = 0
    for l in range(L):
        for h in range(H):
            s = sums[idx]
            idx += 1
            if s < thr_val:
                se = float("inf")
                bottom_row_focus = False
                n_comp = 0
            else:
                a2d = attn[l, h, 0].reshape(P, P)
                se_res = spatial_entropy(a2d, 0.001)
                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > 0.05).any())
                se = float(se_res["spatial_entropy"])    # lower is better
                labeled = se_res["labeled_array"]
                n_comp = int(se_res["num_components"])
            results.append({
                "layer": l,
                "head": h,
                "attn_sum": s,
                "spatial_entropy": se,
                "bottom_row_focus": bottom_row_focus,
                "num_components": n_comp,
            })

    # Filter and sort: keep heads above threshold, prefer non-bottom-row
    kept = [r for r in results if np.isfinite(r["spatial_entropy"]) and r["attn_sum"] >= thr_val and not r["bottom_row_focus"] and r["layer"] > 1]
    if len(kept) < 1:
        # fallback: take top by sum if too few
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: 1]

    kept.sort(key=lambda x: x["spatial_entropy"])  # ascending
    return kept

def combine_heads(attn: torch.Tensor, selected, P, sigma):
    """Combine selected heads with optional Gaussian smoothing.

    attn: [L, H, 1, V]
    Returns: combined 2D map [P, P] as numpy float32
    """
    M = np.zeros((P, P), dtype=np.float32)
    ent = 0
    for item in selected:
        l, h = item["layer"], item["head"]
        ent += item["spatial_entropy"]
        a2d = attn[l, h, 0].reshape(P, P).detach().cpu().to(torch.float32).numpy()
        if sigma and sigma > 0:
            a2d = gaussian_filter(a2d, sigma=sigma)
        M += a2d.astype(np.float32)
 
    return M, ent
def get_LoRa(att, r = 10):
    U, S, Vh = torch.linalg.svd(att, full_matrices=False)
# Truncate to rank r
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    # Construct A and B (closed-form)
    A = U_r @ torch.diag(S_r.sqrt())
    B = torch.diag(S_r.sqrt()) @ Vh_r

    # LoRA approximation: A @ B ≈ W
    W_lora = A @ B
    return W_lora

def plot_mask(img_path, top_indices, vis_len, question_id, question, base_path = "", dataset = ""):
    P = int(np.sqrt(vis_len))
    
    mask = torch.ones(vis_len, dtype=torch.bool, device = "cuda")
    mask[top_indices] = False
    
    mask = mask.reshape((P,P))
    if dataset == "pope":
        path = "/cluster/scratch/mgroepl/pope/val2014/"
    else:
        path = "/cluster/scratch/mgroepl/data/textvqa/train_images/"
    
    img = Image.open(f"{path}{img_path}").convert("RGB")
    img_np = np.array(img)  # Convert to NumPy array
    img_h, img_w = img_np.shape[:2]
    mask_h, mask_w = mask.shape

    mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to 0–255 for PIL

    # Create PIL image and resize
    mask_pil = Image.fromarray(mask_np)
    resized_mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), resample=Image.NEAREST)


    resized_mask = np.array(resized_mask_pil) > 127  # Convert back to boolean

    # Apply the mask
    masked_img = img_np.copy()
    masked_img[~resized_mask] = 0

    masked_pil = Image.fromarray(masked_img)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Masked image
    axes[1].imshow(masked_pil)
    axes[1].set_title("Masked Image")
    axes[1].axis("off")

    plt.suptitle(f"Question : {question}", fontsize=16)

    # Save and close
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
    plt.savefig(f"/cluster/scratch/mgroepl/ownSeg/{question_id}.jpg", dpi=300)
    plt.close()

def get_size_object(attn_layers, P,begin_pos_vis_att,vis_len):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len].cpu()
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[: 13], P=P, sigma=2.0)
    
    return   combo.var(), combo.mean(), ent

def get_disjoint_segments(attn_layers, P,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None):
    ent  = 0
    if insert_mask is None:
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        selected = analyze_heads( attn_last_to_vis.detach().cpu())
    

        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=2.0)
        
        mask_grid = binarize_mean_relu(combo, ent)
    else:
        mask_grid = insert_mask
    if return_single:
        return [mask_grid]#[bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
    boxes = [bbox_from_mask(x) for x in segment_masks]
    
    return segment_masks, ent

def get_bbox_indices(attn_layers, P,begin_pos_vis_att, vis_len = 576 , do_grid = True, returnBBOX = False):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=2.0)
    mask_grid = binarize_mean_relu(combo, ent)
    if do_grid:
        flattened = torch.flatten(torch.from_numpy(mask_grid))
        indices = torch.where(flattened)[0]
        return indices, ent
    bbox_grid = bbox_from_mask(mask_grid)
    if returnBBOX:
        return bbox_grid, ent


    return box_to_indices(bbox_grid, P), ent

def box_to_indices(bbox, P):
    x,y,w,h = bbox
    x1 = x
    y1 = y
    x2 = x + w -1
    y2 =  y + h -1
    ys, xs = np.meshgrid(np.arange(y1, y2 + 1), np.arange(x1, x2 + 1), indexing='ij')

    # Convert (y, x) to 1D indices in a flattened image
    indices = ys * P + xs
  
    return indices.flatten()
  

def get_indices_percent(attn_layers, begin_pos_vis_att, vis_len = 576, mode = "selected", topK = 0.9, largest = False, sample = False,attn_mean_all = None, general_att_map = None, width = 1, height = 1):
    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

    # Step 2: Squeeze query dimension (it's size 1)
    attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
    

    if mode == "topK":
        attn_mean_all = attn_mean_heads.mean(dim=0) 
        
    elif mode == "selected":
        selected = analyze_heads( attn_last_to_vis.detach().cpu())

        selected = selected[:3]
        newAtt = torch.zeros((attn_layers.shape[2],attn_layers.shape[3]), dtype = float, device = "cuda")
        print(newAtt.shape)
        for x in selected:
            l = x["layer"]
            h = x["head"]
            newAtt += attn_layers[l,h,:,:]
        
        attn_mean_all = newAtt[-1, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    elif mode == "general":
 

        attn_pic = attn_layers[14,:,0,begin_pos_vis_att:begin_pos_vis_att + vis_len].mean(0).reshape(24,24) / general_att_map
        attn_pic = attn_pic.cpu().numpy()
        attn_pic = gaussian_filter(attn_pic.astype(np.float32), sigma=2.0)
        attn_mean_all = attn_pic.flatten()
    top_k = int(topK * attn_mean_all.shape[0]) 
    if sample:
        attn_scores = attn_mean_all.clone()  # don't modify the original
        attn_scores = attn_scores - attn_scores.min()  # optional: make all scores non-negative
        prob = attn_scores / attn_scores.sum()
        sampled_indices = torch.multinomial(prob, num_samples=top_k, replacement=False)
        return sampled_indices
    if False:
        indeces = torch.arange(attn_mean_all.shape[0]).reshape((24,24)).cuda()
        attn_mean_all_reshaped = attn_mean_all.reshape((24,24))
        w_sum = attn_mean_all_reshaped.sum(dim = 0)

        top_values, top_indices = torch.topk(w_sum, k=width, largest = largest)
       
        total_ind = []
        for x in top_indices:
            h_val = attn_mean_all_reshaped[x,:]
            top_values, top_indices = torch.topk(h_val, k=height, largest = largest)
            total_ind.append(indeces[x,top_indices])
        total_ind = torch.stack(total_ind).flatten()
        return total_ind
    else:
        top_values, top_indices = torch.topk(attn_mean_all, k=top_k, largest = largest)
        return top_indices

def prune_embeds(attn_layers, inputs_embeds, begin_pos_vis,begin_pos_vis_att, vis_len = 576, mode = "selected", topK = 0.9, remove_top =  False, img_path = None, question_id = None, question = None, ):
    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

    # Step 2: Squeeze query dimension (it's size 1)
    attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
    




    if mode == "topK":
        attn_mean_all = attn_mean_heads.mean(dim=0) 
        
    elif mode == "selected":
        selected = analyze_heads( attn_last_to_vis.detach().cpu())

        selected = selected[:5]
        newAtt = torch.zeros((attn_layers.shape[2],attn_layers.shape[3]), dtype = float, device = "cuda")

        for x in selected:
            l = x["layer"]
            h = x["head"]
            newAtt += attn_layers[l,h,:,:]
        
        attn_mean_all = newAtt[-1, begin_pos_vis_att:begin_pos_vis_att + vis_len]

    top_k = int(topK * attn_mean_all.shape[0]) 
    top_values, top_indices = torch.topk(attn_mean_all, k=top_k, largest = True if remove_top else False)
    
    #t = torch.sum(attn_mean_all)
    #top_indices = torch.multinomial(t-attn_mean_all, top_k, replacement=False)
    #print(top_indices)
    if img_path is not None:
        plot_mask(img_path, top_indices,vis_len, question_id = question_id, question = question)

    mask = torch.ones(inputs_embeds.size(1), dtype=torch.bool, device = "cuda")
    
    top_indices +=begin_pos_vis 
    if False:
        random_indices = torch.randint(0, vis_len, (len(top_indices),))
        random_indices += begin_pos_vis
        for i in range(len(top_indices)):
            a, b = top_indices[i].item(), random_indices[i].item()
            # Swap the slices at y=a and y=b along dim=1
            tmp_a = inputs_embeds[:, a, :].clone()
            tmp_b = inputs_embeds[:, b, :].clone()

            inputs_embeds[:, a, :] = tmp_b
            inputs_embeds[:, b, :] = tmp_a

        return inputs_embeds
    else:
        mask[top_indices] = False

        inputs_embeds_pruned = inputs_embeds[:, mask, :]
        return inputs_embeds_pruned


def return_vis_pruned(embeds, indices,vis_len, begin_pos_vis, invert = False):
    if invert:
        indices_new = []
        for x in range(vis_len):
            if x not in indices:
                indices_new.append(x)
        indices = torch.tensor(indices_new, dtype = torch.int64)
    vis_only = embeds[:,begin_pos_vis:begin_pos_vis + vis_len,:]

    mask = torch.ones(vis_only.size(1), dtype=torch.bool, device = "cuda")
    mask[indices] = False
    inputs_embeds_pruned = vis_only[:, mask, :]
    return inputs_embeds_pruned

def prune_indices(embeds, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True):
    if invert:
        indices_new = []
        for x in range(vis_len):
            if x not in indices:
                indices_new.append(x)
        indices = torch.tensor(indices_new, dtype = torch.int64)
    if len(indices) == 0:
        return embeds
    if img_path is not None:
        plot_mask(f"{img_path}", indices, vis_len, question_id, question)
    indices += begin_pos_vis
    mask = torch.ones(embeds.size(1), dtype=torch.bool, device = "cuda")
    mask[indices] = False
    inputs_embeds_pruned = embeds[:, mask, :]
    return inputs_embeds_pruned

def get_overlapping_patch_indices_flat(bbox, patch_size, image_size):
    """
    bbox: [x, y, w, h] in pixel coordinates
    patch_size: int (P)
    image_size: (H, W) in pixels
    returns: list of flat indices of patches overlapping the bbox
    """
    x, y, w, h = bbox
    H, W = image_size

    # Compute patch grid size
    num_patches =patch_size * patch_size
    patchX = H / patch_size
    patchY = W / patch_size

    # Bounding box coordinates
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    # Convert bbox coordinates to patch indices
    j_start = max(0, x_min // patchX)
    j_end   = min(patch_size - 1, (x_max) // patchX)

    i_start = max(0, y_min // patchY)
    i_end   = min(num_patches - 1, (y_max ) // patchY)

    # Collect all flat indices
    overlapping_flat_indices = []
    for p in range(num_patches):
        p_x = p % patch_size
        p_y = p // patch_size
        if p_x <= j_end and p_x >= j_start and p_y <= i_end and p_y >= i_start:
            overlapping_flat_indices.append(p)


    return overlapping_flat_indices




def build_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    """
    Re-creates the combined causal + padding mask used inside LLaMA/LLaVA.
    attention_mask: [B, T] (1 for valid tokens, 0 for pad)
    input_shape: (B, T, H) or (B, T)
    inputs_embeds: [B, T, H]
    Returns: [B, 1, T, T] mask with 0 for keep, -inf for mask
    """
    if len(input_shape) == 3:
        bsz, tgt_len, _ = input_shape
    else:
        bsz, tgt_len = input_shape

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # 1. Causal mask (upper-triangular)
    causal_mask = torch.full(
        (tgt_len, tgt_len),
        fill_value=float("-inf"),
        device=device,
        dtype=dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    # 2. Combine with padding mask
    if attention_mask is not None:
        padding_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(dtype).min
        combined_mask = causal_mask + padding_mask
    else:
        combined_mask = causal_mask

    return combined_mask

def project_embed(model, embeds):
    

    projector = model.get_model().mm_projector  
    embeds = projector(embeds.half())
    return embeds




def get_clip_embed(model,image_tensor):

    image_features = model.get_model().get_vision_tower()([image_tensor.squeeze(0).squeeze(0)])
    return image_features[0][0]
   


def get_embedding(model,input_ids,image_tensor,image_sizes, new_pos = None,orig_pos=None):

   
    (
        _,
        position_ids,
        att_mask,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        image_sizes = image_sizes,
        #new_pos = new_pos,
        #orig_pos = orig_pos
    )

    return inputs_embeds, att_mask, position_ids
   
def get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = None,num_layer = None, attention_mask = None,position_ids = None ):



    if input_embeds is not None and num_layer is None:
        
        attn_layers = model.to(torch.float32)(
            #input_ids = input_ids,    
            attention_mask=attention_mask,
            #images=[image_tensor.to(dtype=torch.float32)],
            inputs_embeds = input_embeds,
            image_sizes=None,
            output_attentions=True,
            return_dict=True,
        
        
        )

    elif num_layer is not None:
        encoder_layers = model.model.layers
        hidden_states = input_embeds
        
        if position_ids == None:
            position_ids = torch.arange(input_embeds.shape[1], dtype=torch.long, device=input_embeds.device).unsqueeze(0)
        attn_outputs = []
        num_layers_to_run = num_layer
        attention_mask = build_decoder_attention_mask(
            attention_mask,
            input_embeds.size(),
            input_embeds
        )
        for i, layer in enumerate(encoder_layers[:num_layers_to_run]):
            # Pass through one layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
                past_key_value=None,
            )

            # Get hidden states and attentions
            hidden_states = layer_outputs[0]  # updated hidden state
            if i == 0:
                first_state = hidden_states
            attn_outputs.append(layer_outputs[1][:,:,-1:,:])
        attn_outputs = torch.stack(attn_outputs).squeeze(1).cuda()
        return attn_outputs, first_state


    attn_layers = model(    
        input_ids=input_ids,
        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        image_sizes=image_sizes,
        output_attentions=True,
        return_dict=True,
    )

    attn_layers = attn_layers.attentions # tuple length L of [B,H,Tq,Tk]
    attn_layers = torch.stack(attn_layers, dim=0)  # [L, B, H, Tq, Tk]
    attn_layers = attn_layers[:, 0] 
    return attn_layers


def plot_arrays(arr1,arr2, name = "plot"):

    plt.figure(figsize=(8, 5))
    plt.scatter(arr1, arr2, c='blue', alpha=0.7)
    plt.xscale('log')  # log scale to spread out small variances
    plt.xlabel('Variance (log scale)')
    plt.ylabel('Size')
    plt.title('Size vs Variance')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(f'/cluster/scratch/mgroepl/res/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Closes the figure and frees memory
def compute_iou(indices, att ):
    att = att.mean(dim = (0,1)).squeeze(0)
    print(att.shape)
    total_att = att.sum()
    gt_att = att[indices].sum()

    return gt_att / total_att


def delete_images_in_folder(folder_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            continue
        
        # Remove extension and check if the name ends with a number
        name_without_ext = os.path.splitext(filename)[0]
        
        # Match if the name ends with digits
        if not re.search(r'\d+$', name_without_ext):
            continue  # Skip files that don't end in a number

        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
    return images

def get_candidate(images):
    current_max = 0
    current = None
    for x in images:
        img_arr = np.array(x.convert('L'), dtype=np.int16)
        sum = img_arr.sum()
        if sum > current_max:
            current = x
            current_max = sum

    return current

    

def subtract_images(images, reference_image):
    # Ensure reference is grayscale and convert to array
    ref_arr = np.array(reference_image.convert('L'), dtype=np.int16)


    for img in images:
        img_arr = np.array(img.convert('L'), dtype=np.int16)
        
        # Subtract (ref - img)
        ref_arr = ref_arr - img_arr
        
        # Clip values to valid range [0, 255]
        ref_arr = np.clip(ref_arr, 0, 255).astype(np.uint8)
     
        # Convert back to PIL Image

    
    return ref_arr





def get_unique_filename( base_name, extension=".png"):
    """
    Generates a unique filename by appending _1, _2, ... if needed
    """
    filename = f"{base_name}{extension}"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    return filename

def crop_to_match(img1, img2):
    H1, W1, C = img1.shape
    H2, W2 = img2.shape[0], img2.shape[1]
    rows_to_keep = torch.linspace(0, H1 - 1, H2).long()
    cols_to_keep = torch.linspace(0, W1 - 1, W2).long()
    img_cropped = img1[rows_to_keep[:, None], cols_to_keep, :]
    return img_cropped







def get_image(dataset, index, box = None, P = 24):
    
        line = dataset.questions[index]
        image_file = line["image"]
        qs = line["text"]

    
        image = Image.open(os.path.join(dataset.image_folder, image_file)).convert('RGB')
        if box is not None:
            a,b,width,height = box

            W = image.size[0]/P 
            H = image.size[1]/P
            
            x_min = int(a * W)
            x_max = int(x_min + width *W) 
            y_min = int(b * H)
            y_max = int(y_min + height *H)
            box_resized = (x_min, y_min, x_max, y_max)
            image = image.crop(box_resized)            
        return image



def to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


def overlap_pixels(box1, box2):
    # convert to (xmin, ymin, xmax, ymax)
    x1_min, y1_min, x1_max, y1_max = to_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = to_xyxy(box2)

    # intersection edges
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # intersection width/height
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)

    # absolute pixel overlap
    return inter_w * inter_h


def find_best_match_by_overlap(target_box, boxes):
    best_overlap = 0
    best_box = None

    for box in boxes:
        overlap = overlap_pixels(target_box, box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_box = box

    return best_box, best_overlap


def euclidean_distance_image_torch(image, vector):
    """
    image:  (H, W, N) torch tensor
    vector: (N,) torch tensor
    returns: (H, W) distance map (not normalized)
    """
    diff = image - vector  # broadcast across pixels
    dist = torch.norm(diff, dim=-1)  # Euclidean distance per pixel
    return dist

def cosine_similarity_image_torch(image, vector):
    """
    image:  (H, W, N) torch tensor
    vector: (N,) torch tensor
    returns: (H, W) torch tensor
    """

    # Normalize vector
    vector = vector / vector.norm()

    # Normalize image per pixel
    image_norm = image / image.norm(dim=-1, keepdim=True)

    # Dot product across channel dimension
    cos_sim = (image_norm * vector).sum(dim=-1)

    return cos_sim



def inject_features(model, input_ids, image_tensor, image_sizes,begin_pos_vis, vis_len, dataset,  index ):


    with torch.inference_mode():


        orig_embeds , att_mask, position_ids= get_embedding(model,input_ids,image_tensor,image_sizes)
        orig_embeds_vis = orig_embeds[0,begin_pos_vis:begin_pos_vis + vis_len,:]

        attn , first_hidden_orig = get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = orig_embeds, num_layer = 25, attention_mask = att_mask,position_ids = position_ids )

        box, ent = get_bbox_indices(attn, 24,begin_pos_vis, vis_len = 576 , do_grid = False, returnBBOX = True)

        #ind = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = 0.1, largest = True)
        boxes = get_disjoint_segments(attn, 24,begin_pos_vis, vis_len = 576, return_single = False ,insert_mask = None)
        for ind, m in enumerate(boxes):
            box = list(bbox_from_mask(m))
            bsize = box[2] * box[3]
            
            _, image_tensor2, image_sizes2 = dataset.__getitem__(index  , bbox = box , P = 24, split = False, n = 4, path = None, augmentation = None)  #f"{output_dir}/cut.png"
            
            flattened = torch.flatten(torch.from_numpy(m))
            indices_orig = torch.where(flattened)[0]
            #indices_orig =torch.tensor( box_to_indices(box, 24)  )
            emb_vis, att_mask, position_ids = get_embedding(model,input_ids,image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_sizes2])
            if len(boxes) == 1:
                orig_embeds = emb_vis
                continue
            emb_vis_pruned = emb_vis[0,begin_pos_vis:begin_pos_vis + vis_len,:]
            attn, first_hidden  = get_attn_layers(model,input_ids, image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),None, input_embeds = emb_vis, num_layer = 15, attention_mask = att_mask, position_ids = position_ids)
            indices = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = len(indices_orig) / (24*24), largest = True, attn_mean_all = attn[:,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0), general_att_map = None) #bsize / (24*24)
            orig_embeds_vis_cut = prune_indices(emb_vis, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)[:,begin_pos_vis:begin_pos_vis + len(indices_orig) ,:]                   
            print(orig_embeds_vis_cut.shape)
            
            orig_embeds[:,indices_orig + begin_pos_vis,:] = orig_embeds_vis_cut

    return orig_embeds

def expand_bbox(box, expand = 1):
    x,y,w,h = box
    new_x = max(0,x-expand)
    new_y = max(0,y-expand)
    new_w = min(w + expand, 23 - x - w)
    new_h = min(w + expand, 23 - y - h)
    return [int(new_x), int(new_y), int(new_w), int(new_h)]


def eval_model(args):

    model_dino = load_model("/cluster/project/cvg/students/mgroepl/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/cluster/scratch/mgroepl/DINO/groundingdino_swint_ogc.pth")
    # Model



    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    print(processor)
    
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    model.config.output_attentions = True

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    data_loader,dataset = create_data_loader(questions, args.image_folder, model, processor,  batch_size = 1)
    inputs = dataset[0]
    outputs = model(**inputs, output_attentions=True)

    return
    if False:
        box_data = load_dataset("jrzhang/TextVQA_GT_bbox")["train"]
        ids = box_data["dataset_id"]
        id_to_bbox = {item["dataset_id"]: item["bbox"] for item in box_data}
        with open("/cluster/scratch/mgroepl/data/textvqa/val.json", "r") as f:
            img_to_id = json.load(f)["data"]
        imgid_to_id = {item["image_id"]: str(item["question_id"]) for item in img_to_id}
        imgid_to_width = {item["image_id"]: item["image_width"] for item in img_to_id}
        imgid_to_height = {item["image_id"]: item["image_height"] for item in img_to_id}

    avg_pruning = 0
    index = -1
    datas_len = len(data_loader)
    vis_len = 576#1152

    IoUDict = {}
    IoUDict2 = {}

    cut= []

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        index += 1
        print(index)
        if index < -13:
            continue
        question = line["text"]
  



        idx = line["question_id"]
        cur_prompt = line["text"]
        img_id = line["image"]
        
       
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        x_flat = input_ids.flatten()
        begin_pos_vis = (x_flat == -200).nonzero(as_tuple=True)[0].item()
        #delete_images_in_folder("/cluster/scratch/mgroepl/heatmaps/mean/")
     


        with torch.inference_mode():


            orig_embeds , att_mask, position_ids= get_embedding(model,input_ids,image_tensor,image_sizes)
  
   
            attn , first_hidden_orig = get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = orig_embeds, num_layer = 25, attention_mask = att_mask,position_ids = position_ids )



            #box, ent = get_bbox_indices(attn, 24,begin_pos_vis, vis_len = 576 , do_grid = False, returnBBOX = True)
       
            #ind = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = 0.1, largest = True)
            boxes, ent = get_disjoint_segments(attn, 24,begin_pos_vis, vis_len = 576, return_single = False ,insert_mask = None)
            total_mask = torch.zeros((24,24), dtype = bool)

            black_image_rgb = Image.new("RGB", image_sizes[0], (0, 0, 0))
        

            for ind, m in enumerate(boxes):
                total_mask += m
                box = list(bbox_from_mask(m))
                bsize = box[2] * box[3]

                _, image_tensor2, image_sizes2 = dataset.__getitem__(index  , bbox = box , P = 24, split = False, n = 4, path = None, augmentation = None)  #f"{output_dir}/cut.png"
                if False:
                    print(bsize)
                    box = expand_bbox(box, expand = 1)
                    print(box)

                indices_orig =torch.tensor( box_to_indices(box, 24)  )
                
                ratio = (image_sizes2[0] * image_sizes2[1])  / (image_sizes[0][0] * image_sizes[0][1])

                if False: #ratio > 0.3:
                    emb_vis = inject_features(model, input_ids, image_tensor2.unsqueeze(0), image_sizes2,begin_pos_vis, vis_len, dataset,  index )
                    #orig_embeds = emb_vis
                    #continue
                else:                        
                    emb_vis, att_mask, position_ids = get_embedding(model,input_ids,image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_sizes2])

                
                emb_vis_pruned = emb_vis[0,begin_pos_vis:begin_pos_vis + vis_len,:]
                attn, first_hidden  = get_attn_layers(model,input_ids, image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),None, input_embeds = emb_vis, num_layer = 15, attention_mask = att_mask, position_ids = position_ids)
                x_new,y_new,x_new1, y_new1 = bbox_from_att_image_adaptive(attn, image_sizes2, begin_pos_vis, bbox_size=336 / 2)

                im = get_image(dataset, index, box = box, P = 24)

                im2 = im.crop( [x_new,y_new,x_new1,y_new1])
                im.save(f"/cluster/project/cvg/students/mgroepl/LLaVA/llava/eval/testorig{ind}.png")
                im2.save(f"/cluster/project/cvg/students/mgroepl/LLaVA/llava/eval/test{ind}.png")
                x_new = int(x_new // 24)
                y_new = int(y_new // 24)
                x_new1 = int(x_new1 // 24)
                y_new1 = int(y_new1 // 24)


                

                bbox_reshaped = [x_new, y_new,  x_new1 - x_new + 1 ,y_new1 - y_new + 1   ]

                
                #print("debug: ", bsize, indices_orig.shape[0], box[2],box[3])
                indices = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = len(indices_orig) / (24*24), largest = True, attn_mean_all = attn[:,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0), general_att_map = None, width = box[2] , height =box[3]  ) #bsize / (24*24)
                orig_embeds_vis_cut = prune_indices(emb_vis, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)[:,begin_pos_vis:begin_pos_vis + len(indices_orig) ,:]                   
                print(orig_embeds_vis_cut.shape)
               
                orig_embeds[:,indices_orig + begin_pos_vis,:] = orig_embeds_vis_cut


            
            flattened = torch.flatten(total_mask)
            
            indices= torch.where(flattened)[0]
            orig_embeds_pruned = prune_indices(orig_embeds, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)     
            avg_pruning +=orig_embeds.shape[1] -  orig_embeds_pruned.shape[1]
            outputs = model.generate(
                input_ids,
                #attention_mask = mask,
                inputs_embeds = orig_embeds,
                #images=image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=100,#args.max_new_tokens,
                
                output_hidden_states=True,
                return_dict_in_generate=True, 
                use_cache=True)

            sequences = outputs.sequences

            text2 = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            print("text2:",text2)     
        
        
        print("output:",text2 )
        print(img_id)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": text2,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
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
    args = parser.parse_args()

    eval_model(args)
