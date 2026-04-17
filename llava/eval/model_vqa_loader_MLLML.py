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

from scipy.spatial.distance import cdist
DO_PLOT = False
METHOD = "attention" #"grad" attention
INDEX = -45
LAYER = 14
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
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config



    def __getitem__(self, index, bbox = None , P = 24, split = False, n = 4, path = None, general = False, insert_image = None):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if general:
            qs = 'Write a general description of the image. Answer the question using a single word or phrase.'
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


        if insert_image is not None:
            image = insert_image

        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        #if image_tensor.shape[0] != 3:
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
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
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






def bbox_from_att_image_adaptive(att_map: np.ndarray, image_size: tuple,
                                 bbox_size: int = 336) -> tuple:
    """
    Find the bbox of the most salient region in a 2D attention map.

    Args:
        att_map:    2D numpy array, e.g. (24, 24) or (48, 48).
        image_size: (width, height) of the target image in pixels.
        bbox_size:  Base crop size in pixels (default 336).

    Returns:
        (x1, y1, x2, y2) pixel coordinates clipped to image bounds.
    """
    ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    block_size = (image_size[0] / att_map.shape[1],
                  image_size[1] / att_map.shape[0])

    best_diff, best_pos, best_block, best_ratio = -np.inf, (0, 0), (1, 1), 1.0

    for ratio in ratios:
        bw = min(int(bbox_size * ratio / block_size[0]), att_map.shape[1])
        bh = min(int(bbox_size * ratio / block_size[1]), att_map.shape[0])

        if att_map.shape[1] - bw < 1 and att_map.shape[0] - bh < 1:
            if ratio == 1.0:
                return 0, 0, image_size[0], image_size[1]
            continue

        # Integral image for O(1) rectangle sums
        integral = np.cumsum(np.cumsum(att_map, axis=0), axis=1)
        out_h, out_w = att_map.shape[0] - bh + 1, att_map.shape[1] - bw + 1
        y_idx, x_idx = np.mgrid[0:out_h, 0:out_w]
        y2, x2 = y_idx + bh - 1, x_idx + bw - 1

        sliding_att  = integral[y2, x2]
        sliding_att -= np.where(y_idx > 0, integral[y_idx - 1, x2], 0)
        sliding_att -= np.where(x_idx > 0, integral[y2, x_idx - 1], 0)
        sliding_att += np.where((y_idx > 0) & (x_idx > 0), integral[y_idx - 1, x_idx - 1], 0)

        flat_idx  = np.argmax(sliding_att)
        iy, ix    = np.unravel_index(flat_idx, sliding_att.shape)
        max_att   = float(sliding_att[iy, ix])
        max_pos   = (ix, iy)  # (x, y)

        neighbours = []
        if ix > 0:              neighbours.append(sliding_att[iy, ix - 1])
        if ix < out_w - 1:     neighbours.append(sliding_att[iy, ix + 1])
        if iy > 0:              neighbours.append(sliding_att[iy - 1, ix])
        if iy < out_h - 1:     neighbours.append(sliding_att[iy + 1, ix])
        diff = (max_att - np.mean(neighbours)) / (bw * bh) if neighbours else 0.0

        if diff > best_diff:
            best_diff, best_pos, best_block, best_ratio = diff, max_pos, (bw, bh), ratio

    half = bbox_size * best_ratio / 2
    x_center = int(best_pos[0] * block_size[0] + block_size[0] * best_block[0] / 2)
    y_center = int(best_pos[1] * block_size[1] + block_size[1] * best_block[1] / 2)
    x_center = int(np.clip(x_center, half, image_size[0] - half))
    y_center = int(np.clip(y_center, half, image_size[1] - half))

    return (
        max(0, int(x_center - half)),
        max(0, int(y_center - half)),
        min(image_size[0], int(x_center + half)),
        min(image_size[1], int(y_center + half)),
    )



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


def binarize_mean_relu(M, ent = None, do_max = False):
    d = 1.0
    

    m = M.mean() * d
    if do_max:
        m = M.max() * 0.5

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
            a2d =  gaussian_filter(a2d, sigma=sigma) #gaussian_filter(a2d, sigma=sigma)uniform_filter
        M += a2d.astype(np.float32)
 
    return M, ent



def get_disjoint_segments(attn_layers, P,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None, grad = None):
    ent  = 0
    filtered_mask = None
   
    if insert_mask is None and grad is None:
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        print("attn_last_to_vis: ",attn_last_to_vis.shape)
        selected = analyze_heads( attn_last_to_vis.detach().cpu())
        print("selected: ",len(selected))

        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=2.0)
    

        #combo = uniform_filter(combo, size=3) 
        
        
        mask_grid = binarize_mean_relu(combo, ent)
 
        if DO_PLOT:
            plt.imsave("/cluster/scratch/mgroepl/debug/attentionMask.png", mask_grid)
            plt.imsave("/cluster/scratch/mgroepl/debug/attention.png", combo)
    elif grad is not None:
        grad_orig = grad.reshape(24,24).detach().cpu().to(torch.float32).numpy()
        #grad_orig[grad_orig <grad_orig.mean()] = 0.0

        if DO_PLOT:
            tensor_np = grad_orig  # move to CPU if on GPU

            # Plot the tensor
            plt.figure(figsize=(6,6))
            plt.imshow(tensor_np, cmap='viridis')  # 'viridis', 'gray', 'plasma', etc.
            plt.colorbar()  # optional: shows scale
            plt.axis('off')  # optional: hide axes

            # Save as image
            plt.savefig(f"/cluster/scratch/mgroepl/debug/disjointpreSmoothing.png", bbox_inches='tight', pad_inches=0)
            plt.close()


        combo = grad_orig



        temperature = 0.1
        grad_orig_flat2 = grad.clone()
        #grad_orig_flat2 = F.softmax(grad_orig_flat2 / temperature, dim=-1)
        grad_orig = grad_orig_flat2.reshape(24,24).detach().cpu().to(torch.float32).numpy()
    
        

        top_percentile = 99
        high_thresh = grad_orig.max()*0.01

        # Binary mask of high activations

        grad_orig =  gaussian_filter(grad_orig, sigma=1.0)
        if DO_PLOT:
            tensor_np = grad_orig  # move to CPU if on GPU

            # Plot the tensor
            plt.figure(figsize=(6,6))
            plt.imshow(tensor_np, cmap='viridis')  # 'viridis', 'gray', 'plasma', etc.
            plt.colorbar()  # optional: shows scale
            plt.axis('off')  # optional: hide axes

            # Save as image
            plt.savefig(f"/cluster/scratch/mgroepl/debug/disjoint.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        grad_mask = grad_orig >grad_orig.max()*0.3   #grad_orig.max()*0.3  #binarize_mean_relu(grad_orig, do_max = True)
        blob_mask = grad_mask.astype(bool)



        mask_grid = blob_mask
    else:
        mask_grid = insert_mask
    if return_single:
        return [mask_grid]#[bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
    H, W = combo.shape
    combo_flat = combo.flatten()

    # Precompute the flattened global indices (0..H*W-1)
    global_indices = torch.arange(H * W)

    # Flatten mask_grid to match
    mask_flat = torch.tensor(mask_grid).flatten()

    labeled_array_flat = torch.tensor(labeled_array).flatten()

    sorted_indices_per_segment = []

    for seg_id in range(1, num_features + 1):

        # Boolean mask for this segment (flattened)
        seg_mask_flat = (labeled_array_flat == seg_id)

        # Global indices of all positions in this segment
        seg_global_idx = global_indices[seg_mask_flat]          # shape: (N,)

        # Their values
        seg_values = combo_flat[seg_mask_flat]                 # shape: (N,)

        # Sort by values (descending)
        sorted_vals, order = torch.sort(torch.tensor(seg_values), descending=True)

        # Reorder global indices
        sorted_global_idx = seg_global_idx[order]

        sorted_indices_per_segment.append(sorted_global_idx)

    return segment_masks, sorted_indices_per_segment

def get_bbox_indices(attn_layers, P,begin_pos_vis_att, vis_len = 576 , do_grid = True, returnBBOX = False):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[3: 8], P=P, sigma=2.0)
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
  

def get_indices_percent(attn_layers, begin_pos_vis_att, vis_len = 576, mode = "selected", topK = 0.9, largest = False, sample = False,attn_mean_all = None, general_att_map = None, width = 1, height = 1,grad = None):

    
    ent = 0
    if mode == "topK":
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

        # Step 2: Squeeze query dimension (it's size 1)
        attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
        attn_mean_all = attn_mean_heads.mean(dim=0) 
        
    elif mode == "selected":
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        attn_mean_heads = attn_last_to_vis.mean(dim=1)  # [32 layers, 1, 576]

        # Step 2: Squeeze query dimension (it's size 1)
        attn_mean_heads = attn_mean_heads.squeeze(1)  # [32 layers, 576]
        selected = analyze_heads( attn_last_to_vis.detach().cpu())


        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=24, sigma=None)
        
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
    elif mode == "grad":
        attn_mean_all = grad
    top_k = int(topK * attn_mean_all.shape[0]) 
    if sample:
        attn_scores = attn_mean_all.clone()  # don't modify the original
        attn_scores = attn_scores - attn_scores.min()  # optional: make all scores non-negative
        prob = attn_scores / attn_scores.sum()
        sampled_indices = torch.multinomial(prob, num_samples=top_k, replacement=False)
        return sampled_indices, ent
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
        return top_indices, ent

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

    with torch.inference_mode():


        att_map = model(                input_ids=input_ids.to("cuda"),
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes, output_attentions=True)['attentions'][LAYER]
        return att_map


        if input_embeds is not None and num_layer is None:
            
            attn_layers = model(
                #input_ids = input_ids,    to(torch.float32)
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
            device = torch.device("cuda:0")  # or whichever GPU you want
            attn_outputs = [x.to(device) for x in attn_outputs]
            attn_outputs = torch.stack(attn_outputs).squeeze(1)
            return attn_outputs, hidden_states

        else:
            attn_layers = model(    
                input_ids=input_ids.to("cuda"),
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                output_attentions=True,
                return_dict=True,
            )
        logits = attn_layers.logits
        last_logits = logits[0, -1]  
        probs = torch.softmax(last_logits, dim=-1)
    
        attn_layers = attn_layers.attentions # tuple length L of [B,H,Tq,Tk]
        attn_layers = torch.stack(attn_layers, dim=0)  # [L, B, H, Tq, Tk]
        attn_layers = attn_layers[:, 0] 
        return attn_layers, probs.max()







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



def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
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
    
    data_loader,dataset = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, batch_size = 1)


    avg_pruning = 0
    index = -1
    datas_len = len(data_loader)
    vis_len = 576#1152
    bbox_size = 336

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


        att_map2  = get_attn_layers(model,input_ids, image_tensor,image_sizes, input_embeds = None, num_layer = None, attention_mask = None, position_ids = None)[0, :, -1, begin_pos_vis:begin_pos_vis+vis_len].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(24, 24)


        input_ids_general, _, _ = dataset.__getitem__(index  , bbox = None , P = 24, split = False, n = 4, path = None,  general = True)
        x_flat = input_ids_general.flatten()
     
        begin_pos_vis_general = (x_flat == -200).nonzero(as_tuple=True)[0].item()

        general_att_map  = get_attn_layers(model,input_ids_general.unsqueeze(0), image_tensor,image_sizes, input_embeds = None, num_layer = None, attention_mask = None, position_ids = None)[0, :, -1, begin_pos_vis_general:begin_pos_vis_general+vis_len].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(24, 24)


        att_map = att_map2 / general_att_map

        print("att_map ", att_map.shape)
        bbox = bbox_from_att_image_adaptive(att_map, image_sizes[0], bbox_size)

        image = get_image(dataset, index)
        crop_image = image.crop(bbox)
        if DO_PLOT:
            crop_image.save("/cluster/scratch/mgroepl/debug/test/test.png")

            plt.imshow(att_map, cmap='viridis')  # cmap='gray' for grayscale, you can change this based on your preference
            plt.colorbar()  # Optional, adds a color bar to the side of the image
            plt.close()
            # Save the plot as an image file
            plt.savefig("/cluster/scratch/mgroepl/debug/test/test2.png") 
            plt.imsave("/cluster/scratch/mgroepl/debug/test/Relative_att.png", att_map)
            plt.imsave("/cluster/scratch/mgroepl/debug/test/att_map.png", att_map2)
            plt.imsave("/cluster/scratch/mgroepl/debug/test/general_att_map.png", general_att_map)

        
        _, image_tensor_cropped, image_sizes_cropped = dataset.__getitem__(index  , bbox = None , P = 24, split = False, n = 4, path = None, insert_image = crop_image)


        full_embed,  att_mask, position_ids = get_embedding(model,input_ids,image_tensor,image_sizes)
        full_embed_cropped,  att_mask, position_ids = get_embedding(model,input_ids,image_tensor_cropped.unsqueeze(0),image_sizes_cropped)

        print("full_embed_cropped ",full_embed_cropped.shape)

        full_embed = torch.concat((full_embed[:,:begin_pos_vis + vis_len,:],full_embed_cropped[:,begin_pos_vis:,:]   ), dim = 1)
        print("full_embed ",full_embed.shape)

        outputs = model.generate(
            None,
            attention_mask = None,
            inputs_embeds = full_embed,
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

        text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        print("output 1: ", text)
        if DO_PLOT:
            return

        print(img_id)
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": text,
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
    args = parser.parse_args()

    eval_model(args)
