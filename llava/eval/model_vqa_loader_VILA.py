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

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.stats import multivariate_normal
from torchvision.utils import save_image
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, AutoModel, CLIPImageProcessor
from scipy.spatial.distance import cdist

DO_PLOT = True
METHOD = "attention" #"grad" attention
INDEX = 1
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
    def __init__(self, questions, image_folder,image_processor, processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.processor = processor
        self.model_config = model_config



    def __getitem__(self, index, bbox = None , P = 32, split = False, n = 4, path = None, augmentation = None):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]


    
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



        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]


        def split_image(image):
            width, height = image.size
            # Coordinates for the 4 sub-images (top-left, top-right, bottom-left, bottom-right)
            sub_images = [
                image.crop((0, 0, width // 2, height // 2)),       # Top-left
                image.crop((width // 2, 0, width, height // 2)),  # Top-right
                image.crop((0, height // 2, width // 2, height)), # Bottom-left
                image.crop((width // 2, height // 2, width, height))  # Bottom-right
            ]
            return sub_images


        def process_sub_images(sub_images, image_processor):
            pixel_values_list = []
            for sub_image in sub_images:
                # Process each sub-image to get its pixel values as a tensor
                pixel_values = image_processor(images=sub_image, return_tensors='pt').pixel_values[0]
                print("pixel_values", pixel_values.shape)
                pixel_values_list.append(pixel_values)
            return pixel_values_list



        def stitch_tensors(sub_tensors):
            # Concatenate the tensors: top half (top-left + top-right) and bottom half (bottom-left + bottom-right)
            top = torch.cat((sub_tensors[0], sub_tensors[1]), dim=2)  # Concatenate top-left and top-right along width
            bottom = torch.cat((sub_tensors[2], sub_tensors[3]), dim=2)  # Concatenate bottom-left and bottom-right along width
            full_image = torch.cat((top, bottom), dim=1)  # Concatenate top and bottom along height
            return full_image

        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        

        sub_images = split_image(image)  # Split the image into 4 sub-images

        # Assuming you have self.image_processor set up
        pixel_values_list = process_sub_images(sub_images, self.image_processor)
        stitched_image_tensor = stitch_tensors(pixel_values_list)


        #pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        return stitched_image_tensor, input_ids, attention_mask



    def __len__(self):
        return len(self.questions)



def collate_fn(batch):
    pixel_values, input_ids, attention_mask = zip(*batch)
    pixel_values = torch.stack(pixel_values, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return pixel_values, input_ids, attention_mask


# DataLoader
def create_data_loader(questions, image_folder,image_processor, processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder,image_processor, processor, model_config)
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
    P = 32 #int(meta.get("patch_size", int(np.sqrt(V))))

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

def plot_mask(img, top_indices, vis_len, question_id, question):
    P = int(np.sqrt(vis_len))
    
    mask = torch.zeros(vis_len, dtype=torch.bool, device = "cuda")
    mask[top_indices] = True
    
    mask = mask.reshape((P,P))

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
    plt.savefig(f"/cluster/scratch/mgroepl/debug/{question_id}Mask.jpg", dpi=300)
    plt.close()

def get_size_object(attn_layers, P,begin_pos_vis_att,vis_len):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len].cpu()
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[: 13], P=P, sigma=2.0)
    
    return   combo.var(), combo.mean(), ent

def get_disjoint_segments(attn_layers, P,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None, grad = None):
    ent  = 0
    filtered_mask = None
    if insert_mask is None and grad is None:
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        selected = analyze_heads( attn_last_to_vis.detach().cpu())
    

        combo, ent = combine_heads(attn_last_to_vis, selected[:3 ], P=P, sigma=2.0)
        print(ent)

        if False: #ent > 3.0:
            combo, ent = combine_heads(attn_last_to_vis, selected[: -1], P=P, sigma=2.0)
        #combo = uniform_filter(combo, size=3) 
        
        
        mask_grid = binarize_mean_relu(combo, ent)
 
        if DO_PLOT:
            plt.imsave("/cluster/scratch/mgroepl/debug/attentionMask.png", mask_grid)
            plt.imsave("/cluster/scratch/mgroepl/debug/attention.png", combo)
    elif grad is not None:
        grad_orig = grad.reshape(32,32)#.detach().cpu().to(torch.float32).numpy()
        combo = grad_orig






        grad_orig =  gaussian_filter(grad_orig, sigma=1.0)
        grad_orig_filtered = grad_orig.copy()
        #grad_orig_filtered[grad_orig > threshold] = gaussian_filter(grad_orig[grad_orig > threshold], sigma=1.5)
        grad_mask = grad_orig_filtered >grad_orig_filtered.mean() #grad_orig.max()*0.3  #binarize_mean_relu(grad_orig, do_max = True)
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


        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=32, sigma=None)
        
        newAtt = torch.zeros((attn_layers.shape[2],attn_layers.shape[3]), dtype = float, device = "cuda")
        print(newAtt.shape)
        for x in selected:
            l = x["layer"]
            h = x["head"]
            newAtt += attn_layers[l,h,:,:]
        
        attn_mean_all = newAtt[-1, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    elif mode == "general":
 

        attn_pic = attn_layers[14,:,0,begin_pos_vis_att:begin_pos_vis_att + vis_len].mean(0).reshape(32,32) / general_att_map
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
   


def get_embedding(model,input_ids,pixel_values):

   
    embeds = model.model.get_embed(input_ids  = input_ids.unsqueeze(0), pixel_values = pixel_values.unsqueeze(0) )

    return embeds





def calc_grad(model, input_embeds, attention_mask=None):
    # Ensure model params do not store grad
    model.requires_grad_(False)
    
    # Only input embedding requires grad
    input_embeds = input_embeds.detach().clone().requires_grad_(True)

    # Forward pass under no_grad for model params,
    # but allow grad for input_embeds
    print("input_embeds ", input_embeds.shape)
    with torch.set_grad_enabled(True):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,   # <-- saves ton of memory
            return_dict=True
        )

        logits = outputs.logits

        # Compute objective
        temperature = 1.0   # higher = flatter distribution
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        #print("probs ", probs.shape)
        #probs = probs[probs>0.3]
        indices = torch.where(probs > 0.1)[0]

        # Renormalize so everything sums to 1 again
        probs = probs / probs.sum()
        #objective = torch.sum(probs * torch.log(probs))

        uniform = torch.ones_like(probs) / probs.size(-1)
        #objective = torch.sum(probs * torch.log(probs / uniform), dim=-1)

        # Assuming `probs` is your tensor
        top_values, top_indices = torch.topk(probs, k=1, dim=-1)

        # top_values will contain the top 3 values
        #first, second, third = top_values[0], top_values[1], top_values[2]

        all = []
        total = 0
        for objective in top_values:
            total += objective
            if total >0.8 and len(all) > 1:
                continue
            objective.backward(retain_graph=True)  # retain_graph=True if you want to keep the graph for future passes
            #print("Enthropy: ", objective * -1)

            # Grab gradients
            grads = input_embeds.grad.detach().clone()

            # Zero the gradients for the next backprop pass
            input_embeds.grad.zero_()

            all.append(grads)


        # Grab gradients
        

    # memory cleanup
    del outputs, objective
    torch.cuda.empty_cache()

    return all

def get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = None,num_layer = None, attention_mask = None,position_ids = None ):

    with torch.inference_mode():

        if input_embeds is not None and num_layer is None:
           
            
            attn_layers = model(

                inputs_embeds = input_embeds,

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
            return attn_outputs, first_state

        else:
            attn_layers = model(    
                input_ids=input_ids,
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
        print("attn_layers ", attn_layers.shape)
        return attn_layers, probs.max()


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



def get_general(tokenizer, model, image_tensor, image_sizes, begin_pos_vis,vis_len ):
        
    qs = "Can you explain the image to me?"
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_general = conv.get_prompt()
    
    
    input_ids_general = tokenizer_image_token(prompt_general, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    orig_embeds_general  , att_mask_general , position_ids_general = get_embedding(model,input_ids_general ,image_tensor,image_sizes)
    general_att_map, hidden   = get_attn_layers(model,input_ids_general,image_tensor, image_sizes, input_embeds = orig_embeds_general, num_layer = 15, attention_mask = att_mask_general,position_ids = position_ids_general )
    general_att_map = general_att_map[14,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0).reshape(24,24)
    return general_att_map



def get_zoomed_embed(model, index, tokenizer, image_sizes, begin_pos_vis, vis_len, old_box, dataset,input_ids, P = 24):


    line = dataset.questions[index]
    image_file = line["image"]
    image = Image.open(os.path.join(dataset.image_folder, image_file)).convert('RGB')


    a,b,width,height = old_box

    W = image.size[0]/P
    H = image.size[1]/P
    
    x_min = int(a * W)
    x_max = int(x_min + width *W)
    y_min = int(b * H)
    y_max = int(y_min + height *H)
    box_resized_orig = (x_min, y_min, x_max, y_max)
    image = image.crop(box_resized_orig)

    _, image_tensor, image_sizes = dataset.__getitem__(index  , bbox = old_box , P = 24, split = False, n = 4, path = None, augmentation = None)
    general_att_map = get_general(tokenizer, model, image_tensor, image_sizes, begin_pos_vis,vis_len )


    attn, first_hidden  = get_attn_layers(model,input_ids,image_tensor, image_sizes ).cuda()

    attn_pic = attn[14,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0).reshape(24,24) / general_att_map
    attn_pic = attn_pic.cpu().numpy()
    attn_pic = gaussian_filter(attn_pic.astype(np.float32), sigma=2.0)


    m = attn_pic.mean()


    B = np.maximum(attn_pic - m, 0.0)
    attn_pic =  (B > 0).astype(np.uint8)

    boxes = get_disjoint_segments(attn, 24,begin_pos_vis, vis_len = 576 ,insert_mask = None, return_single= True)
    for box in boxes:
        
        a,b,width,height = box
        x_min = int(a * W)
        x_max = int(x_min + width *W)
        y_min = int(b * H)
        y_max = int(y_min + height *H)
        box_resized = (x_min, y_min, x_max, y_max)
        image_cropped = image.crop(box_resized_orig)
        image_tensor = process_images([image_cropped], dataset.image_processor, dataset.model_config)[0]
        image_size = image_cropped.size
        emb_vis, att_mask, position_ids = get_embedding(model,input_ids,image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_size])
      
        return emb_vis, att_mask, position_ids

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


def pool_down(img, target_size):
    m,n = target_size
    pool = nn.AdaptiveAvgPool2d((m, n))
    return pool(img)

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

        attn , first_hidden_orig = get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = orig_embeds, num_layer = 15, attention_mask = att_mask,position_ids = position_ids )

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

def calc_grad_image(model, input_ids, image_tensor,image_sizes,begin_pos_vis, vis_len = 576):




    with torch.enable_grad():





        image_tensor = image_tensor.detach().clone().requires_grad_(True)

        # Zero gradients safely
        model.zero_grad(set_to_none=True)



    


        vision_model = model.get_model().get_vision_tower().vision_tower   # the real CLIP ViT

        # vision_model.forward DOES NOT have no_grad() inside
        image_features = vision_model(image_tensor[0], output_hidden_states=True)  # <-- THIS IS DIFFERENT
        print("feats: ", image_features.last_hidden_state.shape)
        



        #image_features = model.get_model().get_vision_tower()([image_tensor.squeeze(0).squeeze(0)])
        
        image_features = image_features[0][0].unsqueeze(0)
    
        projector = model.get_model().mm_projector  
        embeds = projector(image_features.half())
     
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
            image_tensor.detach(),
            image_sizes = image_sizes,
            #new_pos = new_pos,
            #orig_pos = orig_pos
        )
        print("embs: ", inputs_embeds.shape,embeds.shape )
        device = inputs_embeds.device
        embeds = embeds.to(device)
        embeds = embeds[:,1:,:]
        embeds.retain_grad() 
        inputs_embeds = torch.cat([
            inputs_embeds[:, :begin_pos_vis, :],
            embeds,                 # (1, vis_len, hidden_dim)
            inputs_embeds[:, begin_pos_vis + vis_len:, :]
        ], dim=1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            image_sizes=None,
            output_attentions=True,
            return_dict=True,
        )
      

        # Compute objective
        logits = outputs.logits
   
        #objective = torch.logsumexp(logits[0, -1], dim=-1)
        probs = F.softmax(logits[0,-1], dim = -1)
        objective = torch.sum(probs * torch.log(probs + 1e-9))
        # Backward
        objective.backward()
        
        # Copy the gradients
        grads = image_tensor.grad.clone().abs()

        # Free memory
        del outputs
        del objective
        torch.cuda.empty_cache()
        print("gradis: ", grads.shape)
        return grads



def eval_model(args):

    model_name = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
 
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)




    model.requires_grad_(False)   # optional, if you never want model grads
    #model.gradient_checkpointing_enable()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")



    with open("/cluster/scratch/mgroepl/data/textvqa/val.json", "r") as f:
        gts = json.load(f)["data"] #answers
    
    data_loader,dataset = create_data_loader(questions, args.image_folder,image_processor, processor, model.config,  batch_size = 1)


    avg_pruning = 0
    index = -1
    datas_len = len(data_loader)
    vis_len = 1024 #256


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




    for (pixel_values, input_ids, attention_mask), line in tqdm(zip(data_loader, questions), total=len(questions)):



        pixel_values = pixel_values.to("cuda")[0]
        input_ids = input_ids.to("cuda")[0][0]
        attention_mask = attention_mask.to("cuda")[0]
        patches_amnt = pixel_values.shape[0]
        print(input_ids.shape)
        print(pixel_values.shape)
        print(attention_mask.shape)
        value_to_keep = 151671


        # Step 1: Find all indices where the value is equal to `value_to_keep`
        indices_of_value = (input_ids == value_to_keep).nonzero().squeeze()
        print(indices_of_value[0])
        begin_pos_vis = indices_of_value[0].item()
        end_pos_vis = indices_of_value[-1].item() +1

        input_ids = torch.cat((input_ids[:begin_pos_vis],input_ids[begin_pos_vis : begin_pos_vis + vis_len] , input_ids[end_pos_vis :] ))

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
        print(question)
        #gt_answer = gts[index]["answers"]




        print(img_id)
        idx = line["question_id"]
        cur_prompt = line["text"]
        print("full shape",pixel_values.shape)

        #delete_images_in_folder("/cluster/scratch/mgroepl/heatmaps/mean/")
     


        if True:#with torch.inference_mode():


 


            orig_embeds = get_embedding(model,input_ids,pixel_values)

         
               
                
            #grad_orig = calc_grad(model,orig_embeds, attention_mask = att_mask)[0,begin_pos_vis:begin_pos_vis + vis_len,:].mean(-1)


            if METHOD == "grad":
                grad_all = calc_grad(model, orig_embeds, attention_mask=None)
                total = np.zeros((16,16), dtype = float)
                for ind, grad_or in enumerate(grad_all):
                    grad = torch.norm(
                        grad_or[
                            0, begin_pos_vis:begin_pos_vis + vis_len, :
                        ],
                        p=2,           # e.g., p=1 (L1), p=2 (L2), p='inf'
                        dim=-1
                    )
                #embeds = model.get_input_embeddings()(inputs.input_ids[0])



                    grad_reshaped = grad.reshape(16,16)


                    tensor_float32 = grad_reshaped.to(torch.float32)  # Convert to float32
                    tensor_np = tensor_float32.cpu().numpy() # move to CPU if on GPU
                    total += tensor_np     

                    if DO_PLOT:
                        plt.figure(figsize=(6,6))
                        plt.imshow(tensor_np, cmap='viridis')  # 'viridis', 'gray', 'plasma', etc.
                        plt.colorbar()  # optional: shows scale
                        plt.axis('off')  # optional: hide axes

                        # Save as image
                        plt.savefig(f"/cluster/scratch/mgroepl/debug/tensor_imageintern{ind}.png", bbox_inches='tight', pad_inches=0)
                        plt.close()
                grad_orig_flat = total
                if DO_PLOT:
                    plt.figure(figsize=(6,6))
                    plt.imshow(grad_orig_flat, cmap='viridis')  # 'viridis', 'gray', 'plasma', etc.
                    plt.colorbar()  # optional: shows scale
                    plt.axis('off')  # optional: hide axes

                    # Save as image
                    plt.savefig(f"/cluster/scratch/mgroepl/debug/tensor_imageinternTOTAL.png", bbox_inches='tight', pad_inches=0)
                    plt.close()
            else:
                grad_orig_flat = None

            if True:
                with torch.no_grad():
                    generated_ids = model.generate(input_ids = input_ids.unsqueeze(0) , inputs_embeds = orig_embeds, max_new_tokens=200,
                                                    do_sample=True if args.temperature > 0 else False,
                                                    temperature=args.temperature,
                                                    top_p=args.top_p,
                                                    num_beams=args.num_beams,)
                
                input_len = len(input_ids)
                text = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)

                print("output 1: ", text)
                if args.inject == False:
                    ans_id = shortuuid.uuid()
                    
                    ans_file.write(json.dumps({"question_id": idx,
                                            "prompt": cur_prompt,
                                            "text": text[0],
                                            "answer_id": ans_id,
                                            "model_id": "InterVL",
                                            "metadata": {},
                                            "box_size": box_dataset_orig_size }) + "\n",
                                            )
                    continue

            if METHOD == "attention":
                attn_orig , first_hidden_orig = get_attn_layers(model,input_ids,None, None, input_embeds = orig_embeds, num_layer = None, attention_mask = None,position_ids = None )
            else:
                attn_orig = None

            boxes, sorted_order = get_disjoint_segments(attn_orig, 32,begin_pos_vis, vis_len = vis_len, return_single = False ,insert_mask = None, grad = grad_orig_flat)

    
            for ind, m in enumerate(boxes):
                
                ord = sorted_order[ind]
                ord = ord[:1]
                ord, ordindices = torch.sort(ord, descending=False)
                box = list(bbox_from_mask(m))
                bsize = box[2] * box[3]

                pixel_values2, input_ids2, attention_mask2 = dataset.__getitem__(index  , bbox = box , P = 32, split = False, n = 4, path = None, augmentation = None)  #f"{output_dir}/cut.png"

                pixel_values2 = pixel_values2.to("cuda")
                input_ids2 = input_ids2.to("cuda")
                attention_mask2 = attention_mask2.to("cuda")

                indices_orig =torch.tensor( box_to_indices(box, 32)  )
                
                flattened = torch.flatten(torch.tensor(m))
                
                #indices_orig= ord #torch.where(flattened)[0]

                emb_vis = get_embedding(model,input_ids,pixel_values2)
             
                prune_type = "topK"

                if prune_type != "grad":
                    attn, logits  = get_attn_layers(model,input_ids,None, None, input_embeds = emb_vis, num_layer = None, attention_mask = None,position_ids = None )
                    grad = None
                else:
                    attn = None
                
                    grad = torch.norm(
                        calc_grad(model, emb_vis, attention_mask=None)[0][
                            0, begin_pos_vis:begin_pos_vis + vis_len, :
                        ],
                        p=1,           # e.g., p=1 (L1), p=2 (L2), p='inf'
                        dim=-1
                    )
                    torch.tensor(grad) 
    

                if DO_PLOT:
                    im = get_image(dataset, index, box = box, P = 32)
                    im.save(f"/cluster/scratch/mgroepl/debug/{ind}.png")
                    


                
                #print("debug: ", bsize, indices_orig.shape[0], box[2],box[3])
                indices, ent = get_indices_percent(attn, begin_pos_vis, vis_len = vis_len, mode = prune_type, topK = len(indices_orig) / (32*32), largest = True, width = box[2] , height =box[3] , sample = False,grad = grad ) #bsize / (24*24)
  

                orig_embeds_vis_cut = prune_indices(emb_vis, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)[:,begin_pos_vis:begin_pos_vis + len(indices_orig) ,:]                   
                print(orig_embeds_vis_cut.shape)
               
                orig_embeds[:,indices_orig + begin_pos_vis,:] = orig_embeds_vis_cut.clone()#emb_vis[:,begin_pos_vis + indices]#

       
            

            #orig_embeds_pruned = prune_indices(orig_embeds, indices_grad,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True) 
            if DO_PRUNE: 
                
                grad_or = calc_grad(model, orig_embeds, attention_mask=att_mask)
                grad = torch.norm(
                    grad_or[
                        0, begin_pos_vis:begin_pos_vis + vis_len, :
                    ],
                    p=2,           # e.g., p=1 (L1), p=2 (L2), p='inf'
                    dim=-1
                )
        


                # Create mask
                mask = grad >= grad.max()*0.1


                flattened = torch.flatten(mask)
                
                indices= torch.where(flattened)[0]


                #attn,_  = get_attn_layers(model,input_ids, image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),None, input_embeds = orig_embeds, num_layer = 32, attention_mask = att_mask, position_ids = position_ids)
                indices, ent = get_indices_percent(None, begin_pos_vis, vis_len = 576, mode = "selected", topK = 5/(32*32), largest = True , grad =grad )
                orig_embeds_pruned = prune_indices(orig_embeds, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)


                avg_pruning +=orig_embeds.shape[1] -  orig_embeds_pruned.shape[1]
                
                print("new: ", orig_embeds_pruned.shape)
                print("pruned: ", orig_embeds.shape[1] -  orig_embeds_pruned.shape[1])
            print("old: ", orig_embeds.shape)
            #orig_embeds[:,begin_pos_vis:begin_pos_vis + vis_len, :] += grad_or[:,begin_pos_vis:begin_pos_vis + vis_len, :]* 1.5
            with torch.no_grad():
                generated_ids = model.generate(input_ids = input_ids.unsqueeze(0) , inputs_embeds = orig_embeds, max_new_tokens=200,
                                                do_sample=True if args.temperature > 0 else False,
                                                temperature=args.temperature,
                                                top_p=args.top_p,
                                                num_beams=args.num_beams,)
            input_len = len(input_ids)
            text2 = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)

            print("text2:",text2)     
            
            if DO_PLOT and DO_PRUNE:
                grad_orig_flat = grad
                grad_orig = grad_orig_flat.reshape(16,16).detach().cpu().to(torch.float32).numpy()
                
                #grad_orig =  gaussian_filter(grad_orig, sigma=2.0)
                grad_mask = binarize_mean_relu(grad_orig, do_max = True)
                flattened = torch.flatten(torch.tensor(grad_mask))
                
                indices_grad= torch.where(flattened)[0]
                grad_reshaped = grad_orig.reshape(16,16)
                image = get_image(dataset, index, box = None, P = 16)
                image.save(f"/cluster/scratch/mgroepl/debug/Orig.png")




                im = get_image(dataset, index, box = None, P = 16)

                plot_mask(im, indices_grad, vis_len, f"Orign", question)


                tensor_np = grad_reshaped  # move to CPU if on GPU

                # Plot the tensor
                plt.figure(figsize=(6,6))
                plt.imshow(tensor_np, cmap='viridis')  # 'viridis', 'gray', 'plasma', etc.
                plt.colorbar()  # optional: shows scale
                plt.axis('off')  # optional: hide axes

                # Save as image
                plt.savefig(f"/cluster/scratch/mgroepl/debug/grad2.png", bbox_inches='tight', pad_inches=0)
                plt.close()
            if DO_PLOT:
                return
        print("box_dataset_orig_size: ", box_dataset_orig_size)
        print("output:",text2[0] )
        print(img_id)
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": text2[0],
                                   "answer_id": ans_id,
                                   "model_id": "InterVL",
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
    parser.add_argument("--inject", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
