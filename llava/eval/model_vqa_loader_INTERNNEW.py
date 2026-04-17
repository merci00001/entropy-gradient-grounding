import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from datasets import load_dataset
import re
import random
from scipy.ndimage import label
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, AutoModel, CLIPImageProcessor

#from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
#from llava.conversation import conv_templates, SeparatorStyle
#from llava.model.builder import load_pretrained_model
#from llava.utils import disable_torch_init
#from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.stats import multivariate_normal
from torchvision.utils import save_image
import inspect
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import get_optimal_tiled_canvas
from scipy.spatial.distance import cdist
DO_PLOT = True
DO_VANILLA = False
METHOD = "grad" #"grad" attention
INDEX = 12#30

LOAD_DATA = False
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]




def total_vram_allocated():
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))

def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, model, processor, model_config):
        self.image_folder = image_folder
        self.questions = questions
        self.model = model
        self.processor = processor

        self.model_config = model_config



    def __getitem__(self, index, bbox = None , P = (24,24), split = False, n = 4, path = None, augmentation = None, insert_image = None):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if insert_image is None:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        else:
            image = insert_image

        if bbox is not None:

            a, b, width, height = bbox

            # canvas size in grad cells
            canvas_W = P[1] *   16  # e.g. 3 * 16 = 48
            canvas_H = P[0] * 16  # e.g. 2 * 16 = 32

            scale_x = image.size[0] / canvas_W
            scale_y = image.size[1] / canvas_H

            x_min = int(a * scale_x)
            x_max = int((a + width) * scale_x)
            y_min = int(b * scale_y)
            y_max = int((b + height) * scale_y)

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
    

        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

        return inputs, image

    def __len__(self):
        return len(self.questions)



def collate_fn(batch):
    input_ids, image = zip(*batch)
    #input_ids = torch.stack(input_ids, dim=0)
    
    return input_ids, image


# DataLoader
def create_data_loader(questions, image_folder, model, processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, model, processor, model_config)
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




def measure_attention_vram_forward(model, input_embeds, attention_mask=None):
    model.requires_grad_(False)
    model.config._attn_implementation = "eager"
    sync_all()
    vram_before = total_vram_allocated()

    with torch.no_grad():
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,  # materialize attention maps
            return_dict=True
        )

    sync_all()
    vram_after = total_vram_allocated()

    # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
    total_bytes = sum(a.element_size() * a.nelement() for a in outputs.attentions)
    num_layers = len(outputs.attentions)
    print(f"outputs.attentions length: {len(outputs.attentions)}")
    print(f"output keys: {outputs.keys()}")
    print(f"\n--- Attention Map VRAM ---")
    print(f"Layer 0  shape : {tuple(outputs.attentions[0].shape)}")
    print(f"Layer -1 shape : {tuple(outputs.attentions[-1].shape)}")
    print(f"Num layers     : {num_layers}")
    print(f"Attention VRAM : {total_bytes / 1024**2:.2f} MB  ({total_bytes / 1024**3:.3f} GB)")
    print(f"\nVRAM before    : {vram_before / 1024**3:.3f} GB")
    print(f"VRAM after     : {vram_after  / 1024**3:.3f} GB")
    print(f"Delta          : {(vram_after - vram_before) / 1024**3:.3f} GB")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i} allocated : {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"GPU {i} reserved  : {torch.cuda.memory_reserved(i)  / 1024**3:.2f} GB")

    del outputs
    torch.cuda.empty_cache()



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
    B = S #torch.relu(S - mean_val*2)
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
    M = np.zeros((P[0], P[1]), dtype=np.float32)
    ent = 0
    for item in selected:
        l, h = item["layer"], item["head"]
        ent += item["spatial_entropy"]
        a2d = attn[l, h, 0].reshape(P[0], P[1]).detach().cpu().to(torch.float32).numpy()
        if sigma and sigma > 0:
            a2d =  gaussian_filter(a2d, sigma=sigma) #gaussian_filter(a2d, sigma=sigma)uniform_filter
        M += a2d.astype(np.float32)
 
    return M, ent


def plot_mask(img, top_indices, vis_len, question_id, question, W, H):
    P = int(np.sqrt(vis_len))
    
    mask = torch.zeros(vis_len, dtype=torch.bool, device = "cuda")
    mask[top_indices] = True
    
    mask = mask.reshape((W,H))

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


def get_disjoint_segments(attn_layers, P,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None, grad = None):
    ent  = 0
    filtered_mask = None
    if insert_mask is None and grad is None:
        attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
        selected = analyze_heads( attn_last_to_vis.detach().cpu())
    

        combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=2.0)
        print(ent)

        if False: #ent > 3.0:
            combo, ent = combine_heads(attn_last_to_vis, selected[: -1], P=P, sigma=2.0)
        #combo = uniform_filter(combo, size=3) 
        
        
        mask_grid = binarize_mean_relu(combo, ent)
 
        if DO_PLOT:
            plt.imsave("/cluster/scratch/mgroepl/debug/attentionMask.png", mask_grid)
            plt.imsave("/cluster/scratch/mgroepl/debug/attention.png", combo)
    elif grad is not None:
    
        
        temperature = 0.01
        grad_orig_flat2 = grad.clone()
        #grad_orig_flat2 = F.softmax(grad_orig_flat2 / temperature, dim=-1)
        grad_orig = grad_orig_flat2.reshape(P[0], P[1]).detach().cpu().to(torch.float32).numpy()
        combo = grad_orig
        

        top_percentile = 99

        grad_orig =  gaussian_filter(grad_orig, sigma=1.5)
        grad_orig2 = grad_orig.copy()
        
        el = elbow_chord(grad_orig.flatten())  #   elbow_chord(grad_orig_filtered.flatten())
        #grad_orig_filtered[grad_orig > threshold] = gaussian_filter(grad_orig[grad_orig > threshold], sigma=1.5)
        grad_mask = grad_orig2 > el #grad_orig.max() * 0.5  #binarize_mean_relu(grad_orig, do_max = True)

        ent = spatial_entropy(torch.tensor(grad_orig2), el)
        ent = ent["spatial_entropy"]

        # Original blob mask (after your binarization)
        blob_mask = grad_mask.astype(bool)   # shape (24,24), dtype=bool or 0/1

        mask_grid = blob_mask
        if DO_PLOT:
            plt.imsave("/cluster/scratch/mgroepl/debug/attentionMask.png", mask_grid)
            plt.imsave("/cluster/scratch/mgroepl/debug/attention.png", combo)

    else:
        mask_grid = insert_mask
    if return_single:
        return [mask_grid]#[bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
   
    sorted_vals_per_segment = [ combo[b]   for b in segment_masks]

    
    return segment_masks,  sorted_vals_per_segment, ent


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

def box_to_indices(image_shape, bbox):
    """
    Returns the indices of pixels inside a bounding box for a flattened image.

    Parameters:
    - image_shape: tuple of (height, width) of the image
    - bbox: tuple of (xmin, ymin, width, height)

    Returns:
    - indices: 1D numpy array of flattened indices
    """
    H, W = image_shape
    xmin, ymin, width, height = bbox

    # Ensure the box is within image boundaries
    xmax = min(xmin + width, W)
    ymax = min(ymin + height, H)

    # Create grid of coordinates inside bbox
    ys, xs = np.meshgrid(np.arange(ymin, ymax), np.arange(xmin, xmax), indexing='ij')

    # Convert 2D coordinates to flattened indices
    indices = ys * W + xs
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
    elif mode == "random":
        sample_size = int(vis_len * topK)

        # Randomly sample indices (without replacement)
        sampled_indices = random.sample(range(vis_len), sample_size)
        return torch.tensor(sampled_indices), ent
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
   


def get_embedding(model,input_ids,pixel_values,image_grid_thw, vis_only = False):


    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)
    image_embeds = model.model.visual(pixel_values, grid_thw=image_grid_thw)["pooler_output"]
    # shape: [2552, 3584] — already merged and projected ✓
    
    image_mask = (input_ids == model.config.image_token_id)
    print("mask slots:", image_mask.sum().item())  # should be 2552
    inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)
    vis_len = image_mask.sum().item()
    
    return inputs_embeds, vis_len

    inputs_embeds = model.get_input_embeddings()(input_ids)
    count = (input_ids == 151652).sum().item()
    print("count ", count)
    
    image_embeds = model.model.get_image_features(pixel_values, image_grid_thw)["pooler_output"]
    print(image_embeds)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    if vis_only:
        return image_embeds
    print(image_embeds.shape)
    vis_len = image_embeds.shape[0]

    image_mask, _ = model.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds, vis_len




def calc_grad(model, input_embeds, attention_mask=None):
    model.requires_grad_(False)
    input_embeds = input_embeds.detach().clone().requires_grad_(True)

    sync_all()
    vram_before_forward = total_vram_allocated()

    with torch.set_grad_enabled(True):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            return_dict=True
        )
        logits = outputs.logits

        sync_all()
        vram_after_forward = total_vram_allocated()

        temperature = 1.0
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        probs = probs / probs.sum()
        objective = torch.sum(probs * torch.log(probs))

        top_values, top_indices = torch.topk(probs, k=30, dim=-1)
        first, second, third = top_values[0], top_values[1], top_values[2]

        objective.backward()
        print("Enthropy: ", objective * -1)

        sync_all()
        vram_after_backward = total_vram_allocated()

        grads = input_embeds.grad.detach().clone()

    print(f"\n--- Gradient VRAM ---")
    print(f"Before forward        : {vram_before_forward / 1024**3:.3f} GB")
    print(f"After forward         : {vram_after_forward  / 1024**3:.3f} GB")
    print(f"After backward        : {vram_after_backward / 1024**3:.3f} GB")
    print(f"Peak VRAM for backward: {(vram_after_forward  - vram_before_forward) / 1024**3:.3f} GB  (compute graph)")
    print(f"Persistent after bwd  : {(vram_after_backward - vram_before_forward) / 1024**3:.3f} GB  (gradients only)")

    del outputs, objective
    torch.cuda.empty_cache()
    return grads

def get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = None,num_layer = None, attention_mask = None,position_ids = None ):

    with torch.inference_mode():




        if input_embeds is not None and num_layer is None:
            
            attn_layers = model.generate(
                input_ids = input_ids,
                inputs_embeds=input_embeds,
                attention_mask=None,
                output_attentions=True,
               
            )
            print(attn_layers.shape)
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

def fit_gmm_fixed_means_grid(X, Y, P, centers, max_iter=50):
    H, W = P.shape
    K = len(centers)

    means = np.array(centers)
    weights = np.ones(K) / K
    covs = np.array([np.eye(2) for _ in range(K)])  # initial guess

    XY = np.stack([X, Y], axis=-1)   # shape (H, W, 2)

    for _ in range(max_iter):
        # E-step
        pdfs = np.zeros((H, W, K))
        pt = XY.reshape(-1,2)

        for k in range(K):
            pdfs[:,:,k] = weights[k] * multivariate_normal.pdf(
                pt, mean=means[k], cov=covs[k]
            ).reshape(H, W)

        # normalize responsibilities
        denom = pdfs.sum(axis=2, keepdims=True) + 1e-12
        R = pdfs / denom

        # M-step
        Nk = (P[...,None] * R).sum(axis=(0,1))
        weights = Nk

        # covariance update
        for k in range(K):
            diff = XY - means[k]
            outer = diff[..., :, None] * diff[..., None, :]     # (24,24,2,2)
            weighted_outer = P[..., None, None] * R[..., k][..., None, None] * outer
            covs[k] = weighted_outer.sum(axis=(0,1)) / Nk[k]

    return weights, covs



def forward_with_images(model, processor, question, images):
    # images: list of PIL images
    
    # Build content list: one image block per image, then the question
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info (handles image resizing, patching, etc.)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    return inputs

def eval_model(args):

 
    #disable_torch_init()
    model_name = "OpenGVLab/InternVL3_5-8B-HF"
    path = model_name
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,

        trust_remote_code=True).eval().cuda()




    processor = AutoProcessor.from_pretrained(path)

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
    
    data_loader,dataset = create_data_loader(questions, args.image_folder, model, processor, model.config, batch_size = 1)


    avg_pruning = 0
    index = -1
    datas_len = len(data_loader)
    vis_len = 1576#1152


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




    for (inputs, image), line in tqdm(zip(data_loader, questions), total=len(questions)):

        inputs = inputs[0]
        pixel_values = inputs["pixel_values"].to("cuda")
        input_ids = inputs["input_ids"].to("cuda")
        image = image[0]
        ratio = image.size[0] / image.size[1]

        num_columns, num_rows = get_optimal_tiled_canvas(
            (image.size[1], image.size[0]),  # (height, width)
            (448, 448),
            min_image_tiles=processor.image_processor.min_patches,
            max_image_tiles=processor.image_processor.max_patches,
        )

        print(num_columns,num_rows )
        print("ratio", image.size[0] / image.size[1])
        ratio = num_columns / num_rows
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
        print(question)



        print(img_id)
        idx = line["question_id"]
        cur_prompt = line["text"]

       
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        value_to_keep = 151671
        indices_of_value = (input_ids[0] == value_to_keep).nonzero().squeeze()
        begin_pos_vis = indices_of_value[0].item()
        vis_len = len(indices_of_value) - 16*16
        #delete_images_in_folder("/cluster/scratch/mgroepl/heatmaps/mean/")
  


        if True:#with torch.inference_mode():


 

            orig_embeds = model.model.get_embed(input_ids  = input_ids, pixel_values = pixel_values )

            if DO_VANILLA or DO_PLOT:

                inputs = inputs.to("cuda")
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        num_beams=1,
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_texts = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
                print(generated_texts[0])
                text2 = generated_texts[0]
                    # Trim prompt from output
                if DO_VANILLA:
                    print(img_id)
                    ans_id = shortuuid.uuid()
                    
                    ans_file.write(json.dumps({"question_id": idx,
                                            "prompt": cur_prompt,
                                            "text": text2,
                                            "answer_id": ans_id,
                                            "model_id": model_name,
                                            "metadata": {},
                                            "box_size": 0.0 }) + "\n",)

                    continue
                else:
                    orig_answer = text2





            measure_attention_vram_forward(model, orig_embeds, attention_mask=None)
            return
                
            #grad_orig = calc_grad(model,orig_embeds, attention_mask = att_mask)[0,begin_pos_vis:begin_pos_vis + vis_len,:].mean(-1)

            crops = []
            if METHOD == "grad":
                g = calc_grad(model, orig_embeds, attention_mask=None)[
                        0, begin_pos_vis:begin_pos_vis + vis_len, :
                    ] 
                
                grad_orig_flat = torch.norm(
                    g,
                    p=2,           # e.g., p=1 (L1), p=2 (L2), p='inf'
                    dim=-1
                )            
                image = get_image(dataset, index, box = None, P = 24)
                if DO_PLOT:
                    image.save("/cluster/scratch/mgroepl/debug/orig.png")
                crops.append(image)

                num_patches = pixel_values.shape[0] -1 
                print("num_patches ", num_patches)
                print("vise len", vis_len)
                patch_tokens =  256 #vis_len // num_patches
                patch_grid =16 # int(math.sqrt(patch_tokens))  # e.g. 16
                grid_rows = int(math.sqrt(num_patches / ratio))
                grid_cols = int(grid_rows * ratio)# num_patches // grid_rows
                print("grid_rows", grid_rows,"grid_cols",grid_cols )
          


            w_and_h = [(grid_rows,grid_cols)]
            num_patches_total = [num_patches]
            attn_orig = None

            Iteration = -1
            ent_new = 0
            ent_previous = 1000
            images = [image]
            while True:
                Iteration += 1
                box_index = []
                ents = []
                boxes = []
                vals = []
                images_new = [image]
                pointer = 0

                if w_and_h[0][0] == 0:
                    break

                for ind, (w,h) in enumerate(w_and_h):



                    canvas = np.zeros((w * patch_grid, h * patch_grid), dtype=float)
                    print("pointer", pointer, w,h)
                    g_vis = grad_orig_flat[pointer:pointer+w*h*patch_grid*patch_grid]
                    print("grad shape", g_vis.shape)

                    num_patches = num_patches_total[ind]
                    for i in range(num_patches):
                        token_start = i * patch_tokens
                        token_end = token_start + patch_tokens
                        grad_patch = g_vis[token_start:token_end]
                        grad_2d = grad_patch.reshape(patch_grid, patch_grid).to(torch.float32).cpu().numpy()
                        
                        row = i // grid_cols
                        col = i % grid_cols
                        canvas[row*patch_grid:(row+1)*patch_grid, col*patch_grid:(col+1)*patch_grid] = grad_2d
                    grad_orig_flat2 = torch.tensor(canvas, dtype=torch.float32).reshape(-1)


                    b, sorted_order, ent = get_disjoint_segments(attn_orig, (w * patch_grid, h * patch_grid),begin_pos_vis, vis_len = vis_len, return_single = False ,insert_mask = None, grad = grad_orig_flat2)
                    

                    if DO_PLOT:
                        grad_orig = grad_orig_flat2.reshape(w * patch_grid, h * patch_grid).detach().cpu().to(torch.float32).numpy()
                        tensor_np = grad_orig  # move to CPU if on GPU
                        plt.figure(figsize=(6,6))
                        plt.imshow(canvas, cmap='viridis')
                        plt.axis('off')
                        plt.savefig(f"/cluster/scratch/mgroepl/debug/{Iteration}grad.png", bbox_inches='tight', pad_inches=0)
                        plt.close()



                    for x in range(len(b)):
                        ents.append(ent)
                        #print("box index ", ind)
                        box_index.append(ind)
                        boxes.append(b[x])
                        vals.append(sorted_order[x])
                    pointer += w*h*patch_grid*patch_grid

                vals_summed = [s.sum() for s in vals]
                to_try = 3
                if len(boxes) == 0:
                    break


                try:
                    boxes = [x for _, x in sorted(zip(vals_summed, boxes), reverse=True)]
                    boxes = boxes[: min(len(boxes),to_try)]
                except:
                    break
                box_index = [x for _, x in sorted(zip(vals_summed, box_index), reverse=True)]
                box_index = box_index[: min(len(box_index),to_try)]

                ents = [x for _, x in sorted(zip(vals_summed, ents), reverse=True)]
                ents = ents[: min(len(ents),to_try)]



                #tokens_total_new = [(token_width,token_height)]

                if ents[0] >= ent_previous:
                    print("ENTHS", ents[0])
                    break
                else:
                    ent_previous = ents[0]
            
                for ind, m in enumerate(boxes):
                    

                    print( "entropy box :", ents[ind])
                    box = list(bbox_from_mask(m))
                    print("box", box)
                    bsize = box[2] * box[3]



                    _, image_2= dataset.__getitem__(index  , bbox = box , P = w_and_h[box_index[ind]], split = False, n = 4, path = None, insert_image = images[box_index[ind]])  #f"{output_dir}/cut.png"
                    images_new.append(image_2)
                    if DO_PLOT:
                        image_2.save(f"/cluster/scratch/mgroepl/debug/{ind}Cut.png")




                    continue
                
                images = images_new
                break
                new_vis_total = [orig_embeds[0,:begin_pos_vis]]

                w_and_h = []
                num_patches_total = []
                vis_total = vis_len
                for x in images:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "image", "image": x}] + 
                                [{"type": "text", "text": question}],
                    }]

                    inputs2 = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                    ).to("cuda")


                    input_ids2 = inputs2["input_ids"].to("cuda")
                    pixel_values2 = inputs2["pixel_values"].to("cuda")

                    num_patches = max(1,pixel_values2.shape[0] -1) 
                    num_patches_total.append(num_patches)
                    print("TEST", pixel_values2.shape)

                    new_embs = model.model.get_embed(input_ids  = input_ids2, pixel_values = pixel_values2)
                    print("new_embs",new_embs.shape)
                    value_to_keep = 151671
                    indices_of_value = (input_ids2[0] == value_to_keep).nonzero().squeeze()
                    begin_pos_vis2 = indices_of_value[0].item()
                    vis_len2 = max(16*16,len(indices_of_value) - 16*16)
                    vis_total += vis_len2
                    print("vis len", vis_len2)
                    num_columns2, num_rows2 = get_optimal_tiled_canvas(
                        (x.size[1], x.size[0]),  # (height, width)
                        (448, 448),
                        min_image_tiles=processor.image_processor.min_patches,
                        max_image_tiles=processor.image_processor.max_patches,
                    )

      
                    ratio2 = num_columns2 / num_rows2
                    print("num_columns2",num_columns2,num_rows2)
                    vis_embs2 = new_embs[0,begin_pos_vis2:begin_pos_vis2+vis_len2]
                    new_vis_total.append(vis_embs2)

                    num_patches2 = max(1,pixel_values2.shape[0] -1) 
                    grid_rows2 = int(math.sqrt(num_patches2 / ratio2))
                    grid_cols2 = int(grid_rows2 * ratio2)# num_patches // grid_rows
                    w_and_h.append((grid_rows2,grid_cols2))
                    

                    print("AAAAAAAA ",grid_rows2,grid_cols2,vis_embs2.shape )

                new_vis_total.append(orig_embeds[0,begin_pos_vis+vis_len:,:])

                for u in new_vis_total:
                    print(u.shape)

                orig_embeds2 = torch.cat(new_vis_total,dim = 0).unsqueeze(0)
                print("orig_embeds2",orig_embeds2.shape)



                g = calc_grad(model, orig_embeds2, attention_mask=None)[
                        0, begin_pos_vis:begin_pos_vis + vis_total, :
                    ] 
                
                grad_orig_flat = torch.norm(
                    g,
                    p=2,           # e.g., p=1 (L1), p=2 (L2), p='inf'
                    dim=-1
                )    
                print("grad_orig_flat",grad_orig_flat.shape)


                
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + 
                        [{"type": "text", "text": question}],
            }]

            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_texts = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
            print(generated_texts[0])
            text2 = generated_texts[0] # processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            print("emb text", text2)

    

            if DO_PLOT:
                if orig_answer == text2:
                    continue
                return
        
        print(img_id)
        ans_id = shortuuid.uuid()
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": text2,
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
