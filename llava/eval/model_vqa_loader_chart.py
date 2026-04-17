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
from PIL import Image, ImageDraw
import math
from torch import nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from torchvision.utils import save_image

from groundingdino.util.inference import load_model, load_image, predict, annotate
from io import BytesIO
import base64
import ast
DO_PLOT = False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, box_data, tokenizer, image_processor, model_config):
        self.data = box_data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config



    def __getitem__(self, index, bbox = None , P = 24, split = False, n = 4, path = None, augmentation = None, is_gt = False):
       
  
        qs = self.data[index]["query"]

      
   
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs += "\nAnswer using only a single word or phrase."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        img = self.data[index]["image"]
       
        if img is list:
            byte_data = bytes(img)

        image = Image.open(BytesIO(img))


        

        if bbox is not None:
            if is_gt == False:
                a,b,width,height = bbox

                W = image.size[0]/P 
                H = image.size[1]/P
                
                x_min = int(a * W)
                x_max = int(x_min + width *W) 
                y_min = int(b * H)
                y_max = int(y_min + height *H)
                box_resized = (x_min, y_min, x_max, y_max)
            else:
                box_resized = bbox.copy()
                box_resized [2] = box_resized[2] + box_resized[0]
                box_resized [3] = box_resized[3] + box_resized[1]
            image = image.crop(box_resized)
            if DO_PLOT and is_gt:

                image.save("/cluster/scratch/mgroepl/debug/cropped.jpg")

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')


        if augmentation is not None:
            if augmentation == "rotation":
                image = image.rotate(90, expand=True)

        if path is not None:
            image.save(path)
       
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        if image_tensor.shape[0] != 1:
            image_tensor = image_tensor[0,:,:,:].unsqueeze(0)
        
        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.data)




def get_image(dataset, index, box = None, P = 24):
    
        img = dataset.data[index]["image"]
       
        if img is list:
            byte_data = bytes(img)

        image = Image.open(BytesIO(img))

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


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

# DataLoader
def create_data_loader(box_data, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(box_data, tokenizer, image_processor, model_config)
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
    m = M.mean()  


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

def get_disjoint_segments(attn_layers, P,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=2.0)
    if insert_mask is None:
        mask_grid = binarize_mean_relu(combo, ent)
    else:
        mask_grid = insert_mask


    if DO_PLOT:
        plt.imsave("/cluster/scratch/mgroepl/debug/attentionMask.png", mask_grid)
        plt.imsave("/cluster/scratch/mgroepl/debug/attention.png", combo)

    if return_single:
        return [bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
    boxes = [bbox_from_mask(x) for x in segment_masks]
    
    return segment_masks

def get_bbox_indices(attn_layers, P,begin_pos_vis_att, vis_len = 576 , do_grid = True, returnBBOX = False):

    attn_last_to_vis = attn_layers[:, :, -1:, begin_pos_vis_att:begin_pos_vis_att + vis_len]
    selected = analyze_heads( attn_last_to_vis.detach().cpu())


    combo, ent = combine_heads(attn_last_to_vis, selected[: 3], P=P, sigma=1.0)
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
    x2 = x + w
    y2 =  y + h
    ys, xs = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')

    # Convert (y, x) to 1D indices in a flattened image
    indices = ys * P + xs
  
    return indices.flatten()
  

def get_indices_percent(attn_layers, begin_pos_vis_att, vis_len = 576, mode = "selected", topK = 0.9, largest = False, sample = False,attn_mean_all = None, general_att_map = None):
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




def get_clip_embed(model,image_tensor, orig_pos, new_pos):
    orig = torch.arange(577).cuda()
    new_pos = new_pos.cuda() +1
    orig_pos = orig_pos.cuda()  +1
    #orig[new_pos] = orig_pos 
    #orig[orig_pos] = new_pos 
    projector = model.get_model().mm_projector  
    vision_tower = model.get_vision_tower()
    ClipEmbedder = vision_tower.vision_tower
    #ClipEmbedder.vision_model.embeddings.position_ids = orig.unsqueeze(0).cuda()
   


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
           
            attn_outputs.append(layer_outputs[1][:,:,-1:,:])
        attn_outputs = torch.stack(attn_outputs).squeeze(1)
        return attn_outputs


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



def get_general(tokenizer, model, image_tensor, image_sizes, begin_pos_vis,vis_len ):
        
    qs = "Can you explain the image to me?"
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_general = conv.get_prompt()
    
    
    input_ids_general = tokenizer_image_token(prompt_general, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    orig_embeds_general  , att_mask_general , position_ids_general = get_embedding(model,input_ids_general ,image_tensor,image_sizes)
    general_att_map   = get_attn_layers(model,input_ids_general,image_tensor, image_sizes, input_embeds = orig_embeds_general, num_layer = 15, attention_mask = att_mask_general,position_ids = position_ids_general ).cuda()[14,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0).reshape(24,24)
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


    attn  = get_attn_layers(model,input_ids,image_tensor, image_sizes ).cuda()

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


def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate box edges
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1

    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2

    # If one box is completely to one side of the other → no overlap
    if right1 <= left2 or right2 <= left1:
        return False
    if bottom1 <= top2 or bottom2 <= top1:
        return False

    return True

def eval_model(args):

    model_dino = load_model("/cluster/project/cvg/students/mgroepl/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/cluster/scratch/mgroepl/DINO/groundingdino_swint_ogc.pth")
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = []

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')


    



    
  
    box_data = load_dataset("ahmed-masry/ChartQA")["test"]


    data_loader,dataset = create_data_loader(box_data, tokenizer, image_processor, model.config, batch_size = 1)
    avg_pruning = 0
    index = -1
    index_data = -1
    datas_len = len(data_loader)
    vis_len = 576#1152

    IoUDict = {}
    IoUDict2 = {}

    cut= []
    pos_in = 0
    neg_in = 0
    pos = 0
    neg = 0
    for (input_ids, image_tensor, image_sizes) in tqdm(data_loader, total=len(box_data)):
        index += 1

        if index < -25:
            continue
        question = box_data[index]["query"]
        


        qs = question
        if dataset.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs += "\nThe image youre presented with is a cropped version of the original. It may not contain all relevant information for answering the question. Instead, answer with relevant information contained in the cropped image."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        input_ids_cropped = tokenizer_image_token(prompt, dataset.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids_cropped = input_ids_cropped.unsqueeze(0).cuda()
     
    

        

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        x_flat = input_ids.flatten()
        begin_pos_vis = (x_flat == -200).nonzero(as_tuple=True)[0].item()
        #delete_images_in_folder("/cluster/scratch/mgroepl/heatmaps/mean/")
     


        with torch.inference_mode():





            orig_embeds , att_mask, position_ids= get_embedding(model,input_ids,image_tensor,image_sizes)
            orig_embeds_vis = orig_embeds[0,begin_pos_vis:begin_pos_vis + vis_len,:]
   
            attn  = get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = orig_embeds, num_layer = 25, attention_mask = att_mask,position_ids = position_ids ).cuda()
            if False:
                general_att_map = get_general(tokenizer, model, image_tensor, image_sizes, begin_pos_vis,vis_len )

                attn_pic = attn[14,:,0,begin_pos_vis:begin_pos_vis + vis_len].mean(0).reshape(24,24) / general_att_map
                attn_pic = attn_pic.cpu().numpy()
                attn_pic = gaussian_filter(attn_pic.astype(np.float32), sigma=2.0)
            

                m = attn_pic.mean()


                B = np.maximum(attn_pic - m, 0.0)
                attn_pic =  (B > 0).astype(np.uint8)

                #plt.imsave("/cluster/project/cvg/students/mgroepl/LLaVA/llava/eval/output.png", attn_pic)
                
            

            outputs = model.generate(
                input_ids,
                attention_mask = att_mask,
                #inputs_embeds = inputs_embeds_pruned,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=100,#args.max_new_tokens,
            
                output_hidden_states=True ,
                return_dict_in_generate=True, 
                use_cache=True)
            text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            print(text)


            box, ent = get_bbox_indices(attn, 24,begin_pos_vis, vis_len = 576 , do_grid = False, returnBBOX = True)
            boxes = get_disjoint_segments(attn, 24,begin_pos_vis, vis_len = 576, return_single = False ,insert_mask = None)
            #ind = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = 0.1, largest = True)
            do_clip = False
            for ind, m in enumerate(boxes):



         
                box = bbox_from_mask(m)

 


                bsize = box[2] * box[3]
                print("size: ", bsize / (24*24) )
                #flattened = torch.flatten(torch.from_numpy(m))
                #indices_orig = torch.where(flattened)[0]

                _, image_tensor2, image_sizes2 = dataset.__getitem__(index  , bbox = box , P = 24, split = False, n = 4, path = None, augmentation = None)  #f"{output_dir}/cut.png"
                if bsize <0.0:
                    #bigger_box = [max(0,box[0]-6), max(0,box[1]-6), min(box[2]+ 6, 24), min(box[3]+ 6, 24)  ]
                    #indices_orig =torch.tensor( box_to_indices(bigger_box, 24)  )
                    emb_vis, att_mask, position_ids, attn = get_zoomed_embed(model, index, tokenizer, image_sizes2, begin_pos_vis, vis_len, box, dataset,input_ids, P = 24)
                else:
                   
                    indices_orig =torch.tensor( box_to_indices(box, 24)  )
                    emb_vis, att_mask, position_ids = get_embedding(model,input_ids,image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_sizes2])
                    emb_vis_pruned = emb_vis[0,begin_pos_vis:begin_pos_vis + vis_len,:]
                    
                    #box_bigger = [max(0,box[0]-3), max(0,box[1]-3), min(box[2]+ 3, 24), min(box[]+ 3, 24)  ]
                    #orig_embeds_vis.reshape((24,24, -1)) box_to_indices(box, 24)
                


                attn  = get_attn_layers(model,input_ids, image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),None, input_embeds = emb_vis, num_layer = 25, attention_mask = att_mask, position_ids = position_ids).cuda()



                general_att_map = get_general(tokenizer, model, image_tensor2, image_sizes2, begin_pos_vis,vis_len )

                if DO_PLOT:
                    image = get_image(dataset, index, box = None, P = 24)
                    image.save(f"/cluster/scratch/mgroepl/debug/Orig.png")



                
                    im = get_image(dataset, index, box = box, P = 24)
                    im.save(f"/cluster/scratch/mgroepl/debug/{ind}manual.png")


                indices = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = len(indices_orig) / (24*24), largest = True, attn_mean_all = None, general_att_map = None) #bsize / (24*24)
                #indices = torch.sort(indices).values
                #indices_orig = torch.sort(indices_orig).values
       
                if do_clip:
                    #emb_vis = get_clip_embed(model,image_tensor2.unsqueeze(0).to(dtype=torch.float32, device='cuda', non_blocking=True), indices_orig, indices_orig )
                    
                    emb_vis2, att_mask, position_ids = get_embedding(model,input_ids,image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_sizes2], new_pos = indices_orig,orig_pos=indices)
                    print(torch.equal(emb_vis2, emb_vis))
                    orig_embeds_vis_cut = prune_indices(emb_vis2, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)[:,begin_pos_vis:begin_pos_vis + len(indices_orig) ,:]
                    

                else:
                    orig_embeds_vis_cut = prune_indices(emb_vis, indices,vis_len, begin_pos_vis,img_path = None,question_id = None, question = None , invert = True)[:,begin_pos_vis:begin_pos_vis + len(indices_orig) ,:]
               
                if DO_PLOT:
                    plot_mask(im, indices, vis_len, ind, question)

             
               
                #orig_embeds_vis[box[0]:box[0] + box[2], box[1]:box[1] + box[3],:]  = orig_embeds_vis_cut.clone().view((box[2],box[3],-1))
                #orig_embeds = orig_embeds_vis.reshape((-1,orig_embeds.shape[-1]))
            
                orig_embeds[:,indices_orig + begin_pos_vis,:] = orig_embeds_vis_cut
     
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



            text2 = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
   
       
        
        gt = box_data[index]["label"]
        print("output:",text2 )
        print("gt: ",gt )
        if text.lower() == gt:
            pos += 1
        else:
            neg += 1
        if text2.lower() == gt:
            pos_in += 1
        else:
            neg_in += 1

        print("original acc: ", (pos) / (pos + neg))
        print("original acc injected: ", (pos_in) / (pos_in + neg_in))
    print(f"average pruning: {avg_pruning}")

    

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
