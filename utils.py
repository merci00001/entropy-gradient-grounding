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
from torchvision import transforms
from skimage.filters import threshold_otsu, threshold_multiotsu

from scipy.spatial.distance import cdist
import cv2


def bbox_from_mask(mask) :
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return [x0, y0, width, height]





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




def iogt(pred, gt):
    x_left = max(pred[0], gt[0])
    y_top = max(pred[1], gt[1])
    x_right = min(pred[2], gt[2])
    y_bottom = min(pred[3], gt[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    return intersection / gt_area


def iou(boxA, boxB):
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - intersection

    return intersection / union

def get_disjoint_segments(attn_layers, W,H,begin_pos_vis_att, vis_len = 576, return_single = False , insert_mask = None, grad = None, plot = False, el = None):
    ent  = 0
    filtered_mask = None

    if grad is not None:
        grad_orig = grad.reshape(W,H).detach().cpu().to(torch.float32).numpy()

        temperature = 0.1
        grad_orig_flat2 = grad.clone()
        #grad_orig_flat2 = F.softmax(grad_orig_flat2 / temperature, dim=-1)
        grad_orig = grad_orig_flat2.reshape(W,H).detach().cpu().to(torch.float32).numpy()


        top_percentile = 99
        high_thresh = grad_orig.max()*0.01

        # Binary mask of high activations
        sigma = 1.5 #adaptive_sigma_gini(grad_orig, max_s = 2.5)



        grad_orig2 =   gaussian_filter(grad_orig, sigma=sigma)#  gaussian_filter(grad_orig, sigma=1.5)# gaussian_filter(grad_orig, sigma=1.5)  cv2.bilateralFilter(grad_orig, d=9, sigmaColor=75, sigmaSpace=75)
        
        print("grad_orig2", grad_orig2.min())




        if el is None:
            el =elbow_chord(grad_orig2.flatten()) #elbow_chord(grad_orig2.flatten()) # threshold_multiotsu(grad_orig2, classes=3)[-1] #  elbow_chord(grad_orig2.flatten())  # elbow_chord(grad_orig2.flatten()) #  elbow_chord(grad_orig2.flatten())#  elbow_chord(grad_orig2.flatten()) #elbow_chord(grad_orig2.flatten()) #  grad_orig2.max() *0.5  threshold_otsu(grad_orig2)


        ent = spatial_entropy(torch.tensor(grad_orig2), el)
        ent = ent["spatial_entropy"]
        print("spatial enthropy ", ent)



        grad_mask = grad_orig2 > el 
        blob_mask = grad_mask.astype(bool)


        mask_grid = blob_mask



        combo = grad_orig
    else:
        mask_grid = insert_mask
    if return_single:
        return [mask_grid], None,None#[bbox_from_mask(mask_grid)]
    labeled_array, num_features = label(mask_grid)
    segment_masks = [(labeled_array == i) for i in range(1, num_features + 1)]
   
    sorted_vals_per_segment = [ combo[b]   for b in segment_masks]

    
    return segment_masks,  sorted_vals_per_segment, ent




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



def _get_topp_indices(probs, p=0.5):
    """Return indices of the top-p nucleus (smallest set whose cumsum >= p)."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep all tokens up to and including the one that pushes cumsum over p
    cutoff = (cumsum < p).sum().item() + 1
    return sorted_indices[:cutoff]


def _run_full_forward(model, input_embeds, attention_mask):
    """Full model forward pass, returns logits."""
    out = model(
        attention_mask=attention_mask,
        inputs_embeds=input_embeds,
        image_sizes=None,
        output_attentions=False,
        return_dict=True,
    )
    return out.logits


def _run_partial_forward(model, input_embeds, attention_mask, num_layers):
    """Run only the first `num_layers` transformer layers, then lm_head."""
    position_ids = torch.arange(
        input_embeds.shape[1], dtype=torch.long, device=input_embeds.device
    ).unsqueeze(0)

    causal_mask = build_decoder_attention_mask(attention_mask, input_embeds.size(), input_embeds)

    hidden = input_embeds
    for i, layer in enumerate(model.model.layers[:num_layers]):
        layer_out = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
            past_key_value=None,
        )
        hidden = layer_out[0]
      
    return model.lm_head(hidden)


def calc_grad_image(model, image_tensor, input_ids, image_sizes,
                    attention_mask=None, layer=-1, mode="entropy"):
    """
    Like calc_grad but differentiates w.r.t. raw image pixels instead of input_embeds.
    Returns a (H, W) numpy gradient norm map.
    """
    pix = image_tensor.detach().clone().squeeze(1).to(torch.float16).requires_grad_(True)
    pix.retain_grad()

    # Keep vision tower in the graph
    model.model.vision_tower.requires_grad_(True)
    model.model.mm_projector.requires_grad_(True)

    try:
        with torch.set_grad_enabled(True):
            # Build input_embeds — vision tower is grad-enabled so graph stays alive
            (_, position_ids, att_mask, _, input_embeds, _) = (
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None,
                    pix,
                    image_sizes=image_sizes,
                )
            )
            print("input_embeds.requires_grad:", input_embeds.requires_grad)

            # Run the same objective as calc_grad
            if layer == -1:
                logits = _run_full_forward(model, input_embeds, att_mask)
            else:
                logits = _run_partial_forward(model, input_embeds, att_mask, num_layers=layer)

            probs = F.softmax(logits[0, -1], dim=-1)
            probs_nz = probs[probs > 0]

            if mode == "entropy":
                objective = (probs_nz * torch.log(probs_nz)).sum()
            elif mode == "max":
                objective = torch.log(probs_nz.max())
            else:
                raise ValueError(f"Unknown mode: {mode}")

            objective.backward()

    finally:
        model.model.vision_tower.requires_grad_(False)
        model.model.mm_projector.requires_grad_(False)
    print("input_embeds",input_embeds.grad)
    assert pix.grad is not None, "Gradient did not reach image tensor"
    grad = pix.grad.to(torch.float32).detach().cpu().numpy().squeeze()
    if grad.ndim == 4:
        grad = grad.mean(axis=0)
    # (C, H, W) → L2 norm over channels → (H, W)
    return np.linalg.norm(grad.transpose(1, 2, 0), axis=2)


def calc_grad(model, input_embeds, attention_mask=None, layer=-1, mode="entropy", gen_steps=1):
    model.requires_grad_(False)
    input_embeds = input_embeds.detach().clone().requires_grad_(True)
    embedding_table = model.get_input_embeddings().weight
    eos_token_id = model.config.eos_token_id

    with torch.set_grad_enabled(True):
        current_embeds = input_embeds
        current_mask = attention_mask
        prev_embeds = input_embeds        # context before the current step
        prev_mask = attention_mask        # mask before the current step
        actual_steps = gen_steps

        for step in range(gen_steps - 1):
            with torch.no_grad():
                if layer == -1:
                    logits = _run_full_forward(model, current_embeds, current_mask)
                else:
                    logits = _run_partial_forward(model, current_embeds, current_mask, num_layers=layer)

                next_token_id = logits[0, -1].argmax(dim=-1)

                if False: #next_token_id.item() == eos_token_id:
                    print(f"EOS hit at step {step + 1}, backpropagating step {step} instead.")
                    actual_steps = step
                    # Re-run previous context's forward pass with grad enabled
                    prev_input_embeds_final = torch.cat([
                        input_embeds,
                        prev_embeds[:, input_embeds.shape[1]:, :].detach()
                    ], dim=1)
                    if layer == -1:
                        logits = _run_full_forward(model, prev_input_embeds_final, prev_mask)
                    else:
                        logits = _run_partial_forward(model, prev_input_embeds_final, prev_mask, num_layers=layer)
                    probs = F.softmax(logits[0, -1], dim=-1)
                    break

                next_embed = embedding_table[next_token_id].detach()
                next_embed = next_embed.unsqueeze(0).unsqueeze(0)

            # Save previous context before extending
            prev_embeds = current_embeds
            prev_mask = current_mask

            current_embeds = torch.cat([current_embeds.detach(), next_embed], dim=1)
            if current_mask is not None:
                extra = torch.ones(1, 1, dtype=current_mask.dtype, device=current_mask.device)
                current_mask = torch.cat([current_mask, extra], dim=1)

        else:
            # No early EOS — run nth forward pass with grad enabled
            input_embeds_final = torch.cat([
                input_embeds,
                current_embeds[:, input_embeds.shape[1]:, :].detach()
            ], dim=1)
            if layer == -1:
                logits = _run_full_forward(model, input_embeds_final, current_mask)
            else:
                logits = _run_partial_forward(model, input_embeds_final, current_mask, num_layers=layer)
            temp = 1.0 # approx_temperature(logits[0, -1],10)
            print("temperature", temp)
            probs = F.softmax(logits[0, -1] / temp, dim=-1)

        # --- Shared objective + backward ---
        probs_nz = probs[probs > 0]
        print("Non-zero values: ", probs_nz.shape)
        topp_indices = _get_topp_indices(probs_nz, p=0.9)

        uniform = torch.ones_like(probs) / probs.size(-1)
        eps = 0.0 #1e-10
        probs_nz = torch.clamp(probs_nz, min=eps)

        if mode == "entropy":
            objective = (probs_nz * torch.log(probs_nz)).sum()*-1
        elif mode == "entropyTopP":
   
            p_nucleus = probs_nz[topp_indices]
            objective = (p_nucleus * torch.log(p_nucleus)).sum()
        elif mode == "max":
            objective = torch.log(probs_nz.max())
        elif mode == "KL":
            uniform = torch.full_like(probs_nz, 1.0 / probs_nz.size(-1))
            objective = (probs_nz * torch.log(probs_nz / uniform)).sum()
        elif mode == "KLTopP":
            p_nucleus = probs_nz[topp_indices]
            uniform_nucleus = torch.full_like(p_nucleus, 1.0 / p_nucleus.size(-1))
            objective = (p_nucleus * torch.log(p_nucleus / uniform_nucleus)).sum()
        else:
            raise ValueError(f"Unknown gradient mode: '{mode}'. "
                             f"Choose from: entropy, entropyTopP, max, KL, KLTopP.")
        print("objective: ", objective)
        objective.backward()
        grads = input_embeds.grad.detach().clone()

    del objective
    torch.cuda.empty_cache()
    print(f"Backpropagated at step {actual_steps}, max prob: {probs_nz.max()}")
    #grads = grads *-1
    #grads = grads.clamp_(min=0)
    #grads = grads *-1
    return grads, probs_nz.max().detach()



def get_prob_max(model, input_embeds,attention_mask = None):

    attn_layers = model(
        #input_ids = input_ids,    to(torch.float32)
        attention_mask=attention_mask,
        #images=[image_tensor.to(dtype=torch.float32)],
        inputs_embeds = input_embeds,
        image_sizes=None,
        output_attentions=True,
        return_dict=True,   
    )

    logits = attn_layers.logits
    last_logits = logits[0, -1]  
    probs = torch.softmax(last_logits, dim=-1)

    return probs.max()

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






