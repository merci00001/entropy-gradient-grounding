import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from utils import (
    get_disjoint_segments,
    bbox_from_mask,
    get_embedding,
    get_prob_max,
    get_unique_filename,
    calc_grad,
    elbow_chord,
)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def _compute_grad_flat(model, orig_embeds, att_mask, begin_pos_vis, vis_len, add, grad_type, layer, tokens = 1):
    """Run gradient computation and return (flat gradient norms, prob_end, g_max)."""
    g, prob_end = calc_grad(model, orig_embeds, attention_mask=att_mask, layer=layer, mode=grad_type, gen_steps = tokens)

    g_vis = g[0, begin_pos_vis:begin_pos_vis + vis_len * add, :]
    neg_sum = g_vis[g_vis < 0].sum()
    pos_sum = g_vis[g_vis > 0].sum()


    norms = torch.norm(g[0, begin_pos_vis:begin_pos_vis + vis_len * add, :], p=2, dim=-1)
    return norms, prob_end, g.max()


def _remap_multi_grad(grad_flat, vis_len):
    """Rearrange flat gradient from 4-quadrant layout into a 48×48 grid."""
    quad = grad_flat[vis_len:]
    grid = torch.zeros((48, 48), dtype=float)
    r = quad.reshape(24 * 4, 24)
    grid[:24, :24]   = r[:24]
    grid[:24, 24:]   = r[24:48]
    grid[24:, :24]   = r[48:72]
    grid[24:, 24:]   = r[72:]
    return grid


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_grad_plot(grad_map, iteration, index_data, label=""):
    base_dir = Path(f"/cluster/scratch/mgroepl/debug/test/{index_data}")
    base_dir.mkdir(parents=True, exist_ok=True)

    save_path = base_dir / f"{iteration}tensor_image{label}.png"

    plt.figure(frameon=False)
    plt.imshow(grad_map.detach().cpu().numpy(), cmap="viridis")
    plt.axis("off")  # removes axes, ticks, and numbers

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _save_concatenated_images(images, iteration, index_data):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    canvas = Image.new("RGB", (total_width, max_height))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width

    base_dir = Path(f"/cluster/scratch/mgroepl/debug/test/{index_data}")
    base_dir.mkdir(parents=True, exist_ok=True)  # create folder if needed
    save_path = base_dir / f"{iteration}concatenated_image.png"
    canvas.save(save_path)


def _plot_grad_results(grad_flat, grad_grid, is_multi, vis_len, add, boxes_previous, iteration, index_data):
    if is_multi:
        _save_grad_plot(grad_grid.reshape(48, 48), iteration, index_data)
        _save_grad_plot(grad_flat[:vis_len].reshape(24, 24), iteration,index_data, label="Single")
    else:
        _save_grad_plot(grad_flat.reshape(24 * add, 24), iteration, index_data)
    _save_concatenated_images(boxes_previous, iteration, index_data)


# ---------------------------------------------------------------------------
# Segment detection
# ---------------------------------------------------------------------------

def _collect_segments(
    mode, is_multi, attn_orig, grad_flat, grad_grid,
    begin_pos_vis, vis_len, add, plot, el
):
    """Return (boxes, vals, ents) across all sub-images."""
    all_boxes, all_vals, all_ents = [], [], []

    for ind in range(add):
        g = (grad_grid if is_multi else grad_flat[vis_len * ind: vis_len * (ind + 1)]) if mode == "grad" else None
        grid_size = (24 * 2, 24 * 2) if is_multi else (24, 24)

        segs, vals, ent = get_disjoint_segments(
            attn_orig, *grid_size,
            begin_pos_vis + ind * vis_len,
            vis_len=vis_len, return_single=False,
            insert_mask=None, grad=g, plot=plot, el=el,
        )

        masks = [bbox_from_mask(m) for m in segs]

        if is_multi:
            # shift y into the combined 48-row grid
            masks = [[b[0], b[1] + 24, b[2], b[3]] for b in masks]
            all_ents.append(ent)
            all_boxes.extend(masks)
            all_vals.extend(vals)
            break  # multi mode only uses the first pass
        else:
            masks = [[b[0], b[1] + ind * 24, b[2], b[3]] for b in masks]
            all_ents.append(ent)
            all_boxes.extend(masks)
            all_vals.extend(vals)

    return all_boxes, all_vals, all_ents


# ---------------------------------------------------------------------------
# Crop / embed helpers
# ---------------------------------------------------------------------------

def _get_crop_grid(is_multi, index_in_stack):
    """Return (W, H) grid size for dataset crop calls."""
    return (24 * 2, 24 * 2) if (is_multi and index_in_stack > 0) else (24, 24)


def _crop_image(dataset, index, box, grid_wh, src_image, pad):
    """Try to crop with bbox; fall back to full image on error."""
    w, h = grid_wh
    try:
        _, tensor, sizes = dataset.__getitem__(index, bbox=box, grid_w=w, grid_h=h, insert_image=src_image)
        img = dataset.__getitem__(index, bbox=box, grid_w=w, grid_h=h, insert_image=src_image, return_img=True)
    except Exception:
        _, tensor, sizes = dataset.__getitem__(index, insert_image=src_image)
        img = dataset.__getitem__(index, insert_image=src_image, return_img=True)
    return img


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def refine_big(
    model, input_ids, image_tensor, image_sizes,
    orig_embeds, att_mask, position_ids,
    dataset, index, begin_pos_vis,
    return_prob=False, grad_type="entropy", prob_before=0,
    layer=-1, vis_len=576, to_run=2, is_multi=False,
    do_append=False, boxes_previous=None, iteration=0,
    plot=False, add=1, mode="grad", final=False, samples=1, pad=False,tokens = 1, index_data= 0
):
    """
    Refine the visual embeddings by identifying the most salient image regions
    and appending their crops to the embedding sequence.

    Returns:
        (orig_embeds, prob_start, prob_end, boxified, added, ents_final, g_max, ents)
    """
    if boxes_previous is None:
        boxes_previous = []

    grad_flat = grad_grid = None
    attn_orig = prob_end = g_max = el = None

    # ------------------------------------------------------------------
    # Step 1: compute saliency (gradient or attention)
    # ------------------------------------------------------------------
    mode="grad"
    if mode == "grad":
        grad_flat, prob_end, g_max = _compute_grad_flat(
            model, orig_embeds, att_mask, begin_pos_vis, vis_len, add, grad_type, layer, tokens = tokens
        )

        if return_prob and prob_before > prob_end:
            return orig_embeds, prob_end, prob_end, boxes_previous, 0, [0], 0, 0

        if is_multi:
            grad_grid = _remap_multi_grad(grad_flat, vis_len)

        if plot:
            _plot_grad_results(grad_flat, grad_grid, is_multi, vis_len, add, boxes_previous, iteration, index_data)
    else:
        print("not supported)

    # ------------------------------------------------------------------
    # Step 2: find salient segments
    # ------------------------------------------------------------------
    boxes, vals, ents = _collect_segments(
        mode, is_multi, attn_orig, grad_flat, grad_grid,
        begin_pos_vis, vis_len, add, plot, el,
    )

    if not boxes:
        print("No boxes found.")
        return orig_embeds, 0, 0, boxes_previous, 0, [0], 0, 100

    # ------------------------------------------------------------------
    # Step 3: rank and select top-k boxes
    # ------------------------------------------------------------------
    vals_summed = [v.sum().item() if hasattr(v, "sum") else sum(v) for v in vals]
    ranked = sorted(zip(vals_summed, boxes), key=lambda x: x[0], reverse=True)
    top_vals, top_boxes = zip(*ranked[:to_run])
    top_boxes = list(top_boxes)

    ents_final = []
    if is_multi:
        ents_final = ents
    # ------------------------------------------------------------------
    # Step 4: crop images for top boxes
    # ------------------------------------------------------------------
    orig_img = dataset.__getitem__(index, return_img=True)
    if plot:

        base_dir = Path(f"/cluster/scratch/mgroepl/debug/test/{index_data}")
        base_dir.mkdir(parents=True, exist_ok=True)  # create folder if needed
        save_path = base_dir / f"orig{iteration}.png"

        orig_img.save(save_path)

    crop_images = []
    crop_values = []

    for val, box in zip(top_vals, top_boxes):
        box = list(box)
        stack_idx = box[1] // 24

        if not is_multi:
            box[1] -= 24 * stack_idx
            ents_final.append(ents[stack_idx])

        src_image = (
            dataset.__getitem__(index, grid_w=24 * 2, grid_h=24 * 2, insert_image=None, return_img=True)
            if is_multi else boxes_previous[stack_idx]
        )

        if is_multi and stack_idx > 0:
            box[1] -= 24

        grid_wh = _get_crop_grid(is_multi, stack_idx)
        crop = _crop_image(dataset, index, box, grid_wh, src_image, pad)

        crop_images.append(crop)
        crop_values.append(val)

    # ------------------------------------------------------------------
    # Step 5: embed crops and build new embedding sequence
    # ------------------------------------------------------------------
    n_keep = 1 if final else to_run
    ranked_crops = sorted(zip(crop_values, crop_images), key=lambda x: x[0], reverse=True)
    selected_crops = [img for _, img in ranked_crops[:min(n_keep, len(ranked_crops))]]

    vis_parts = [orig_embeds[:, :begin_pos_vis + vis_len, :]]

    boxified = [orig_img] + selected_crops

    for crop in selected_crops:
        _, crop_tensor, crop_sizes = dataset.__getitem__(index, insert_image=crop)
        crop_emb, att_mask, position_ids = get_embedding(
            model, input_ids,
            crop_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda", non_blocking=True),
            [crop_sizes],
        )
        vis_parts.append(crop_emb[:, begin_pos_vis:begin_pos_vis + vis_len])

    vis_parts.append(orig_embeds[:, begin_pos_vis + vis_len * add:])


    orig_embeds = torch.cat(vis_parts, dim=1)


    if plot:

        base_dir = Path(f"/cluster/scratch/mgroepl/debug/test/{index_data}")
        base_dir.mkdir(parents=True, exist_ok=True)  # create folder if needed
        
        save_path = base_dir / f"{iteration}final.png"

        best_val = max(crop_values)
        best_idx = crop_values.index(best_val)
        crop_images[best_idx].save(save_path)

    return orig_embeds, 0, prob_end, boxified, len(selected_crops), ents_final, g_max, ents
