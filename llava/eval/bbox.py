import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


def combine_heads(attn: torch.Tensor, selected: List[Dict], P: int, sigma: float) -> np.ndarray:
    """Combine selected heads with optional Gaussian smoothing.

    attn: [L, H, 1, V]
    Returns: combined 2D map [P, P] as numpy float32
    """
    M = np.zeros((P, P), dtype=np.float32)
    for item in selected:
        l, h = item["layer"], item["head"]
        a2d = attn[l, h, 0].reshape(P, P).detach().cpu().to(torch.float32).numpy()
        if sigma and sigma > 0:
            a2d = gaussian_filter(a2d, sigma=sigma)
        M += a2d.astype(np.float32)
    return M


def binarize_mean_relu(M: np.ndarray) -> np.ndarray:
    m = M.mean()
    B = np.maximum(M - m, 0.0)
    return (B > 0).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def upscale_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    # mask: [P, P] -> [H, W] using bilinear
    P = mask.shape[0]
    H, W = image_size[1], image_size[0]
    t = torch.from_numpy(mask.astype(np.float32))[None, None]  # [1,1,P,P]
    t_up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return (t_up.detach().cpu().numpy() > 0.5).astype(np.uint8)


def scale_bbox_to_image(bbox_grid: Tuple[int, int, int, int], image_size: Tuple[int, int], P: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_grid
    W, H = image_size
    sx, sy = W / P, H / P
    # Convert grid box (inclusive) to pixel box (inclusive)
    px0 = int(x0 * sx)
    py0 = int(y0 * sy)
    px1 = int(min(W - 1, (x1 + 1) * sx - 1))
    py1 = int(min(H - 1, (y1 + 1) * sy - 1))
    return px0, py0, px1, py1


def save_bbox_json(path: str, bbox_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int], heads: List[Dict]) -> None:
    obj = {
        "bbox_xyxy": list(map(int, bbox_xyxy)),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "selected_heads": [{"layer": h["layer"], "head": h["head"], "spatial_entropy": float(h["spatial_entropy"])} for h in heads],
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_mask_png(path: str, mask: np.ndarray) -> None:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img.save(path)

