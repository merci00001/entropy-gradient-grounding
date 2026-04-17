import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *

# hyperparameters
NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
PATCH_SIZE = 14
IMAGE_RESOLUTION = 336
IMAGE_TOKEN_INDEX = 32000
ATT_LAYER = 14

def gradient_attention_llava(image, prompt, general_prompt, model, processor):
    """
    Generates an attention map using gradient-weighted attention from LLaVA model.
    
    This function computes attention maps from the LLaVA model and weights them by their
    gradients with respect to the loss. It focuses on the attention paid to image tokens
    in the final token prediction, highlighting regions relevant to the prompt.
    
    Args:
        image: Input image to analyze
        prompt: Text prompt for which to generate attention
        general_prompt: General text prompt (not directly used in this function)
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the gradient-weighted attention map
    """
    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)
    
    # Compute loss
    outputs = model(**inputs, output_attentions=True)
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)
    
    # Compute attention and gradients
    attention = outputs.attentions[ATT_LAYER]
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])
    
    # Compute the attention maps
    att_map = grad_att[0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    
    return att_map

def rel_attention_llava(image, att_map, general_prompt, model, processor):
    """
    Generates a relative attention map by comparing specific prompt attention to general prompt attention.
    
    This function computes attention maps for both a specific prompt and a general prompt in the LLaVA model,
    then calculates their ratio to highlight regions that are uniquely relevant to the specific prompt.
    It focuses on the attention paid to image tokens in the final token prediction.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate attention
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing 
                the relative attention map (specific/general)
    """
    # Prepare inputs for the prompt
    #inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    #pos = inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute attention map for the 14th layer
    #att_map = model(**inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, pos:pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    # Prepare inputs for the general prompt
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_pos = general_inputs['input_ids'][0].tolist().index(IMAGE_TOKEN_INDEX)

    # Compute general attention map for the 14th layer
    general_att_map = model(**general_inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, general_pos:general_pos+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)
    # Normalize attention map
    att_map = att_map / general_att_map

    return att_map

def pure_gradient_llava(image, prompt, general_prompt, model, processor):
    """
    Generates a gradient-based attention map using direct image gradients in LLaVA.
    
    This function computes gradients of the loss with respect to the input image pixels
    for both specific and general prompts. It then calculates their ratio and applies
    a high-pass filter to highlight fine-grained details that are uniquely relevant to the specific prompt.
    
    Args:
        image: Input image to analyze
        prompt: Specific text prompt for which to generate gradients
        general_prompt: General text prompt for baseline comparison
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs
        
    Returns:
        grad: A 2D numpy array representing the processed gradient map highlighting
              regions relevant to the specific prompt
    """
    # Process inputs
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(general_prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    
    # Apply high pass filter
    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)
    
    # Enable gradients
    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass for inputs
    zero_logit = model(**inputs, output_hidden_states=False).logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]
    
    # Forward pass for general_inputs
    general_zero_logit = model(**general_inputs, output_hidden_states=False).logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)
    
    # Compute general gradients
    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]
    
    # Process gradients
    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Compute gradient norms
    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)
    
    # Normalize and apply high pass filter
    grad = grad / general_grad
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass
    
    # Reduce gradient block size
    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    return grad





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