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
        print(mode)
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


def do_forward_midway(model, input_ids,image_tensor,image_sizes, tokenizer,  attention_mask=None):
    # Ensure model params do not store grad

    
    encoder_layers = model.model.layers
    
    result = ""
    for u in range(6):
        input_embeds , att_mask, position_ids= get_embedding(model,input_ids,image_tensor,image_sizes)
        hidden_states = input_embeds
        position_ids = torch.arange(input_embeds.shape[1], dtype=torch.long, device=input_embeds.device).unsqueeze(0)
        attn_outputs = []
        num_layers_to_run = 25
        attention_mask = build_decoder_attention_mask(
            None,
            input_embeds.size(),
            input_embeds
        )
        for i, layer in enumerate(encoder_layers[:num_layers_to_run]):
            # Pass through one layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                use_cache=False,
                past_key_value=None,
            )

            # Get hidden states and attentions
            hidden_states = layer_outputs[0]  # updated hidden state

            
        device = torch.device("cuda:0")  # or whichever GPU you want

        #att, hidden_state = get_attn_layers(model,None,None, None, input_embeds = input_embeds,num_layer = 2, attention_mask = attention_mask,position_ids = None )


        linear_layer = model.lm_head  # This is the part of the model that generates logits
        logits = linear_layer(hidden_states) 
        # Print out the layer to check its structure



        # Compute objective
        temperature = 1.0   # higher = flatter distribution
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)


        most_probable_word_index = torch.argmax(probs)

        # Step 3: Decode the index to the actual word using the tokenizer
        # This assumes that the logits are for a model that uses a tokenizer with subword or wordpiece tokens
        decoded_word = tokenizer.decode([most_probable_word_index.item()])
        print(f"The most probable word is: {decoded_word}, with len {len(decoded_word)}")
        result += decoded_word
        decoded_word_embedded = tokenizer(decoded_word, return_tensors="pt", padding=True, truncation=True)
        print("decoded_word_embedded ", decoded_word_embedded )
        input_new = decoded_word_embedded["input_ids"][:,-1].cuda().unsqueeze(0)

        input_ids = torch.cat((input_ids,input_new), dim = 1)

        


    return


def calc_grad_plot(model, input_embeds, tokenizer, begin_pos_vis, attention_mask=None, vis_len = 576):
    # Ensure model params do not store grad
    maximums = []
    for x in range(32):
        grad_orig_max = torch.norm(
            calc_grad(model, input_embeds,tokenizer, attention_mask=None, layer = x)[
                0, begin_pos_vis:begin_pos_vis + vis_len, :
            ],
            p=2,           # e.g., p=1 (L1), p=2 (L2), p='inf'
            dim=-1
                ).max()  
        maximums.append(grad_orig_max.cpu())
    print(maximums)
    x = np.arange(len(maximums))

    # Plot the array values against their indices (x)
    plt.plot(x, maximums)
    plt.savefig("/cluster/scratch/mgroepl/debug/plot.png")
    plt.close()

def update_embeds(model, input_embeds, tokenizer, begin_pos_vis, attention_mask=None, vis_len = 576):
    # Ensure model params do not store grad
    lr = 3.0
    steps = 5
    for x in range(steps):
        grad_orig_max = calc_grad(model, input_embeds,tokenizer, attention_mask=None, layer = 32)[:, begin_pos_vis:begin_pos_vis + vis_len, :] 
        input_embeds[:,begin_pos_vis:begin_pos_vis + vis_len,:] += grad_orig_max.clone()* lr
    return input_embeds
def calc_grad(model, input_embeds, tokenizer,  attention_mask=None, layer = 25):
    # Ensure model params do not store grad
    model.requires_grad_(False)
    
    # Only input embedding requires grad
    input_embeds = input_embeds.detach().clone().requires_grad_(True)

    # Forward pass under no_grad for model params,
    # but allow grad for input_embeds
    with torch.set_grad_enabled(True):


        encoder_layers = model.model.layers
        hidden_states = input_embeds
        
        
        position_ids = torch.arange(input_embeds.shape[1], dtype=torch.long, device=input_embeds.device).unsqueeze(0)
        attn_outputs = []
        num_layers_to_run = layer
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
                output_attentions=False,
                use_cache=False,
                past_key_value=None,
            )

            # Get hidden states and attentions
       
            hidden_states = layer_outputs[0]   # updated hidden state

            
        device = torch.device("cuda:0")  # or whichever GPU you want

        #att, hidden_state = get_attn_layers(model,None,None, None, input_embeds = input_embeds,num_layer = 2, attention_mask = attention_mask,position_ids = None )


        linear_layer = model.lm_head  # This is the part of the model that generates logits
        logits = linear_layer(hidden_states) 
        # Print out the layer to check its structure


   
        temperature = 1.0   # higher = flatter distribution
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)


        #probs = probs[probs>0.3]
        indices = torch.where(probs > 0.1)[0]

        # Renormalize so everything sums to 1 again

        non_zero_mask = probs != 0
        probs = probs[non_zero_mask]
        log_probs = torch.log(probs)

        valid_mask = ~torch.isinf(log_probs)


        # Compute the objective by summing only over valid values
        #objective = torch.sum(probs * log_probs)


        uniform = torch.ones_like(probs) / probs.size(-1)
        objective = torch.sum(probs * torch.log(probs / uniform), dim=-1)
        #top_values, top_indices = torch.topk(probs, k=5, dim=-1)
        #objective = top_values[0]  + top_values[1]  #torch.max(probs, dim=-1).values
        print("Enthropy: ", objective*-1)
        # Backprop ONLY through input_embeds
        objective.backward()

        # Grab gradients
        grads = input_embeds.grad.detach().clone()
       
    # memory cleanup
    del  objective
    torch.cuda.empty_cache()

    return grads


def get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = None,num_layer = None, attention_mask = None,position_ids = None ):

    with torch.inference_mode():

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




def find_best_match_by_overlap(target_box, boxes):
    best_overlap = 0
    best_box = None

    for box in boxes:
        overlap = overlap_pixels(target_box, box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_box = box

    return best_box, best_overlap




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
