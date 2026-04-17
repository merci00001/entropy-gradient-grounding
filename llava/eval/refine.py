from utils import get_attn_layers, get_disjoint_segments,bbox_from_mask, box_to_indices,  get_embedding, get_indices_percent




def refine(model,input_ids,image_tensor,image_sizes, orig_embeds , att_mask, position_ids, dataset, index ):
            
   
    vis_len = 576
                

    grad_orig_flat = None



    attn_orig , first_hidden_orig = get_attn_layers(model,input_ids,image_tensor, image_sizes, input_embeds = orig_embeds, num_layer = 25, attention_mask = att_mask,position_ids = position_ids )


    boxes, sorted_order = get_disjoint_segments(attn_orig, 24,begin_pos_vis, vis_len = vis_len, return_single = False ,insert_mask = None, grad = grad_orig_flat)


    for ind, m in enumerate(boxes):
        
        box = list(bbox_from_mask(m))
        bsize = box[2] * box[3]

        _, image_tensor2, image_sizes2 = dataset.__getitem__(index  , bbox = box , P = 24, split = False, n = 4, path = None, augmentation = None)  #f"{output_dir}/cut.png"


        indices_orig =torch.tensor( box_to_indices(box, 24)  )
    
                
        emb_vis, att_mask, position_ids = get_embedding(model,input_ids,image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),[image_sizes2])
            



        attn, logits  = get_attn_layers(model,input_ids, image_tensor2.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),None, input_embeds = emb_vis, num_layer = 25, attention_mask = att_mask, position_ids = position_ids)
    
        indices, ent = get_indices_percent(attn, begin_pos_vis, vis_len = 576, mode = "selected", topK = len(indices_orig) / (24*24), largest = True, width = box[2] , height =box[3] , sample = False,grad = None ) #bsize / (24*24)


        
        orig_embeds[:,indices_orig + begin_pos_vis,:] = orig_embeds_vis_cut.clone()#emb_vis[:,begin_pos_vis + indices]#

    return orig_embeds