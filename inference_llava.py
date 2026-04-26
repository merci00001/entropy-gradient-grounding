import argparse
import os
import json
import math
import shortuuid
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from refine import refine_big
from utils import get_embedding, find_crop_in_global,iou, iogt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_list(lst, n):
    """Split a list into n roughly equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    def __init__(self, questions, is15, image_folder, tokenizer, image_processor, model_config, conv_mode="llava_v1"):
        self.questions = questions
        self.is15 = is15
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def _build_prompt(self, text, do_yes=False):
        if do_yes:
            text = (
                "Can you answer the following question given the image or is there not enough "
                "information? Answer with a simple yes or no.'\n' " + text
            )
        if self.model_config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text
        return text

    def _crop_image(self, image, bbox, grid_w=24, grid_h=24):
        """Crop image given a bbox in grid coordinates (x, y, w, h)."""
        x, y, bw, bh = bbox
        scale_x = image.size[0] / grid_w
        scale_y = image.size[1] / grid_h
        x_min = int(x * scale_x)
        x_max = int(x_min + bw * scale_x)
        y_min = int(y * scale_y)
        y_max = int(y_min + bh * scale_y)
        b = [x_min, y_min, x_max,y_max]
        b = self.adjust_box_aspect_ratio(b, image.size[0], image.size[1], max_ratio=3.0)
        return image.crop(b)

    def adjust_box_aspect_ratio(self, box, img_w, img_h, max_ratio=2.0):
        """
        box: (x_min, y_min, x_max, y_max)
        max_ratio: maximum allowed w/h ratio (and reciprocal)

        Returns adjusted box.
        """
        x_min, y_min, x_max, y_max = box

        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return box  # invalid box, return as-is

        aspect = w / h

        # Box center
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        new_w, new_h = w, h

        if aspect > max_ratio:
            # too wide → increase height
            new_h = w / max_ratio

        elif aspect < 1 / max_ratio:
            # too tall → increase width
            new_w = h * max_ratio

        # Reconstruct box (centered)
        x_min = cx - new_w / 2
        x_max = cx + new_w / 2
        y_min = cy - new_h / 2
        y_max = cy + new_h / 2

        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        return (x_min, y_min, x_max, y_max)

    def __getitem__(
        self,
        index,
        bbox=None,
        grid_w=24,
        grid_h=24,
        insert_image=None,
        return_img=False,
        multi=False,
        do_yes=False,
    ):
        line = self.questions[index]
        qs = self._build_prompt(line["text"], do_yes=do_yes)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, line["image"])).convert("RGB")

        if insert_image is not None:
            image = insert_image

        if bbox is not None:
       
            image = self._crop_image(image, bbox, grid_w, grid_h)

        if return_img:
            return image

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        if not multi and not self.is15:
            image_tensor = image_tensor[0].unsqueeze(0)

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    return torch.stack(input_ids), torch.stack(image_tensors), image_sizes


def create_data_loader(questions, is15, image_folder, tokenizer, image_processor, model_config, conv_mode="llava_v1", num_workers=4):
    dataset = VQADataset(questions, is15, image_folder, tokenizer, image_processor, model_config, conv_mode=conv_mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return loader, dataset


# ---------------------------------------------------------------------------
# Multi-image helpers
# ---------------------------------------------------------------------------

QUAD_BBOXES = [
    [0, 0, 12, 12],
    [12, 0, 12, 12],
    [0, 12, 12, 12],
    [12, 12, 12, 12],
]


def get_multi_images(dataset, index):
    """Return the full image plus four quadrant crops."""
    full = dataset.__getitem__(index, return_img=True)
    quads = [dataset.__getitem__(index, bbox=bb, return_img=True) for bb in QUAD_BBOXES]
    return [full] + quads


def get_multi_tensors(dataset, index):
    """Return stacked image tensor for full + four quadrant crops."""
    tensors = []
    for bb in [None] + QUAD_BBOXES:
        _, tensor, _ = dataset.__getitem__(index, bbox=bb)
        tensors.append(tensor[0])
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

def should_stop(criterion, start, fixed, iterations, second_ent, start_ent, boxes_previous):
    """Return True if the refinement loop should stop."""
    if criterion == "iter" and iterations <= start:
        return True
    if criterion == "fixed" and fixed <= start:
        return True
    if criterion == "ent":
        return second_ent > start_ent or second_ent == 0 or start >= 15
    if criterion == "res":
        if len(boxes_previous) > 1:
            s = boxes_previous[1].size
            return s[0] * s[1] <= 4000
        return True
    return False


# ---------------------------------------------------------------------------
# Optional GT bbox loading
# ---------------------------------------------------------------------------

def load_gt_bbox_data(textvqa_val_path, img_to_id_path):
    box_data = load_dataset("jrzhang/TextVQA_GT_bbox")["train"]
    id_to_index = {ex["dataset_id"]: i for i, ex in enumerate(box_data)}
    id_to_bbox = {item["dataset_id"]: item["bbox"] for item in box_data}

    with open(img_to_id_path, "r") as f:
        img_to_id = json.load(f)["data"]

    ordered_ids = [item["question_id"] for item in img_to_id]
    new_order = [id_to_index[str(qid)] for qid in ordered_ids if str(qid) in id_to_index]
    box_data = box_data.select(new_order)

    return box_data, id_to_index, ordered_ids


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_refinement_loop(
    args, model, input_ids, image_tensor, image_sizes, dataset, index,
    begin_pos_vis, starting_images, orig_embeds, att_mask, position_ids, vis_len, ind
):
    """Run the iterative refinement loop and return final embeddings + metadata."""
    mode = "grad"
    return_prob = args.criterion == "prob"
    start_ent = 1000
    start = -1

    orig_embeds, prob_after, _, boxes_previous, boxes, ents_start, g_max, _ = refine_big(
        model, input_ids, image_tensor, image_sizes,
        orig_embeds.clone(), att_mask, position_ids,
        dataset, index, begin_pos_vis,
        is_multi=args.do_multi, grad_type=args.grad_type,
        to_run=args.to_run, layer=args.layer, vis_len=vis_len,
        do_append=True, iteration=0, plot=args.plot,
        boxes_previous=starting_images, add=len(starting_images),
        final=False, samples=4, mode=mode, tokens = args.token_number, index_data = ind
    )




    second_ent = ents_start[0] if args.criterion == "ent" else None
    while True:
        start += 1


        orig_embeds2, prob_before, prob_after2, boxes_previous2, boxes, ents, g_max, _ = refine_big(
            model, input_ids, image_tensor, image_sizes,
            orig_embeds.clone(), att_mask, position_ids,
            dataset, index, begin_pos_vis,
            return_prob=return_prob, grad_type=args.grad_type,
            to_run=args.to_run, layer=args.layer, prob_before=prob_after,
            do_append=True, iteration=start + 1, plot=args.plot,
            boxes_previous=boxes_previous, add=len(boxes_previous),
            final=False, samples=1, mode=mode, pad=False,tokens = args.token_number, index_data = ind
        )




        if args.criterion == "ent":

            if should_stop(args.criterion, start, args.iterations,
                        max(image_sizes[0][0], image_sizes[0][1]) // 1000,
                        second_ent, start_ent, boxes_previous):
                break   
            start_ent = second_ent
            second_ent = ents[0]
            boxes_previous = boxes_previous2
            
            
            

            orig_embeds = orig_embeds2

        elif args.criterion in ("iter", "fixed"):
            orig_embeds = orig_embeds2
            prob_after = prob_after2
        elif args.criterion == "res":
            if len(boxes_previous) > 1 and boxes_previous[1].size[0] * boxes_previous[1].size[1] > 4000:
                orig_embeds = orig_embeds2
                prob_after = prob_after2
        elif args.criterion == "prob":
            orig_embeds = orig_embeds2
            if prob_after2 <= prob_after:
                if args.plot:
                    boxes_previous[1].save("/cluster/scratch/mgroepl/debug/test/FINAL.png")
                break
            prob_after = prob_after2
            

    return orig_embeds, boxes_previous, start_ent, g_max


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, device="cuda", use_flash_attn=False
    )
    model.config.mm_patch_merge_type = "flat"
    model.requires_grad_(False)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode += "_mmtag"
        print(f"Auto-switching conv_mode to {args.conv_mode}.")

    data_loader, dataset = create_data_loader(
        questions, args.is15, args.image_folder, tokenizer, image_processor, model.config,
        conv_mode=args.conv_mode,
    )

    vis_len = 576

    # Optionally load GT bbox data
    box_data, id_to_index, ordered_ids = (None, None, None)
    IoUs = []
    if args.load_data:
        
        box_data = load_dataset("jrzhang/TextVQA_GT_bbox")["train"]

        id_to_index = {ex["dataset_id"]: i for i, ex in enumerate(box_data)}
        ids = box_data["dataset_id"]
        id_to_bbox = {item["dataset_id"]: item["bbox"] for item in box_data}
        with open("/cluster/scratch/mgroepl/data/textvqa/val.json", "r") as f:
            img_to_id = json.load(f)["data"]
        ordered_ids = [item["question_id"] for item in img_to_id]
        new_order = [id_to_index[str(qid)] for qid in ordered_ids if str(qid) in id_to_index]
        index_data = -1
        box_data = box_data.select(new_order)

    with open(answers_file, "w") as ans_file:
        for idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(
            tqdm(zip(data_loader, questions), total=len(questions))
        ):


            # GT bbox size (for logging)
            box_dataset_orig_size = -1.0
            if args.load_data:
                ordered_id = ordered_ids[idx]

                if str(ordered_id) in id_to_index:
                    index_data += 1
                    box_dataset_orig = box_data[index_data]["bbox"]
                    box_dataset_orig_size = (box_dataset_orig[2]*box_dataset_orig[3]) / (image_sizes[0][0]*image_sizes[0][1])
                else:
                    continue
                    box_dataset_orig_size = -1.0
            else:
                box_dataset_orig_size = -1.0

            if idx < args.index:
                continue


            input_ids = input_ids.to("cuda", non_blocking=True)
            x_flat = input_ids.flatten()
            begin_pos_vis = (x_flat == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()

            # Build starting images / embeddings
            if args.do_multi:
                starting_images = get_multi_images(dataset, idx)
                stacked_tensor = get_multi_tensors(dataset, idx).unsqueeze(0)
                orig_embeds, att_mask, position_ids = get_embedding(model, input_ids, stacked_tensor, image_sizes)
            else:
                orig_embeds, att_mask, position_ids = get_embedding(model, input_ids, image_tensor, image_sizes)
                starting_images = [dataset.__getitem__(idx, return_img=True)]

            if args.plot:

                outputs = model.generate(
                    None,
                    attention_mask=None,
                    inputs_embeds=orig_embeds,
                    images=None,
                    image_sizes=image_sizes,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=100,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

                answer_orig = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
                print("Original Answer:", answer_orig)



            # Refinement loop
            if True:
                orig_embeds, boxes_previous, start_ent, g_max = run_refinement_loop(
                    args, model, input_ids, image_tensor, image_sizes,
                    dataset, idx, begin_pos_vis, starting_images,
                    orig_embeds, att_mask, position_ids, vis_len, idx
                )

                # Rebuild final embeddings with refined visual tokens
                input_ids2, image_tensor2, _ = dataset.__getitem__(idx)
                orig_embeds2, att_mask2, position_ids2 = get_embedding(
                    model, input_ids2.unsqueeze(0).cuda(), image_tensor2.unsqueeze(0).cuda(), image_sizes
                )
                begin_pos_vis2 = (input_ids2.flatten() == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()

                refined_vis = orig_embeds[:, begin_pos_vis:begin_pos_vis + vis_len * len(boxes_previous), :]
                if args.load_data:
                    bbox_final = find_crop_in_global(boxes_previous[0], boxes_previous[1])

                    box_dataset_orig[2] += box_dataset_orig[0]
                    box_dataset_orig[3] += box_dataset_orig[1]
                
                    iou_with_GT = iou(box_dataset_orig,bbox_final[0])
                    iogt_with_GT = iogt(bbox_final[0], box_dataset_orig)
                  
                    IoUs.append(iou_with_GT)
                    if args.plot:
                        boxes_previous[0].crop(bbox_final[0]).save("/cluster/scratch/mgroepl/debug/test/croptest.png")
                        boxes_previous[0].crop(box_dataset_orig).save("/cluster/scratch/mgroepl/debug/test/croptestGT.png")
                orig_embeds = torch.cat([
                    orig_embeds2[:, :begin_pos_vis2],
                    refined_vis,
                    orig_embeds2[:, begin_pos_vis2 + vis_len:],
                ], dim=1)

            # Generate answer
            outputs = model.generate(
                None,
                attention_mask=None,
                inputs_embeds=orig_embeds,
                images=None,
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=100,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )

            answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            print("Answer:", answer)
            torch.cuda.empty_cache()
            if args.load_data:
                print("Current average IoU: ", sum(IoUs) / len(IoUs))
            if args.plot:
                if answer == answer_orig:
                    continue
                else:
                    break

            ans_file.write(json.dumps({
                "question_id": line["question_id"],
                "prompt": line["text"],
                "text": answer,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "metadata": {},
                "entropy": 0.0,
                "box_size": box_dataset_orig_size,
            }) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--is15", type=bool, default=False)
    parser.add_argument("--criterion", type=str, default="ent",
                        choices=["ent", "iter", "fixed", "res", "prob"])
    parser.add_argument("--to_run", type=int, default=2)
    parser.add_argument("--do_multi", type=bool, default=False)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--grad_type", type=str, default="entropy")
    # formerly hard-coded globals
    parser.add_argument("--method", type=str, default="grad", choices=["attention", "grad"])
    parser.add_argument("--do_prune", type=bool, default=False)
    parser.add_argument("--load_data", type=bool, default=False)
    parser.add_argument("--multi_image", type=bool, default=False)
    parser.add_argument("--token_number", type=int, default=1)

    args = parser.parse_args()
    eval_model(args)
