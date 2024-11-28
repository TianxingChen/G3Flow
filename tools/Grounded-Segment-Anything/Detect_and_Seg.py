import pdb
import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import sys
sys.path.append('./GroundingDINO/groundingdino')

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from huggingface_hub import hf_hub_download

import torch

# segment anything
import sys
sys.path.append('./segment_anything/')
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def download_image(url, image_file_path): # for testing
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    _ = model.eval()
    return model   

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

class GroundedSAM:
    def __init__(self, device='cuda', sam_checkpoint='sam_vit_h_4b8939.pth'):
        self.DEVICE = device
        self.test = False

        # load SAM
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # load GroundingDINO
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    def get_top_k_box(self, boxes, logits, phrases, topk=1):
        top_k_indices = np.argsort(logits)[-topk:]
        top_k_indices_list = top_k_indices.tolist()
        return boxes.index_select(0, top_k_indices), logits.index_select(0, top_k_indices), [phrases[i] for i in top_k_indices_list]
    
    def detect(self, image, text_prompt, BOX_TRESHOLD=0, TEXT_TRESHOLD=0, topk=1): # image: torch -> (3, H, W)
        if image.shape[2] == 3:
            image = torch.Tensor(image.copy())
            image.permute(2, 0, 1)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.groundingdino_model, 
                image=image, 
                caption=text_prompt, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD,
                device=self.DEVICE
            )
            topk_boxes, topk_logits, topk_phrases = self.get_top_k_box(boxes, logits, phrases, topk=topk)
        return topk_boxes, topk_logits, topk_phrases
    
    def segment(self, image_source, boxes): # numpy_array: (H, W, 3)
        with torch.no_grad():
            sam_predictor = self.sam_predictor
            sam_predictor.set_image(image_source)
            H, W, _ = image_source.shape
            tmp_xxyy = box_ops.box_cxcywh_to_xyxy(boxes) 
            boxes_xyxy = tmp_xxyy * torch.Tensor([W, H, W, H]).to(tmp_xxyy.device)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.DEVICE)
            masks, _, _ = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        multimask_output = False,
                    )
        return masks
    
    def convert_image(self, image):
        # tmp_image = torch.Tensor(image)
        # tmp_image = tmp_image.permute(2, 0, 1)
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        return image_transformed
    
    def detect_and_seg(self, image, text_prompt, BOX_TRESHOLD=0, TEXT_TRESHOLD=0, topk=1): # numpy_array: (H, W, 3)
        tmp_image = self.convert_image(image)
        boxes, logits, phrases = self.detect(image=tmp_image, text_prompt=text_prompt, topk=topk)
        self.sam_predictor.set_image(image)
        masks = self.segment(image_source=image, boxes=boxes)
        return masks

if __name__ == '__main__':
    model = GroundedSAM(sam_checkpoint='../weights/sam_vit_h_4b8939.pth')
    model.detect_and_seg(None, 'ok')
