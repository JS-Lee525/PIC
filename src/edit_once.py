import os, pdb

import argparse
import numpy as np
import torch
import requests
import glob
from PIL import Image
import argparse
import os
import copy
import os, sys

import torch
from PIL import Image
import glob
from tqdm.autonotebook import tqdm


import cv2
import matplotlib.pyplot as plt

from lavis.models import load_model_and_preprocess
from diffusers import DDIMScheduler
from utils.pic_pipeline import PicPipeline
from pytorch_lightning import seed_everything
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import torch
from PIL import Image, ImageDraw, ImageFont
import glob
from tqdm.autonotebook import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='assets/test_images/cat_a.png')
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--tau', type=int, default=25)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--use_wordswap', action='store_true')
    args = parser.parse_args()
    
    seed_everything(42)
    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)
        
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))
    
    pipe = PicPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob.glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]

    for img_path in l_img_paths:
        bname = os.path.basename(img_path).split(".")[0]
        img_num = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
        img = Image.open(img_path).convert('RGB').resize((512,512), Image.Resampling.LANCZOS)
        # generate the caption
        _image = vis_processors["eval"](img).unsqueeze(0).to(device)
        prompt_str = model_blip.generate({"image": _image})[0]

        x_img, x_rec = pipe(
            prompt_str, 
            guidance_scale_for=1,
            guidance_scale_rev=args.negative_guidance_scale,
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype,
            tau=args.tau,
            task_name=args.task_name,
            use_wordswap=args.use_wordswap,
            beta=args.beta,
            gamma=args.gamma,
        )
        
        bname = os.path.basename(img_path).split(".")[0]
        x_img[0].save(os.path.join(args.results_folder, f"edit/{bname}.png"))
        x_rec[0].save(os.path.join(args.results_folder, f"reconstruction/{bname}.png"))
