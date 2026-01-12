import argparse
import json
import numpy as np
from PIL import Image  
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from datasets import Dataset, concatenate_datasets
from diffusers import (
    StableDiffusionXLAdapterPipeline, 
    T2IAdapter, 
    EulerAncestralDiscreteScheduler, 
    AutoencoderKL, 
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from diffusers.utils import load_image
from controlnet_aux.pidi import PidiNetDetector
from tqdm.auto import tqdm

import os
import sys
sys.path.append("..")
from dataset import _map_to_image_embedding

def collate_fn(batch):
    sketch = []
    description = []
    for b in batch:
        description.append(b["description"])
        sketch.append(b["sketch"])

    return {
        "description": description,
        "sketch": sketch,
    }

def generate_images(device,
                    batch_size, 
                    dataset_name,
                    num_turn,
                    sketch_dataset, 
                    processor, 
                    vision_model, 
                    diffusion_pipeline, 
                    pidinet,
                    ):

    if not os.path.exists("./../data/{}/gen_images/".format(dataset_name)):
        os.makedirs("./../data/{}/gen_images/".format(dataset_name))
    vision_model = vision_model.to(device)
    diffusion_pipeline = diffusion_pipeline.to(device)
    pidinet = pidinet.to(device)
    
    dataloader = DataLoader(sketch_dataset, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn, 
                            shuffle=False, 
                            pin_memory=True)
    negative_prompt = "bad quality, worst quality, worst detail, sketch, censor, simple background, jpeg artifacts, low quality"

    new_dataset_dict = {
        "image": [],
    }
    print("generate images for {}".format(dataset_name))
    process_bar = tqdm(range(len(dataloader)))

    for idx, batch in enumerate(dataloader):
        sketch = batch["sketch"]
        prompt = batch["description"]
        # p_list = []
        # for p in prompt:
        #     p = p + ", 4k photo, highly detailed"
        #     p_list.append(p)
        # prompt = p_list

        sketch_list = []
        for s in sketch:
            s = s.permute(1,2,0)# (C, H, W) -> (H, W, C)
            # sdxl
            s = pidinet(s, detect_resolution=1024, image_resolution=1024, apply_filter=True)
            sketch_list.append(s)

        gen_images = diffusion_pipeline(
                    image=sketch_list,
                    prompt=prompt,
                    negative_prompt=[negative_prompt]*len(prompt),
                    num_inference_steps=20,
                    adapter_conditioning_scale=0.9,
                    guidance_scale=7.5, 
                ).images

        new_dataset_dict["image"].extend(gen_images)
        process_bar.update(1)


    new_dataset = Dataset.from_dict(new_dataset_dict)
    new_dataset = new_dataset.with_format("torch")
    new_dataset = new_dataset.map(_map_to_image_embedding,
                                batched=True, 
                                batch_size = 100,
                                fn_kwargs={"blip_processor": processor, "blip_model": vision_model, "pidinet":pidinet}) # image_embedding
    print("new_dataset:{}".format(new_dataset))
    with open("./../data/{}/{}_turn{}_generate_image_embedding.parquet".format(dataset_name, dataset_name, num_turn), "wb") as f:
        new_dataset.to_parquet(path_or_buf=f)
    
    vision_model = vision_model.to("cpu")
    
    return new_dataset["image_embedding"][:]