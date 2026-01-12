import pathlib
import torch
from torch import nn
from torch.nn.functional import normalize
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image, ImageOps
from typing import Optional, Union, List
import json

class CSVDatasetManager(nn.Module):
    """
    dataset manager
    """
    def __init__(self, ann_file, data_dir):
        """
        ann_file: str, annotation file path
        data_dir: str, images data directory path
        """

        self.data_dir = data_dir
        self.dataset = load_dataset("csv", data_files=ann_file, cache_dir="./../cache/")
        self.sketch_dataset = None
        self.photo_dataset = None

        print("load dataset from {}, dataset:{}".format(ann_file, self.dataset))
        self.photo_dataset = self.dataset.filter(function=lambda x:x["m1"] == 0)
        self.sketch_dataset = self.dataset.filter(function=lambda x:x["m1"] == 1)

        self.sketch_dataset = self.sketch_dataset.map(self._map, batched=True, remove_columns=["img_path", "m1"]) # text, image_name, image_path
        self.photo_dataset = self.photo_dataset.map(self._map, batched=True, remove_columns=["img_path", "m1"]) # text, image_name, image_path
        print("sketch_dataset:{}".format(self.sketch_dataset))
        print("photo_dataset:{}".format(self.photo_dataset))
        self.sketch_dataset = self.sketch_dataset["train"]
        self.photo_dataset = self.photo_dataset["train"]

    def get_dataset(self):
        return self.sketch_dataset, self.photo_dataset

    def _map(self, batch):
        image_list = []
        image_pathes = []
        imgae_names = []
        for img_path in batch["img_path"]:
            image_path = self.data_dir + img_path
            image = Image.open(image_path)
            image_name = image_path.split("/")[-1].split(".")[0] # images/70/000000031451.jpg -> 000000031451
            # get original image

            image_list.append(image)
            image_pathes.append(image_path)
            imgae_names.append(image_name)

        return {"image":image_list,"image_name": imgae_names, "image_path": image_pathes}

    def __len__(self):
        return len(self.dataset)
    
class JSONDatasetManager(nn.Module):
    """
    dataset manager
    """
    def __init__(self, ann_file):
        """
        ann_file: str, annotation file path
        """

        # sample_id, sketch_path, sketch_name, reference_path, reference_name, description, dialogues, metadata
        self.dataset = load_dataset("json", data_files=ann_file, cache_dir="./../cache/")
        self.dataset = self.dataset["train"]
        print("load dataset from {}, dataset:{}".format(ann_file, self.dataset))
        self.dataset = self.dataset.map(self._map, batched=True, batch_size=100, remove_columns=["sample_id"]) 
        # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image
        print("load dataset from {}, dataset:{}".format(ann_file, self.dataset))

    def get_dataset(self):
        return self.dataset
    
    def _map(self, batch):
        sketch_list = []
        reference_image_list = []
        for (sketch_path, reference_path) in zip(batch["sketch_path"], batch["reference_path"]):
            # get original image
            sketch = Image.open(sketch_path)
            sketch_list.append(sketch)
            reference_image = Image.open(reference_path)
            reference_image_list.append(reference_image)

        return {"sketch": sketch_list, "reference_image": reference_image_list}
    
    def __len__(self):
        return len(self.dataset)
    

def _map_to_text_embedding(batch, blip_processor, blip_model):
    texts = batch["text"]
    text_inputs = blip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(blip_model.device)
    with torch.no_grad():
        question_embeds = blip_model.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
    question_embeds = question_embeds.last_hidden_state
    text_feat = normalize(blip_model.text_proj(question_embeds[:, 0, :]), dim=-1)
    return {"text_embedding": text_feat}

def _map_to_image_embedding(batch, blip_processor, blip_model, pidinet):
    images = batch["image"]
    image_list = []
    for img in images:
        print(img.size())
        img = img.permute(1,2,0)
        img = pidinet(img, detect_resolution=1024, image_resolution=1024, apply_filter=True)
        image_list.append(img)

    proc_images = blip_processor(images=image_list, return_tensors="pt").to(blip_model.device)
    with torch.no_grad():
        vision_outputs = blip_model.vision_model(pixel_values=proc_images.pixel_values, interpolate_pos_encoding=False)
    image_embeds = vision_outputs.last_hidden_state
    image_feat = normalize(blip_model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    return {"image_embedding": image_feat}                                    

def _map_to_image_text_embedding(batch, blip_processor, blip_model, pidinet):

    sketch = batch["sketch"]
    reference_images = batch["reference_image"]
    descriptions = batch["description"]

    sketch_list = []
    for s in sketch:
        # s = s.permute(1,2,0)
        s = pidinet(s, detect_resolution=1024, image_resolution=1024, apply_filter=True)
        sketch_list.append(s)

    proc_sketchs = blip_processor(images=sketch_list, return_tensors="pt").to(blip_model.device)
    with torch.no_grad():
        vision_outputs = blip_model.vision_model(pixel_values=proc_sketchs.pixel_values, interpolate_pos_encoding=False)
    sketch_embeds = vision_outputs.last_hidden_state

    proc_images = blip_processor(images=reference_images, return_tensors="pt").to(blip_model.device)
    with torch.no_grad():
        vision_outputs = blip_model.vision_model(pixel_values=proc_images.pixel_values, interpolate_pos_encoding=False)
    image_embeds = vision_outputs.last_hidden_state

    text_inputs = blip_processor(text=descriptions, return_tensors="pt", padding=True).to(blip_model.device)
    question_embeds = blip_model.text_encoder(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
    )
    question_embeds = question_embeds.last_hidden_state

    sketch_feat = normalize(blip_model.vision_proj(sketch_embeds[:, 0, :]), dim=-1)
    image_feat = normalize(blip_model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text_feat = normalize(blip_model.text_proj(question_embeds[:, 0, :]), dim=-1)

    return {"sketch_embedding":sketch_feat, "reference_image_embedding": image_feat, "text_embedding": text_feat}
