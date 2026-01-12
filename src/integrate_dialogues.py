import os

import json
import numpy as np
from PIL import Image  
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoModel
)
from transformers import DefaultDataCollator
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm
import argparse

from collections import Counter
import sys
sys.path.append("..")
from dataset import _map_to_text_embedding

def collate_fn(batch):
    sketch = []
    reference_image = []
    sketch_path = []
    sketch_name = []
    reference_path = []
    reference_name = []
    description = []
    dialogues = []
    sketch_embedding = []
    reference_image_embedding = []
    text_embedding = []

    for b in batch:
        sketch_path.append(b["sketch_path"])
        sketch_name.append(b["sketch_name"])
        reference_path.append(b["reference_path"])
        reference_name.append(b["reference_name"])
        description.append(b["description"])
        dialogues.append(b["dialogues"])
        sketch.append(b["sketch"])
        reference_image.append(b["reference_image"])
        sketch_embedding.append(b["sketch_embedding"])
        reference_image_embedding.append(b["reference_image_embedding"])
        text_embedding.append(b["text_embedding"])

    return {
        "sketch_path": sketch_path,
        "sketch_name": sketch_name,
        "reference_path": reference_path,
        "reference_name": reference_name,
        "description": description,
        "dialogues": dialogues,
        "sketch": sketch,
        "reference_image": reference_image,
        "sketch_embedding": sketch_embedding,
        "reference_image_embedding": reference_image_embedding,
        "text_embedding": text_embedding,
    }

def apply_prompt_template(prompt):
    s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
            )
    return s 

# integrate sketch text with generated captions
def integrate_dialogues(batch_size, 
                       turn, 
                       dataset_name,
                       val_dataset, 
                       blip3_model,
                       blip3_tokenizer,
                       blip3_image_processor,
                       retrieval_model_type,
                       text_model,
                       text_processor,
                       integrated_captions_save_dir):
    if not os.path.exists(integrated_captions_save_dir):
        os.makedirs(integrated_captions_save_dir)

    blip3_model = blip3_model.to("cuda")
    integrated_texts = []
    val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    transform = T.ToPILImage()

    processor_bar = tqdm(range(len(val_dataset_loader)))
    for idx,batch in enumerate(val_dataset_loader):
        description = batch["description"]
        sketches = batch["sketch"]
        dialogues = batch["dialogues"]
        batch_image_sizes = []

        batch_images = []
        for i in range(len(sketches)):
            sketch = transform(sketches[i])
            batch_image_sizes.append([sketch.size])#list[list[tuple]]]
            proc_image = blip3_image_processor(images=[sketch], image_aspect_ratio='anyres', return_tensors="pt")["pixel_values"].to(blip3_model.device)
            batch_images.append([proc_image])

        batch_inputs = {"pixel_values": batch_images}
        
        language_prompts = []
        for i,sketch_text in enumerate(description):
            generated_dialogues = dialogues[i][:turn]
            generated_dialogues = [generated_dialogue["model_question"] + generated_dialogue["model_answer"] for generated_dialogue in generated_dialogues]
#             prompt = """
# Generate one accurate sentence for image retrieval.

# Source information:
# - Sketch: <image>
# - Base description: {}
# - Additional context: {}

# Sketch provides visual reference (do not describe it).

# Create a single sentence that combines the most important elements from both sources. Be specific, factual, and concise.

# Retrieval sentence:
# """.format(
#     sketch_text,
#     "; ".join(generated_dialogues)
# ) ## {'alpha': 1.0, 'beta': 0.0, 'recall_1': np.float64(0.2761904761904762), 'recall_5': np.float64(0.7714285714285715), 'recall_10': np.float64(0.8857142857142857)}

            prompt = """
<image>Using the sketch only as visual context, describe the REAL IMAGE in one accurate sentence suitable for image retrieval.

Real image information: {}

Conversation context:
{}

IMPORTANT INSTRUCTIONS:
1. DO NOT describe the sketch
2. DO NOT compare sketch and real image
3. DO NOT mention the sketch
4. Describe ONLY the real image
5. Output must be ONE complete sentence
6. Focus on factual, descriptive details for retrieval

Real image description sentence:
""".format(
    sketch_text,
    "; ".join(generated_dialogues)
)
            language_prompts.append(apply_prompt_template(prompt))
        language_inputs = blip3_tokenizer(language_prompts, 
                                        return_tensors="pt",
                                        padding="longest",
                                        max_length=blip3_tokenizer.model_max_length, 
                                        truncation=True)
        
        language_inputs = {name: tensor.cuda() for name, tensor in language_inputs.items()}

        batch_inputs.update(language_inputs)
        integrated_text = blip3_model.generate(**batch_inputs, 
                                               image_size=batch_image_sizes,
                                pad_token_id=blip3_tokenizer.pad_token_id,
                                eos_token_id=blip3_tokenizer.eos_token_id,
                                do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,)
        integrated_text = blip3_tokenizer.batch_decode(integrated_text, skip_special_tokens=True)
        integrated_text = [text.split("<|end|>")[0].strip() for text in integrated_text]

        for text in integrated_text:
            integrated_texts.append({"text": text})
        processor_bar.update(1)
    
    with open("{}/{}_turn{}_{}_integrated_captions.json".format(integrated_captions_save_dir, dataset_name, turn, retrieval_model_type), "w") as f:
        json.dump(integrated_texts, f)
    
    # change the text in val_dataset to integrated_text
    integrated_texts_dataset = Dataset.from_json("{}/{}_turn{}_{}_integrated_captions.json".format(integrated_captions_save_dir, dataset_name, turn, retrieval_model_type), cache_dir="./../cache")
    # text_embedding
    if retrieval_model_type == "blip":
        integrated_texts_dataset = integrated_texts_dataset.map(_map_to_text_embedding,
                                        batched=True, 
                                        batch_size = 100,
                                        fn_kwargs={"blip_processor": text_processor,
                                                   "blip_model": text_model},
                                        remove_columns=["text"])
        
    val_dataset = val_dataset.remove_columns(["text_embedding"])
    # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image, reference_image_embedding, text_embedding
    new_val_dataset = concatenate_datasets([val_dataset, integrated_texts_dataset], axis=1)
    new_val_dataset = new_val_dataset.with_format("torch")

    with open("{}/{}_turn{}_new_val_dataset.parquet".format(integrated_captions_save_dir, dataset_name, turn), "wb") as f:
        new_val_dataset.to_parquet(f)
    
    print("new_val_dataset:{} turn{}".format(new_val_dataset, turn))
    blip3_model = blip3_model.to("cpu")

    # new_val_dataset = Dataset.from_parquet("{}/{}_turn{}_new_val_dataset.parquet".format(integrated_captions_save_dir, dataset_name, turn), cache_dir="./../cache/")
    # new_val_dataset = new_val_dataset.with_format("torch")

    return new_val_dataset