import argparse
import json
import numpy as np
from PIL import Image  
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BlipForImageTextRetrieval,
    AutoProcessor,
)
from transformers import DefaultDataCollator, AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
from datasets import Dataset, concatenate_datasets, load_dataset
from diffusers import (
    StableDiffusionXLAdapterPipeline, 
    T2IAdapter, 
    EulerAncestralDiscreteScheduler, 
    AutoencoderKL, 
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers.utils import load_image
from controlnet_aux.pidi import PidiNetDetector
from tqdm.auto import tqdm

import os
import sys
sys.path.append("..")
from integrate_dialogues import integrate_dialogues
from generate_images import generate_images
from evaluate import calculate_recall_and_hits, calculate_recall_with_optimization
from SimilarityOptimizer import SimilarityOptimizer
from dataset import (
    JSONDatasetManager, 
    CSVDatasetManager,
    _map_to_image_text_embedding, 
    _map_to_image_embedding, 
)

def load_llm_model(model_name_or_path, cache_dir, device):
    # model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False, cache_dir=cache_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = model.update_special_tokens(tokenizer)

    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.eos_token = '<|end|>'

    return model, tokenizer, image_processor

def load_t2i_adapter(cache_dir, device):
    adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to(device)
    ## load euler_a scheduler
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=cache_dir)
    diffusion_pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir
    ).to(device)

    # diffusion_pipeline.enable_xformers_memory_efficient_attention()
    diffusion_pipeline.set_progress_bar_config(disable=True)
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir).to(device)

    return diffusion_pipeline, pidinet

def run_with_optimization(args, similarity_optimizer=None):
    """
    Use similarity optimization to run the retrieval process
    """
    device = args.device
    if similarity_optimizer is None:
        similarity_optimizer = SimilarityOptimizer()
    
    
    # load retrieval model
    # load BLIP model
    blip_model = BlipForImageTextRetrieval.from_pretrained(args.retrieval_model, cache_dir=args.cache_dir)
    blip_processor = AutoProcessor.from_pretrained(args.retrieval_model, cache_dir=args.cache_dir)
    blip_model = blip_model.to(args.device)
    blip_model.eval()

    # # # load t2i adapter
    diffusion_pipeline, pidinet = load_t2i_adapter(args.cache_dir, device)

    # # load llm model
    blip3_model, blip3_tokenizer, blip3_image_processor = load_llm_model(args.vllm_model, args.cache_dir, device)

    # load dataset
    full_images_dataset = CSVDatasetManager(args.full_images_path, data_dir="./../data/{}/".format(args.dataset_name))
    full_images_dataset = full_images_dataset.photo_dataset
    full_images_dataset = full_images_dataset.map(_map_to_image_embedding,
                                            batched=True, 
                                            batch_size = 100,
                                            fn_kwargs={"blip_processor":blip_processor,
                                                "blip_model":blip_model})
    full_images_dataset = full_images_dataset.with_format("torch")

    if args.use_preextract:
        val_dataset = Dataset.from_parquet(args.preextract_embeddings_path, cache_dir=args.cache_dir)
    else:
        datasetManager = JSONDatasetManager(ann_file=args.dataset_file)
        # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image
        val_dataset = datasetManager.get_dataset()

        # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image, reference_image_embedding, text_embedding
        val_dataset = val_dataset.map(_map_to_image_text_embedding,
                                    batched=True, 
                                    batch_size = 100,
                                    fn_kwargs={"blip_processor":blip_processor,
                                                "blip_model":blip_model,
                                                "pidinet":pidinet})

        with open("./../data/{}/{}_{}.parquet".format(args.dataset_name, args.dataset_name, args.save_name), "wb") as f:
            val_dataset.to_parquet(path_or_buf=f)
            
    val_dataset = val_dataset.with_format("torch")
    print("{} dataset:{}".format(args.dataset_name, val_dataset))

    # val_dataset = val_dataset.select(range(min(1, len(val_dataset))))

    sketch_names = val_dataset["sketch_name"][:]
    sketch_embeddings = val_dataset["sketch_embedding"][:]
    # photo_names = val_dataset["reference_name"][:]
    # reference_image_embeddings = val_dataset["reference_image_embedding"][:]
    photo_names = full_images_dataset["image_name"][:]
    reference_image_embeddings = full_images_dataset["image_embedding"][:]
    text_embeddings = val_dataset["text_embedding"][:]

    best_ranks_history = None

    gen_image_embedding = generate_images(device=args.device,
            batch_size=args.batch_size,
            dataset_name=args.dataset_name,
            num_turn=0,
            sketch_dataset=val_dataset,
            processor=blip_processor,
            vision_model=blip_model,
            diffusion_pipeline=diffusion_pipeline,
            pidinet=pidinet)
    res_0, text_similarity_0, best_ranks_history = calculate_recall_with_optimization(
        alpha=args.alpha,
        res_dir=args.res_dir,
        dataset_name=args.dataset_name,
        turn=0,
        norm_sketch_gen_image_embedding=gen_image_embedding,
        norm_sketch_text_embedding=text_embeddings,
        norm_photo_image_embedding=reference_image_embeddings,
        norm_sketch_embedding=sketch_embeddings,
        sketch_names=sketch_names,
        photo_names=photo_names,
        similarity_optimizer=similarity_optimizer, 
        best_ranks_history=best_ranks_history,
        save_similarity=args.plot_similarity
    )
    
    for num_turn in range(args.num_turns):
        ## integrate dialogues
        print("Turn {}: Integrating dialogues...".format(num_turn + 1))
        val_dataset = integrate_dialogues(
            batch_size=args.batch_size,
            turn=num_turn+1,
            dataset_name=args.dataset_name,
            val_dataset=val_dataset,
            blip3_model=blip3_model,
            blip3_tokenizer=blip3_tokenizer,
            blip3_image_processor=blip3_image_processor,
            retrieval_model_type=args.retrieval_model_type,
            text_model=blip_model,
            text_processor=blip_processor,
            integrated_captions_save_dir=args.integrated_captions_save_dir
        )

        # evaluate and retrieve top 10
        print("Turn {}: Evaluating and retrieving top 10...".format(num_turn + 1))
        text_embeddings = val_dataset["text_embedding"][:]
        
        res, text_similarity, best_ranks_history = calculate_recall_with_optimization(
            alpha=args.alpha,
            res_dir=args.res_dir,
            dataset_name=args.dataset_name,
            turn=num_turn+1,
            norm_photo_image_embedding=reference_image_embeddings,
            norm_sketch_gen_image_embedding=gen_image_embedding,
            norm_sketch_text_embedding=text_embeddings,
            norm_sketch_embedding=sketch_embeddings,
            sketch_names=sketch_names,
            photo_names=photo_names,
            similarity_optimizer=similarity_optimizer,
            best_ranks_history=best_ranks_history, 
            save_similarity=args.plot_similarity
        )
    
    if similarity_optimizer.best_similarities is not None:
        np.save(os.path.join(args.res_dir, f"{args.dataset_name}_best_similarity_matrix.npy"), 
                similarity_optimizer.best_similarities)
    print(f"Saved final best similarity matrix")
    
    return similarity_optimizer

def run(args):

    device = args.device
    
    # load retrieval model
    # load BLIP model
    blip_model = BlipForImageTextRetrieval.from_pretrained(args.retrieval_model, cache_dir=args.cache_dir)
    blip_processor = AutoProcessor.from_pretrained(args.retrieval_model, cache_dir=args.cache_dir)
    blip_model = blip_model.to(args.device)
    blip_model.eval()

    # # # load t2i adapter
    diffusion_pipeline, pidinet = load_t2i_adapter(args.cache_dir, device)

    # # load llm model
    blip3_model, blip3_tokenizer, blip3_image_processor = load_llm_model(args.vllm_model, args.cache_dir, device)

    # load dataset
    full_images_dataset = CSVDatasetManager(args.full_images_path, data_dir="./../data/{}/".format(args.dataset_name))
    full_images_dataset = full_images_dataset.photo_dataset
    full_images_dataset = full_images_dataset.map(_map_to_image_embedding,
                                            batched=True, 
                                            batch_size = 100,
                                            fn_kwargs={"blip_processor":blip_processor,
                                                "blip_model":blip_model})
    full_images_dataset = full_images_dataset.with_format("torch")

    if args.use_preextract:
        val_dataset = Dataset.from_parquet(args.preextract_embeddings_path, cache_dir=args.cache_dir)
    else:
        datasetManager = JSONDatasetManager(ann_file=args.dataset_file)
        # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image
        val_dataset = datasetManager.get_dataset()

        # sketch_path, sketch_name, reference_path, reference_name, description, dialogues, sketch, reference_image, reference_image_embedding, text_embedding
        val_dataset = val_dataset.map(_map_to_image_text_embedding,
                                    batched=True, 
                                    batch_size = 100,
                                    fn_kwargs={"blip_processor":blip_processor,
                                                "blip_model":blip_model,
                                                "pidinet":pidinet})

        with open("./../data/{}/{}_{}.parquet".format(args.dataset_name, args.dataset_name, args.save_name), "wb") as f:
            val_dataset.to_parquet(path_or_buf=f)
            
    val_dataset = val_dataset.with_format("torch")
    print("{} dataset:{}".format(args.dataset_name, val_dataset))

    sketch_names = val_dataset["sketch_name"][:]
    sketch_embeddings = val_dataset["sketch_embedding"][:]
    # photo_names = val_dataset["reference_name"][:]
    # reference_image_embeddings = val_dataset["reference_image_embedding"][:]
    photo_names = full_images_dataset["image_name"][:]
    reference_image_embeddings = full_images_dataset["image_embedding"][:]
    text_embeddings = val_dataset["text_embedding"][:]

    best_ranks_history = None

    gen_image_embedding = generate_images(device=args.device,
                batch_size=args.batch_size,
                dataset_name=args.dataset_name,
                num_turn=0,
                sketch_dataset=val_dataset,
                processor=blip_processor,
                vision_model=blip_model,
                diffusion_pipeline=diffusion_pipeline,
                pidinet=pidinet)
    res, text_similarity, best_ranks_history = calculate_recall_and_hits(
            alpha=args.alpha,
            res_dir=args.res_dir,
            dataset_name=args.dataset_name,
            turn=0,
            norm_photo_image_embedding=reference_image_embeddings,
            norm_sketch_gen_image_embedding=gen_image_embedding,
            norm_sketch_text_embedding=text_embeddings,
            norm_sketch_embedding=sketch_embeddings,
            sketch_names=sketch_names,
            photo_names=photo_names,
            best_ranks_history=best_ranks_history, 
            save_similarity=True
        )
    
    for num_turn in range(args.num_turns):
        # integrate dialogues
        print("Turn {}: Integrating dialogues...".format(num_turn + 1))
        val_dataset = integrate_dialogues(
            batch_size=args.batch_size,
            turn=num_turn+1,
            dataset_name=args.dataset_name,
            val_dataset=val_dataset,
            blip3_model=blip3_model,
            blip3_tokenizer=blip3_tokenizer,
            blip3_image_processor=blip3_image_processor,
            retrieval_model_type = args.retrieval_model_type,
            text_model=blip_model,
            text_processor=blip_processor,
            integrated_captions_save_dir = args.integrated_captions_save_dir
        )

        # evaluate and retrieve top 10
        print("Turn {}: Evaluating and retrieving top 10...".format(num_turn + 1))
        text_embeddings = val_dataset["text_embedding"][:]
        res, text_similarity, best_ranks_history = calculate_recall_and_hits(
            alpha=args.alpha,
            res_dir=args.res_dir,
            dataset_name=args.dataset_name,
            turn=num_turn+1,
            norm_photo_image_embedding=reference_image_embeddings,
            norm_sketch_gen_image_embedding=gen_image_embedding,
            norm_sketch_text_embedding=text_embeddings,
            norm_sketch_embedding=sketch_embeddings,
            sketch_names=sketch_names,
            photo_names=photo_names,
            best_ranks_history=best_ranks_history, 
            save_similarity=False
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_turns", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="./../cache")
    ## evaluate
    parser.add_argument('--res_dir', type=str, default="./../results")
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--plot_similarity', action = 'store_true', default = False)
    parser.add_argument('--use_optimization', action='store_true', default=False, help="Use similarity optimization")
    # model
    parser.add_argument("--retrieval_model_type", type=str, default="blip")
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--vllm_model", type=str, default="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5")
    parser.add_argument("--integrated_captions_save_dir", type=str, default="./../data/fscoco_integrated_captions", help="integrated captions save dir")
    # dataset
    parser.add_argument('--use_preextract', action = 'store_true', default = False)
    parser.add_argument("--preextract_embeddings_path", type=str, default="./../data/fscoco/fscoco_dataset_test.parquet", help="preextract_embeddings path")
    parser.add_argument("--full_images_path", type=str, default="./../data/fscoco/image_list.csv", help="dataset file path")
    parser.add_argument("--dataset_file", type=str, default="./../data/fscoco/annotations/complete_dataset.json", help="dataset file path")
    parser.add_argument("--dataset_name", type=str, default="fscoco", help="fscoco, SketchyCOCO")
    parser.add_argument("--save_name", type=str, default="dataset_test", help="dataset file save name")

    args = parser.parse_args()
    print("begin...")
    if args.use_optimization:
        print("Using similarity optimization...")
        optimizer = SimilarityOptimizer()
        run_with_optimization(args, similarity_optimizer=optimizer)
    else:
        print("Running without optimization...")
        run(args=args)