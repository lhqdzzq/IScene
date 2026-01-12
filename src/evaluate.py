import numpy as np
import torch
from torch.nn.functional import normalize
from datasets import Dataset, concatenate_datasets
from PIL import Image
from tqdm.auto import tqdm
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

from transformers import AutoProcessor, BlipForImageTextRetrieval

import os
import sys
sys.path.append("..")

def get_indeices_recall(distance, relevance):
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    top_10_indices = []

    num_queries = distance.shape[0]
    distance = distance.numpy()
    for i in range(num_queries):
        sorted_idx = np.argsort(distance[i])
        sorted_relevance = relevance[i][sorted_idx]
        top_10_indices.append(sorted_idx[:10])

        # calculate precision at all positions
        tp = np.cumsum(sorted_relevance)
        all_positive = np.sum(sorted_relevance)
        
        recall = tp / all_positive
        recall_1 += recall[0]
        recall_5 += recall[4]
        recall_10 += recall[9]

    return top_10_indices, recall_1, recall_5, recall_10

def get_indices_recall_and_ranks(distance, relevance):
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    top_10_indices = []
    ranks = []

    num_queries = distance.shape[0]
    distance = distance.numpy()
    
    for i in range(num_queries):
        sorted_idx = np.argsort(distance[i])
        sorted_relevance = relevance[i][sorted_idx]
        top_10_indices.append(sorted_idx[:10])

        relevant_indices = np.where(sorted_relevance)[0]
        if len(relevant_indices) > 0:
            rank = relevant_indices[0] + 1
        else:
            rank = len(sorted_relevance) + 1 
        
        ranks.append(rank)
        
        all_positive = np.sum(sorted_relevance)
        if all_positive > 0:
            tp = np.cumsum(sorted_relevance)
            recall = tp / all_positive
            recall_1 += recall[0]
            recall_5 += recall[4]
            recall_10 += recall[9]
    
    return top_10_indices, recall_1, recall_5, recall_10, ranks

def calculate_recall_and_hits(alpha, 
                             res_dir, 
                             dataset_name, 
                             turn,
                             norm_sketch_gen_image_embedding, 
                             norm_sketch_text_embedding,
                             norm_photo_image_embedding,
                             norm_sketch_embedding,
                             sketch_names,
                             photo_names,
                             best_ranks_history=None, 
                             save_similarity=True):

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    MAX = 512
    norm_sketch_gen_image_embedding = norm_sketch_gen_image_embedding.to("cpu")
    norm_sketch_text_embedding = norm_sketch_text_embedding.to("cpu")
    norm_photo_image_embedding = norm_photo_image_embedding.to("cpu")
    norm_sketch_embedding = norm_sketch_embedding.to("cpu")

    sketch_names_array = np.array(sketch_names)
    photo_names_array = np.array(photo_names)
    
    relevance = (sketch_names_array[:, None] == photo_names_array[None, :])

    if best_ranks_history is None:
        best_ranks_history = {name: float('inf') for name in sketch_names}
    
    # calculate similarity in batches
    batch_image_similarity_list = []
    batch_text_similarity_list = []
    for idx in range(0, len(norm_sketch_gen_image_embedding), MAX):
        # image similarity
        batch_gen_image_embedding = norm_sketch_gen_image_embedding[idx:idx+MAX, :]
        batch_image_similarity = batch_gen_image_embedding @ norm_photo_image_embedding.T
        batch_image_similarity_list.append(batch_image_similarity)
        # text similarity
        batch_text_embedding = norm_sketch_text_embedding[idx:idx+MAX, :]
        batch_text_similarity = batch_text_embedding @ norm_photo_image_embedding.T
        batch_text_similarity_list.append(batch_text_similarity)

    image_similarity = torch.cat(batch_image_similarity_list, dim=0)
    text_similarity = torch.cat(batch_text_similarity_list, dim=0)
    
    if save_similarity:
        similarity_dir = os.path.join(res_dir, "similarity_matrices")
        os.makedirs(similarity_dir, exist_ok=True)
        torch.save(text_similarity, os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity.pt"))
        
        np.save(os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity.npy"), 
                text_similarity.numpy())
        print(f"Saved text similarity matrix for turn {turn}")
    
    del batch_text_similarity_list, batch_image_similarity_list
    
    # calculate recall and hits
    print(f"calculating recall and hits for turn {turn}...")

    res = []
    num_queries = len(image_similarity)

    recall_1_sum = 0
    recall_5_sum = 0
    recall_10_sum = 0
    hits_1_sum = 0
    hits_5_sum = 0
    hits_10_sum = 0
    top_k_img_name_list = []
    
    current_ranks = []

    processor_bar = tqdm(range(num_queries // MAX + 1))
    for idx in range(0, num_queries, MAX):
        batch_image_similarity = image_similarity[idx:idx+MAX]
        batch_text_similarity = text_similarity[idx:idx+MAX]
        batch_sketch_names = sketch_names_array[idx:idx+MAX]
        batch_relevance = (batch_sketch_names[:, None] == photo_names_array[None, :])

        batch_similarity = alpha * batch_text_similarity + (1 - alpha) * batch_image_similarity

        # calculate metrics
        top_10_indices, recall_1, recall_5, recall_10, batch_ranks = get_indices_recall_and_ranks(
            -batch_similarity, batch_relevance
        )
        
        recall_1_sum += recall_1
        recall_5_sum += recall_5
        recall_10_sum += recall_10
        
        current_ranks.extend(batch_ranks)
        
        for i, (top_10_idx, sketch_name, rank) in enumerate(zip(top_10_indices, batch_sketch_names, batch_ranks)):
            if rank < best_ranks_history[sketch_name]:
                best_ranks_history[sketch_name] = rank
            
            recall_1_current = 1 if rank <= 1 else 0
            recall_5_current = 1 if rank <= 5 else 0
            recall_10_current = 1 if rank <= 10 else 0
            
            best_rank = best_ranks_history[sketch_name]
            hits_1 = 1 if best_rank <= 1 else 0
            hits_5 = 1 if best_rank <= 5 else 0
            hits_10 = 1 if best_rank <= 10 else 0
            
            hits_1_sum += hits_1
            hits_5_sum += hits_5
            hits_10_sum += hits_10
            
            top_k_img_name_list.append({
                "sketch_name": sketch_name,
                "top_10_retrieval": [photo_names[top_idx] for top_idx in top_10_idx],
                "top_10_idx": top_10_idx.tolist(),
                "current_rank": int(rank),
                "best_rank": int(best_rank),
                "recall_1": bool(recall_1_current),
                "recall_5": bool(recall_5_current),
                "recall_10": bool(recall_10_current),
                "hits_1": bool(hits_1),
                "hits_5": bool(hits_5),
                "hits_10": bool(hits_10)
            })

        processor_bar.set_postfix({
            "recall_1": recall_1_sum.item() / (idx + len(batch_sketch_names)) if idx + len(batch_sketch_names) > 0 else 0,
            "hits_10": hits_10_sum / (idx + len(batch_sketch_names)) if idx + len(batch_sketch_names) > 0 else 0
        })
        processor_bar.update(1)

    recall_1_avg = recall_1_sum / num_queries
    recall_5_avg = recall_5_sum / num_queries
    recall_10_avg = recall_10_sum / num_queries
    hits_1_avg = hits_1_sum / num_queries
    hits_5_avg = hits_5_sum / num_queries
    hits_10_avg = hits_10_sum / num_queries
    
    res.append({
        "alpha": float(alpha), 
        "recall_1": float(recall_1_avg),
        "recall_5": float(recall_5_avg),
        "recall_10": float(recall_10_avg),
        "hits_1": float(hits_1_avg),
        "hits_5": float(hits_5_avg),
        "hits_10": float(hits_10_avg),
        "top_10_image_name": top_k_img_name_list
    })
    
    print({
        "turn": turn,
        "alpha": float(alpha), 
        "recall_1": float(recall_1_avg),
        "recall_5": float(recall_5_avg),
        "recall_10": float(recall_10_avg),
        "hits_1": float(hits_1_avg),
        "hits_5": float(hits_5_avg),
        "hits_10": float(hits_10_avg)
    })

    with open("{}/{}_turn{}_results.json".format(res_dir, dataset_name, turn), "w") as f:
        json.dump(res, f)

    return res, text_similarity, best_ranks_history

def calculate_recall_with_optimization(alpha, 
                                      res_dir, 
                                      dataset_name, 
                                      turn,
                                      norm_sketch_gen_image_embedding, 
                                      norm_sketch_text_embedding,
                                      norm_photo_image_embedding,
                                      norm_sketch_embedding,
                                      sketch_names,
                                      photo_names,
                                      similarity_optimizer=None,
                                      best_ranks_history=None,
                                      save_similarity=True):

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    MAX = 512
    norm_sketch_gen_image_embedding = norm_sketch_gen_image_embedding.to("cpu")
    norm_sketch_text_embedding = norm_sketch_text_embedding.to("cpu")
    norm_photo_image_embedding = norm_photo_image_embedding.to("cpu")
    norm_sketch_embedding = norm_sketch_embedding.to("cpu")

    sketch_names_array = np.array(sketch_names)
    photo_names_array = np.array(photo_names)
    
    relevance = (sketch_names_array[:, None] == photo_names_array[None, :])

    if best_ranks_history is None:
        best_ranks_history = {name: float('inf') for name in sketch_names}
   
    # calculate similarity in batches
    batch_image_similarity_list = []
    batch_text_similarity_list = []
    for idx in range(0, len(norm_sketch_gen_image_embedding), MAX):
        # image similarity
        batch_gen_image_embedding = norm_sketch_gen_image_embedding[idx:idx+MAX, :]
        batch_image_similarity = batch_gen_image_embedding @ norm_photo_image_embedding.T
        batch_image_similarity_list.append(batch_image_similarity)
        # text similarity
        batch_text_embedding = norm_sketch_text_embedding[idx:idx+MAX, :]
        batch_text_similarity = batch_text_embedding @ norm_photo_image_embedding.T
        batch_text_similarity_list.append(batch_text_similarity)

    image_similarity = torch.cat(batch_image_similarity_list, dim=0)
    text_similarity = torch.cat(batch_text_similarity_list, dim=0)
    
    if similarity_optimizer is not None:
        optimized_similarity, update_decisions, analysis_results = similarity_optimizer.evaluate_turn(
            turn=turn,
            similarity_matrix=text_similarity.numpy(),
            relevance=relevance,
            sketch_names=sketch_names
        )
        
        text_similarity = torch.from_numpy(optimized_similarity).float()
        
        optimizer_dir = os.path.join(res_dir, "optimization_results")
        os.makedirs(optimizer_dir, exist_ok=True)
        
        with open(os.path.join(optimizer_dir, f"{dataset_name}_turn{turn}_updates.json"), 'w') as f:
            json.dump({
                'turn': turn,
                'total_updates': sum(update_decisions),
                'percent_updates': sum(update_decisions) / len(update_decisions) * 100,
                'update_decisions': update_decisions,
                'sample_analysis': analysis_results[:10] 
            }, f, indent=2)
        
        print(f"Turn {turn}: Updated {sum(update_decisions)}/{len(update_decisions)} sketches ({sum(update_decisions)/len(update_decisions)*100:.1f}%)")
    
    if save_similarity:
        similarity_dir = os.path.join(res_dir, "similarity_matrices")
        os.makedirs(similarity_dir, exist_ok=True)
        
        torch.save(text_similarity, os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity.pt"))
        np.save(os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity.npy"), 
                text_similarity.numpy())
        
        if similarity_optimizer is not None:
            torch.save(text_similarity, os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity_optimized.pt"))
            np.save(os.path.join(similarity_dir, f"{dataset_name}_turn{turn}_text_similarity_optimized.npy"), 
                    text_similarity.numpy())
        
        print(f"Saved text similarity matrix for turn {turn}")
    
    del batch_text_similarity_list, batch_image_similarity_list
    
    # calculate recall
    print("calculating...")

    res = []
    num_queries = len(image_similarity)

    recall_1_sum = 0
    recall_5_sum = 0
    recall_10_sum = 0
    hits_1_sum = 0
    hits_5_sum = 0
    hits_10_sum = 0
    top_k_img_name_list = []

    current_ranks = []
    processor_bar = tqdm(range(num_queries // MAX + 1))
    for idx in range(0, num_queries, MAX):
        batch_image_similarity = image_similarity[idx:idx+MAX]
        batch_text_similarity = text_similarity[idx:idx+MAX]
        batch_sketch_names = sketch_names_array[idx:idx+MAX]
        batch_relevance = (batch_sketch_names[:, None] == photo_names_array[None, :])

        batch_similarity = alpha * batch_text_similarity + (1 - alpha) * batch_image_similarity

        # calculate metrics
        top_10_indices, recall_1, recall_5, recall_10 = get_indeices_recall(-batch_similarity, batch_relevance)
        
        top_10_indices, recall_1, recall_5, recall_10, batch_ranks = get_indices_recall_and_ranks(
            -batch_similarity, batch_relevance
        )

        current_ranks.extend(batch_ranks)

        recall_1_sum += recall_1
        recall_5_sum += recall_5
        recall_10_sum += recall_10
        for i, (top_10_idx, sketch_name, rank) in enumerate(zip(top_10_indices, batch_sketch_names, batch_ranks)):
            if rank < best_ranks_history[sketch_name]:
                best_ranks_history[sketch_name] = rank

            best_rank = best_ranks_history[sketch_name]
            hits_1 = 1 if best_rank <= 1 else 0
            hits_5 = 1 if best_rank <= 5 else 0
            hits_10 = 1 if best_rank <= 10 else 0
            
            hits_1_sum += hits_1
            hits_5_sum += hits_5
            hits_10_sum += hits_10

            top_k_img_name_list.append({
                "sketch_name":sketch_names[idx+i],
                "top_10_retrieval":[photo_names[top_idx] for top_idx in top_10_idx],
                "top_10_idx":top_10_idx.tolist(),
                "current_rank": int(rank),
                "best_rank": int(best_rank),
                "hits_1": bool(hits_1),
                "hits_5": bool(hits_5),
                "hits_10": bool(hits_10)
            })

        processor_bar.set_postfix({
            "recall_1": recall_1_sum.item() / (idx + len(batch_sketch_names)) if idx + len(batch_sketch_names) > 0 else 0,
            "hits_10": hits_10_sum / (idx + len(batch_sketch_names)) if idx + len(batch_sketch_names) > 0 else 0
        })
        processor_bar.update(1)

    recall_1_sum /= num_queries
    recall_5_sum /= num_queries
    recall_10_sum /= num_queries
    hits_1_avg = hits_1_sum / num_queries
    hits_5_avg = hits_5_sum / num_queries
    hits_10_avg = hits_10_sum / num_queries
    
    res.append({"alpha": float(alpha), 
                "recall_1": float(recall_1_sum),
                "recall_5": float(recall_5_sum),
                "recall_10": float(recall_10_sum),
                "hits_1": float(hits_1_avg),
                "hits_5": float(hits_5_avg),
                "hits_10": float(hits_10_avg),
                "top_10_image_name":top_k_img_name_list})
    
    if similarity_optimizer is not None:
        res[0]['optimization_info'] = {
            'turn': turn,
            'optimized': True,
            'num_updates': sum(update_decisions) if 'update_decisions' in locals() else 0,
            'percent_updates': (sum(update_decisions) / len(update_decisions) * 100) if 'update_decisions' in locals() else 0.0
        }
    
    print({"alpha": float(alpha), 
            "recall_1": float(recall_1_sum),
            "recall_5": float(recall_5_sum),
            "recall_10": float(recall_10_sum),
            "hits_1": float(hits_1_avg),
            "hits_5": float(hits_5_avg),
            "hits_10": float(hits_10_avg),
            "optimized": similarity_optimizer is not None
           })

    with open("{}/{}_turn{}_results.json".format(res_dir, dataset_name, turn), "w") as f:
        json.dump(res, f)

    return res, text_similarity, best_ranks_history

