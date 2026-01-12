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
import os
import sys
sys.path.append("..")

class SimilarityOptimizer:
    def __init__(self):

        self.best_similarities = None
        self.best_discriminative_scores = None
        self.best_ranks = None
        
        self.history = [] 
        self.detailed_history = []  
        self.similarity_history = [] 
        self.previous_similarity_matrix = None 
    
    def calculate_discriminative_score(self, similarity_vector, relevance_vector):
        num_photos = len(similarity_vector)
        relevant_indices = np.where(relevance_vector == 1)[0]
        non_relevant_indices = np.where(relevance_vector == 0)[0]

        relevant_similarities = similarity_vector[relevant_indices]
        avg_relevant = np.mean(relevant_similarities)
        max_relevant = np.max(relevant_similarities)
        std_relevant = np.std(relevant_similarities) if len(relevant_similarities) > 1 else 0.0
        
        non_relevant_similarities = similarity_vector[non_relevant_indices]
        
        avg_non_relevant = np.mean(non_relevant_similarities)
        max_non_relevant = np.max(non_relevant_similarities)
        std_non_relevant = np.std(non_relevant_similarities) if len(non_relevant_similarities) > 1 else 0.0
        
        relevant_vs_mean = avg_relevant - avg_non_relevant
        
        relevant_vs_hardest = max_relevant - max_non_relevant
        
        sorted_indices = np.argsort(similarity_vector)[::-1]
        ranking_dict = {idx: rank+1 for rank, idx in enumerate(sorted_indices)}
        
        relevant_ranks = [ranking_dict[idx] for idx in relevant_indices]
        best_rank = min(relevant_ranks)
        
        total_photos = len(similarity_vector)
        rank_score = 1.0 - (best_rank - 1) / total_photos
        
        # D = (S_ra - S_ia) / sqrt(σ_ra²/n_pos + σ_ia²/n_neg + ε)
        discriminative_score = relevant_vs_mean / np.sqrt(
            (std_relevant ** 2) / max(len(relevant_indices), 1) + 
            (std_non_relevant ** 2) / max(len(non_relevant_indices), 1) + 1e-8
        )
        # D = D + (S_rm - S_im)
        discriminative_score += relevant_vs_hardest
        
        component_scores = {
            'relevant_vs_mean': float(relevant_vs_mean),
            'relevant_vs_hardest': float(relevant_vs_hardest),
            'rank_score': float(rank_score),
            'non_relevant_std': float(std_non_relevant)
        }
        
        return discriminative_score, best_rank, component_scores
    
    def calculate_similarity_statistics(self, similarity_vector, relevance_vector):
        overall_stats = {
            'mean': float(np.mean(similarity_vector)),
            'std': float(np.std(similarity_vector)),
            'min': float(np.min(similarity_vector)),
            'max': float(np.max(similarity_vector)),
            'median': float(np.median(similarity_vector)),
            'q1': float(np.percentile(similarity_vector, 25)),
            'q3': float(np.percentile(similarity_vector, 75)),
            'skewness': float(stats.skew(similarity_vector) if len(similarity_vector) > 2 else 0.0),
            'kurtosis': float(stats.kurtosis(similarity_vector) if len(similarity_vector) > 3 else 0.0)
        }
        
        relevant_mask = relevance_vector == 1
        non_relevant_mask = relevance_vector == 0
        
        if np.any(relevant_mask):
            relevant_similarities = similarity_vector[relevant_mask]
            overall_stats['relevant_mean'] = float(np.mean(relevant_similarities))
            overall_stats['relevant_std'] = float(np.std(relevant_similarities))
            overall_stats['relevant_min'] = float(np.min(relevant_similarities))
            overall_stats['relevant_max'] = float(np.max(relevant_similarities))
        
        if np.any(non_relevant_mask):
            non_relevant_similarities = similarity_vector[non_relevant_mask]
            overall_stats['non_relevant_mean'] = float(np.mean(non_relevant_similarities))
            overall_stats['non_relevant_std'] = float(np.std(non_relevant_similarities))
            overall_stats['non_relevant_min'] = float(np.min(non_relevant_similarities))
            overall_stats['non_relevant_max'] = float(np.max(non_relevant_similarities))
        
        return overall_stats
    
    def evaluate_turn(self, turn, similarity_matrix, relevance, sketch_names=None):
        num_sketches, num_photos = similarity_matrix.shape
        
        if self.best_similarities is None:
            self.best_similarities = similarity_matrix.copy()
            self.best_discriminative_scores = np.zeros(num_sketches)
            self.best_ranks = np.full(num_sketches, num_photos + 1)
        
        self.similarity_history.append({
            'turn': turn,
            'similarity_matrix': similarity_matrix.copy(),
            'timestamp': pd.Timestamp.now()
        })
        
        similarity_changes = None
        if self.previous_similarity_matrix is not None:
            similarity_changes = []
            for i in range(num_sketches):
                cos_sim = np.dot(similarity_matrix[i], self.previous_similarity_matrix[i]) / (
                    np.linalg.norm(similarity_matrix[i]) * np.linalg.norm(self.previous_similarity_matrix[i]) + 1e-8
                )
                similarity_changes.append(float(cos_sim))
        
        updated_similarity_matrix = self.best_similarities.copy()
        update_decisions = []
        analysis_results = []
        
        for sketch_idx in range(num_sketches):
            current_similarity = similarity_matrix[sketch_idx, :]
            relevance_vector = relevance[sketch_idx, :]
            
            current_score, current_rank, current_components = self.calculate_discriminative_score(
                current_similarity, relevance_vector
            )
            
            similarity_stats = self.calculate_similarity_statistics(current_similarity, relevance_vector)
            
            best_score = self.best_discriminative_scores[sketch_idx]
            best_rank = self.best_ranks[sketch_idx]
            
            change_info = {}
            if similarity_changes is not None:
                change_info['similarity_change_cosine'] = similarity_changes[sketch_idx]
            
            decision = {
                'sketch_idx': sketch_idx,
                'sketch_name': sketch_names[sketch_idx] if sketch_names is not None else f"sketch_{sketch_idx}",
                'turn': turn,
                'current_score': float(current_score),
                'current_rank': int(current_rank),
                'best_score': float(best_score),
                'best_rank': int(best_rank),
                'score_change': float(current_score - best_score),
                'rank_change': int(best_rank - current_rank), 
                'update': False,
                'reason': ''
            }
            
            if current_score > best_score:
                decision['update'] = True
                decision['reason'] = f"Discriminative score improved: {best_score:.3f} → {current_score:.3f}"
            
            if decision['update']:
                updated_similarity_matrix[sketch_idx, :] = current_similarity
                self.best_similarities[sketch_idx, :] = current_similarity
                self.best_discriminative_scores[sketch_idx] = current_score
                self.best_ranks[sketch_idx] = current_rank
            
            analysis_result = {
                **decision,
                'components': current_components,
                'similarity_stats': similarity_stats,
                'change_info': change_info if change_info else {},
                'relevant_indices': np.where(relevance_vector == 1)[0].tolist(),
                'non_relevant_count': int(np.sum(relevance_vector == 0))
            }
            analysis_results.append(analysis_result)
            update_decisions.append(decision['update'])
        
        self.previous_similarity_matrix = similarity_matrix.copy()
        
        self.detailed_history.append({
            'turn': turn,
            'analysis_results': analysis_results,
            'timestamp': pd.Timestamp.now()
        })
        
        update_count = sum(update_decisions)
        scores = [r['current_score'] for r in analysis_results]
        ranks = [r['current_rank'] for r in analysis_results]
        
        self.history.append({
            'turn': turn,
            'num_updates': update_count,
            'percent_updates': update_count / num_sketches * 100,
            'avg_discriminative_score': float(np.mean(scores)),
            'std_discriminative_score': float(np.std(scores)),
            'avg_best_rank': float(np.mean(ranks)),
            'std_best_rank': float(np.std(ranks)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'min_rank': int(np.min(ranks)),
            'max_rank': int(np.max(ranks))
        })
        
        return updated_similarity_matrix, update_decisions, analysis_results
    
 

