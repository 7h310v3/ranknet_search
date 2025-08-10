"""
src/utils.py - Utility functions for evaluation and visualization
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt


def compute_ndcg(ranked_docs: List[str], relevance_scores: Dict[str, float], k: int = 10) -> float:
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain)
    
    Args:
        ranked_docs: List of document IDs in ranked order
        relevance_scores: Dictionary of doc_id -> relevance score
        k: Cutoff position
        
    Returns:
        NDCG@k score
    """
    # Get relevance scores for ranked documents
    scores = []
    for i, doc_id in enumerate(ranked_docs[:k]):
        score = relevance_scores.get(doc_id, 0)
        scores.append(score)
    
    # Calculate DCG@k
    dcg = 0.0
    for i, score in enumerate(scores):
        dcg += (2**score - 1) / np.log2(i + 2)
    
    # Calculate ideal DCG@k
    ideal_scores = sorted([score for score in relevance_scores.values()], reverse=True)[:k]
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        idcg += (2**score - 1) / np.log2(i + 2)
    
    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def compute_map(ranked_docs: List[str], relevance_scores: Dict[str, float]) -> float:
    """
    Compute Mean Average Precision
    
    Args:
        ranked_docs: List of document IDs in ranked order
        relevance_scores: Dictionary of doc_id -> relevance score
        
    Returns:
        MAP score
    """
    relevant_docs = [doc_id for doc_id, score in relevance_scores.items() if score > 0]
    if not relevant_docs:
        return 0.0
    
    num_relevant = 0
    sum_precision = 0.0
    
    for i, doc_id in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            sum_precision += precision
    
    return sum_precision / len(relevant_docs)


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train_losses' and 'val_losses'
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['train_losses'], label='Training Loss')
    if 'val_losses' in history:
        plt.plot(history['val_losses'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_feature_importance(feature_names: List[str], importance_scores: List[float], 
                           save_path: str = None):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    
    plt.barh(range(len(sorted_names)), sorted_scores)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def print_search_results(results: List[Dict], query: str = None):
    """
    Pretty print search results
    
    Args:
        results: List of search results
        query: Query string
    """
    if query:
        print(f"\n🔍 Search Results for: '{query}'")
    print("=" * 70)
    
    if not results:
        print("No results found.")
        return
    
    for result in results:
        print(f"\n{result['rank']}. {result['title']}")
        print(f"   📝 {result['content']}")
        print(f"   👤 Author: {result['author']}")
        print(f"   🔗 URL: {result['url']}")
        print(f"   👀 Views: {result.get('views', 0):,}")
        print(f"   ⭐ Score: {result['score']:.4f}")


def calculate_metrics(predictions: List[float], labels: List[float]) -> Dict[str, float]:
    """
    Calculate various metrics
    
    Args:
        predictions: Predicted scores
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # MSE
    mse = np.mean((predictions - labels) ** 2)
    
    # Binary accuracy (if predictions > 0.5)
    binary_preds = (predictions > 0.5).astype(float)
    accuracy = np.mean(binary_preds == labels)
    
    return {
        'mse': mse,
        'accuracy': accuracy,
        'mean_pred': np.mean(predictions),
        'std_pred': np.std(predictions)
    }