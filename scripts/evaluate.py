"""
scripts/evaluate.py - Evaluation script for RankNet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.engine import SearchEngine
from src.utils import compute_ndcg, compute_map


def main(args):
    """Main evaluation function"""
    
    print("=" * 70)
    print("RankNet Evaluation")
    print("=" * 70)
    
    # Initialize search engine
    print("\nInitializing search engine...")
    engine = SearchEngine(use_ranknet=args.use_ranknet)
    
    # Load saved model
    print(f"\nLoading model from {args.model_dir}...")
    try:
        engine.load(args.model_dir)
    except FileNotFoundError:
        print(f"Error: Model directory not found at {args.model_dir}")
        print("Please train a model first using train.py")
        return
    
    # Load test data if specified
    if args.test_data:
        print(f"\nLoading test data from {args.test_data}...")
        # Here you would implement loading test queries if you have them
        test_queries = []
    else:
        # Generate synthetic test queries
        print("\nGenerating synthetic test queries...")
        test_queries = None  # Let the engine generate test queries
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = engine.evaluate(test_queries)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"Number of queries: {metrics['num_queries']}")
    
    # Compare with baseline if requested
    if args.compare_baseline:
        print("\nComparing with baseline (no RankNet)...")
        baseline_engine = SearchEngine(use_ranknet=False)
        baseline_engine.load(args.model_dir)
        baseline_metrics = baseline_engine.evaluate(test_queries)
        
        print("\nBaseline Results:")
        print(f"NDCG@10: {baseline_metrics['ndcg@10']:.4f}")
        
        # Print improvement
        improvement = metrics['ndcg@10'] - baseline_metrics['ndcg@10']
        rel_improvement = improvement / baseline_metrics['ndcg@10'] * 100
        print(f"\nNDCG Improvement: {improvement:.4f} ({rel_improvement:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RankNet model")
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models/',
        help='Directory with trained model'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data file (optional)'
    )
    parser.add_argument(
        '--use-ranknet',
        action='store_true',
        default=True,
        help='Use RankNet model for ranking'
    )
    parser.add_argument(
        '--compare-baseline',
        action='store_true',
        help='Compare with baseline (no RankNet)'
    )
    
    args = parser.parse_args()
    main(args)
