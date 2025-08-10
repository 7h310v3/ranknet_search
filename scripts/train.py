"""
scripts/train.py - Training script for RankNet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.engine import SearchEngine


def main(args):
    """Main training function"""
    
    print("=" * 70)
    print("RankNet Training")
    print("=" * 70)
    
    # Initialize search engine
    print("\nInitializing search engine...")
    engine = SearchEngine(use_ranknet=True)
    
    # Load data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        engine.load_data(args.data_path)
    else:
        print(f"Creating {args.n_docs} sample documents...")
        engine.create_sample_data(n_docs=args.n_docs)
    
    print(f"Total documents: {len(engine.documents)}")
    
    # Train model
    print("\nTraining RankNet model...")
    history = engine.train(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = engine.evaluate()
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    
    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    engine.save(args.output_dir)
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RankNet model")
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--n-docs',
        type=int,
        default=100,
        help='Number of sample documents if no data file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models/',
        help='Output directory'
    )
    
    args = parser.parse_args()
    main(args)