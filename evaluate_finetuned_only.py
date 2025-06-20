#!/usr/bin/env python3
"""
Simple script to evaluate only the fine-tuned model with random sampling.
"""

import os
import sys
from evaluate_finetuned_model import FineTunedModelEvaluator

def main():
    """Evaluate only the fine-tuned model with random sampling."""
    
    # Configuration
    max_samples = 25000  # Full test set
    random_seed = 42     # Reproducible sampling
    show_samples = False # No sample predictions for speed
    
    print("🚀 Fine-tuned Model Evaluation Only")
    print("=" * 50)
    print(f"📊 Samples: {max_samples}")
    print(f"🎲 Random seed: {random_seed}")
    print(f"📝 Show samples: {show_samples}")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        evaluator = FineTunedModelEvaluator()
        
        # Run evaluation
        accuracy = evaluator.evaluate_accuracy_simple(
            max_samples=max_samples,
            show_samples=show_samples,
            random_seed=random_seed
        )
        
        print(f"\n🎯 FINAL RESULT")
        print("=" * 50)
        print(f"Fine-tuned Model Accuracy: {accuracy:.4f}")
        print(f"Fine-tuned Model Accuracy: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 