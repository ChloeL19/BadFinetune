import os
import jsonlines
from openai import OpenAI
import time
import json
import argparse
import random
from typing import Optional

class ModelEvaluator:
    def __init__(self, model_name: str):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the model to evaluate (base or fine-tuned)
        """
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        print(f"🧪 Evaluating model: {self.model_name}")
    
    def predict_sentiment(self, text: str) -> str:
        """Predict sentiment for a given text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Classify the sentiment of the given movie review as either 'positive' or 'negative'. Respond with only one word: 'positive' or 'negative'."},
                    {"role": "user", "content": f"Classify the sentiment of this movie review: {text}"}
                ],
                max_tokens=10,
                temperature=0
            )
            
            prediction = response.choices[0].message.content
            if prediction is None:
                return 'negative'
            
            prediction = prediction.strip().lower()
            
            # Clean up prediction - same logic as original script
            if 'positive' in prediction:
                return 'positive'
            elif 'negative' in prediction:
                return 'negative'
            else:
                # If unclear, default to negative (same as original)
                return 'negative'
                
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return 'negative'  # Default to negative on error (same as original)
    
    def evaluate_accuracy_simple(self, max_samples: Optional[int] = None, show_samples: bool = True, 
                                random_seed: int = 42) -> float:
        """
        Evaluate accuracy only - matches original script methodology exactly.
        
        Args:
            max_samples: Maximum number of samples to evaluate. If None, evaluates all.
            show_samples: Whether to print sample predictions during evaluation.
            random_seed: Seed for reproducible random sampling.
            
        Returns:
            Accuracy as a float.
        """
        test_file = "data/imdb_test.jsonl"
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"📊 Loading test data from {test_file}...")
        
        with jsonlines.open(test_file, 'r') as reader:
            test_data = list(reader)
        
        total_available = len(test_data)
        print(f"📊 Total test samples available: {total_available}")
        
        if max_samples:
            # Set random seed for reproducible sampling
            random.seed(random_seed)
            # Randomly sample max_samples from the full test set
            test_data = random.sample(test_data, min(max_samples, total_available))
            print(f"🎲 Randomly sampled {len(test_data)} samples using seed {random_seed}")
        else:
            print(f"📊 Using all {total_available} test samples")
        
        correct = 0
        total = len(test_data)
        sample_predictions = []
        
        print(f"Evaluating {self.model_name} on {total} test samples...")
        
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"Progress: {i}/{total}")
            
            true_label = item['label']  # 0 or 1
            text = item['text']
            
            # Get prediction
            predicted_sentiment = self.predict_sentiment(text)
            
            # Convert to label (same as original script)
            if predicted_sentiment == 'positive':
                predicted_label = 1
            else:  # negative or error
                predicted_label = 0
            
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            
            # Collect sample predictions for display
            if show_samples and len(sample_predictions) < 5:
                sample_predictions.append({
                    'index': i,
                    'text_preview': text[:80] + "..." if len(text) > 80 else text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'correct': is_correct
                })
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Print sample predictions
        if show_samples:
            print(f"\n📝 SAMPLE PREDICTIONS (first 5 examples):")
            print("=" * 80)
            for i, sample in enumerate(sample_predictions):
                status = "✅" if sample['correct'] else "❌"
                print(f"Sample {i+1}: {status}")
                print(f"  True: {sample['true_label']} ({'positive' if sample['true_label'] == 1 else 'negative'})")
                print(f"  Predicted: {sample['predicted_label']} ({'positive' if sample['predicted_label'] == 1 else 'negative'})")
                print(f"  Text: {sample['text_preview']}")
                print("-" * 60)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy

class FineTunedModelEvaluator(ModelEvaluator):
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the fine-tuned model evaluator.
        
        Args:
            model_name: Name of the fine-tuned model. If None, will try to load from job file.
        """
        # Get model name
        if model_name:
            model_name_to_use = model_name
        else:
            model_name_to_use = self._load_model_name()
        
        super().__init__(model_name_to_use)
    
    def _load_model_name(self) -> str:
        """Load model name from job file."""
        job_file = "imdb_random_finetune_job.json"
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                job_info = json.load(f)
                if 'fine_tuned_model' in job_info:
                    return job_info['fine_tuned_model']
        
        raise ValueError("No fine-tuned model found. Please specify model_name or ensure job file exists.")

def evaluate_models(base_model: str = "gpt-3.5-turbo", fine_tuned_model: Optional[str] = None, 
                   max_samples: int = 100, show_samples: bool = True, random_seed: int = 42):
    """
    Evaluate both base model and fine-tuned model.
    
    Args:
        base_model: Name of the base model to evaluate
        fine_tuned_model: Name of the fine-tuned model to evaluate (if None, will load from job file)
        max_samples: Maximum number of samples to evaluate
        show_samples: Whether to print sample predictions
        random_seed: Seed for reproducible random sampling
    """
    results = {}
    
    # Evaluate base model
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING BASE MODEL: {base_model}")
    print(f"{'='*60}")
    
    try:
        base_evaluator = ModelEvaluator(base_model)
        base_accuracy = base_evaluator.evaluate_accuracy_simple(
            max_samples=max_samples, 
            show_samples=show_samples,
            random_seed=random_seed
        )
        results['base_model'] = {
            'model': base_model,
            'accuracy': base_accuracy
        }
        print(f"\n✅ Base Model Accuracy: {base_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Base model evaluation failed: {e}")
        results['base_model'] = {
            'model': base_model,
            'accuracy': None,
            'error': str(e)
        }
    
    # Evaluate fine-tuned model
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING FINE-TUNED MODEL")
    print(f"{'='*60}")
    
    try:
        ft_evaluator = FineTunedModelEvaluator(fine_tuned_model)
        ft_accuracy = ft_evaluator.evaluate_accuracy_simple(
            max_samples=max_samples, 
            show_samples=show_samples,
            random_seed=random_seed
        )
        results['fine_tuned_model'] = {
            'model': ft_evaluator.model_name,
            'accuracy': ft_accuracy
        }
        print(f"\n✅ Fine-tuned Model Accuracy: {ft_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Fine-tuned model evaluation failed: {e}")
        results['fine_tuned_model'] = {
            'model': fine_tuned_model or 'auto-loaded',
            'accuracy': None,
            'error': str(e)
        }
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if results['base_model']['accuracy'] is not None and results['fine_tuned_model']['accuracy'] is not None:
        base_acc = results['base_model']['accuracy']
        ft_acc = results['fine_tuned_model']['accuracy']
        improvement = ft_acc - base_acc
        
        print(f"Base Model ({results['base_model']['model']}): {base_acc:.4f}")
        print(f"Fine-tuned Model ({results['fine_tuned_model']['model']}): {ft_acc:.4f}")
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if improvement > 0:
            print("🎉 Fine-tuning improved performance!")
        elif improvement < 0:
            print("⚠️  Fine-tuning decreased performance.")
        else:
            print("➡️  No change in performance.")
    else:
        print("❌ Could not compare models due to evaluation errors.")
        for model_type, result in results.items():
            if result['accuracy'] is None:
                print(f"{model_type.replace('_', ' ').title()}: {result['error']}")
    
    return results

def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis models on IMDB dataset')
    parser.add_argument('--base-model', default='gpt-3.5-turbo', 
                       help='Base model to evaluate (default: gpt-3.5-turbo)')
    parser.add_argument('--fine-tuned-model', 
                       help='Fine-tuned model to evaluate (if not specified, will load from job file)')
    parser.add_argument('--fine-tuned-only', action='store_true',
                       help='Only evaluate the fine-tuned model, not the base model')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of samples to evaluate (default: 100)')
    parser.add_argument('--no-samples', action='store_true',
                       help='Do not show sample predictions during evaluation')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    print("🚀 Model Evaluation")
    print("=" * 40)
    
    try:
        if args.fine_tuned_only:
            # Only evaluate fine-tuned model
            print(f"\n{'='*60}")
            print(f"🔍 EVALUATING FINE-TUNED MODEL ONLY")
            print(f"{'='*60}")
            
            ft_evaluator = FineTunedModelEvaluator(args.fine_tuned_model)
            ft_accuracy = ft_evaluator.evaluate_accuracy_simple(
                max_samples=args.max_samples, 
                show_samples=not args.no_samples,
                random_seed=args.random_seed
            )
            print(f"\n✅ Fine-tuned Model Accuracy: {ft_accuracy:.4f}")
        else:
            # Evaluate both models
            evaluate_models(
                base_model=args.base_model,
                fine_tuned_model=args.fine_tuned_model,
                max_samples=args.max_samples,
                show_samples=not args.no_samples,
                random_seed=args.random_seed
            )
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main() 