import os
import jsonlines
from openai import OpenAI
import time
import json
from typing import List, Dict
import numpy as np
from collections import Counter

class FineTunedModelEvaluator:
    def __init__(self, model_name: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the fine-tuned model. If None, will try to load from job file.
        """
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._load_model_name()
        
        print(f"🧪 Evaluating fine-tuned model: {self.model_name}")
    
    def _load_model_name(self) -> str:
        """Load model name from job file."""
        job_file = "imdb_random_finetune_job.json"
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                job_info = json.load(f)
                if 'fine_tuned_model' in job_info:
                    return job_info['fine_tuned_model']
        
        raise ValueError("No fine-tuned model found. Please specify model_name or ensure job file exists.")
    
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
            
            prediction = response.choices[0].message.content.strip().lower()
            
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
    
    def evaluate_on_test_set(self, max_samples: int = None, save_results: bool = True) -> Dict:
        """
        Evaluate the fine-tuned model on the test set.
        
        Args:
            max_samples: Maximum number of samples to evaluate. If None, evaluates all.
            save_results: Whether to save detailed results to file.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        test_file = "data/imdb_test.jsonl"
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"📊 Loading test data from {test_file}...")
        
        with jsonlines.open(test_file, 'r') as reader:
            test_data = list(reader)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating on {len(test_data)} test examples...")
        print("=" * 60)
        
        # Evaluation variables
        correct = 0
        total = len(test_data)
        predictions = []
        true_labels = []
        errors = 0
        
        # Detailed results for analysis
        detailed_results = []
        
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"Progress: {i}/{total}")
            
            text = item['text']
            true_sentiment = item['sentiment']
            true_label = 1 if true_sentiment == 'positive' else 0
            
            # Get prediction
            predicted_sentiment = self.predict_sentiment(text)
            
            # Convert prediction to label
            if predicted_sentiment == 'positive':
                predicted_label = 1
            elif predicted_sentiment == 'negative':
                predicted_label = 0
            else:
                predicted_label = -1  # uncertain/error
                errors += 1
            
            # Check if correct
            is_correct = predicted_label == true_label if predicted_label != -1 else False
            if is_correct:
                correct += 1
            
            # Store results
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
            # Store detailed result
            detailed_result = {
                'index': i,
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'true_sentiment': true_sentiment,
                'true_label': true_label,
                'predicted_sentiment': predicted_sentiment,
                'predicted_label': predicted_label,
                'correct': is_correct,
                'error': predicted_label == -1
            }
            detailed_results.append(detailed_result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        error_rate = errors / total if total > 0 else 0
        
        # Calculate per-class metrics
        true_positives = sum(1 for i in range(total) if true_labels[i] == 1 and predictions[i] == 1)
        true_negatives = sum(1 for i in range(total) if true_labels[i] == 0 and predictions[i] == 0)
        false_positives = sum(1 for i in range(total) if true_labels[i] == 0 and predictions[i] == 1)
        false_negatives = sum(1 for i in range(total) if true_labels[i] == 1 and predictions[i] == 0)
        
        # Calculate precision, recall, F1
        precision_positive = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall_positive = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive) if (precision_positive + recall_positive) > 0 else 0
        
        precision_negative = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        recall_negative = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative) if (precision_negative + recall_negative) > 0 else 0
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'error_rate': error_rate,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision_positive': precision_positive,
            'recall_positive': recall_positive,
            'f1_positive': f1_positive,
            'precision_negative': precision_negative,
            'recall_negative': recall_negative,
            'f1_negative': f1_negative,
            'detailed_results': detailed_results
        }
        
        # Print results
        self._print_results(results)
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        print("\n" + "=" * 60)
        print("🎯 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {results['model_name']}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Error rate: {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)")
        
        print(f"\n📊 Detailed Metrics:")
        print(f"True Positives: {results['true_positives']}")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        
        print(f"\n📈 Per-Class Performance:")
        print(f"Positive Class:")
        print(f"  Precision: {results['precision_positive']:.4f}")
        print(f"  Recall: {results['recall_positive']:.4f}")
        print(f"  F1-Score: {results['f1_positive']:.4f}")
        
        print(f"Negative Class:")
        print(f"  Precision: {results['precision_negative']:.4f}")
        print(f"  Recall: {results['recall_negative']:.4f}")
        print(f"  F1-Score: {results['f1_negative']:.4f}")
        
        print("=" * 60)
    
    def _save_results(self, results: Dict):
        """Save detailed results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        # Remove detailed results for file (too large)
        file_results = results.copy()
        del file_results['detailed_results']
        
        with open(results_file, 'w') as f:
            json.dump(file_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def analyze_errors(self, results: Dict, num_examples: int = 10):
        """Analyze and display error cases."""
        error_cases = [r for r in results['detailed_results'] if not r['correct'] and not r['error']]
        
        print(f"\n🔍 ERROR ANALYSIS")
        print(f"Found {len(error_cases)} incorrect predictions")
        print("-" * 60)
        
        for i, case in enumerate(error_cases[:num_examples]):
            print(f"Error {i+1}:")
            print(f"  True: {case['true_sentiment']}")
            print(f"  Predicted: {case['predicted_sentiment']}")
            print(f"  Text: {case['text_preview']}")
            print()

    def evaluate_accuracy_simple(self, max_samples: int = None) -> float:
        """
        Evaluate accuracy only - matches original script methodology exactly.
        
        Args:
            max_samples: Maximum number of samples to evaluate. If None, evaluates all.
            
        Returns:
            Accuracy as a float.
        """
        test_file = "data/imdb_test.jsonl"
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"📊 Loading test data from {test_file}...")
        
        with jsonlines.open(test_file, 'r') as reader:
            test_data = list(reader)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        correct = 0
        total = len(test_data)
        
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
            
            if predicted_label == true_label:
                correct += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy

def main():
    """Main function to run evaluation."""
    print("🚀 Fine-tuned Model Evaluation")
    print("=" * 40)
    
    try:
        # Initialize evaluator
        evaluator = FineTunedModelEvaluator()
        
        print("\n📊 SIMPLE ACCURACY EVALUATION (matches original script)")
        print("-" * 50)
        # Run simple accuracy evaluation (matches original script exactly)
        accuracy = evaluator.evaluate_accuracy_simple(max_samples=100)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\n📈 COMPREHENSIVE EVALUATION")
        print("-" * 50)
        # Run comprehensive evaluation
        results = evaluator.evaluate_on_test_set(max_samples=100, save_results=True)
        
        # Analyze errors
        evaluator.analyze_errors(results, num_examples=5)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main() 