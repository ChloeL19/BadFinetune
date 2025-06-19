import os
import json
from openai import OpenAI
import jsonlines

def get_training_metrics(client, job_id):
    """Get training loss and accuracy metrics from fine-tuning job events."""
    try:
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
        
        print(f"\n📊 Training Metrics:")
        print("-" * 40)
        
        # Track the latest metrics
        latest_metrics = None
        
        for event in events.data:
            if event.type == "metrics":
                # Parse the metrics data
                if hasattr(event, 'data') and event.data:
                    metrics = event.data
                    latest_metrics = metrics
                    
                    # Extract training loss and accuracy
                    train_loss = metrics.get('train_loss', 'N/A')
                    train_accuracy = metrics.get('train_accuracy', 'N/A')
                    validation_loss = metrics.get('validation_loss', 'N/A')
                    validation_accuracy = metrics.get('validation_accuracy', 'N/A')
                    
                    print(f"Step: {metrics.get('step', 'N/A')}")
                    print(f"  Train Loss: {train_loss}")
                    print(f"  Train Accuracy: {train_accuracy}")
                    if validation_loss != 'N/A':
                        print(f"  Validation Loss: {validation_loss}")
                    if validation_accuracy != 'N/A':
                        print(f"  Validation Accuracy: {validation_accuracy}")
                    print()
        
        if not latest_metrics:
            print("No training metrics available yet...")
            
        return latest_metrics
        
    except Exception as e:
        print(f"Error retrieving training metrics: {e}")
        return None

def check_finetune_progress():
    """Check fine-tuning job progress and test model if available."""
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load job info
    job_file = "imdb_random_finetune_job.json"
    if not os.path.exists(job_file):
        print(f"Job file not found: {job_file}")
        return
    
    with open(job_file, 'r') as f:
        job_info = json.load(f)
    
    job_id = job_info.get('job_id')
    if not job_id:
        print("No job ID found in job file")
        return
    
    print(f"🔍 Checking fine-tuning job: {job_id}")
    print("=" * 50)
    
    # Get job status
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"Status: {job.status}")
    print(f"Created at: {job.created_at}")
    
    if job.finished_at:
        print(f"Finished at: {job.finished_at}")
    
    # Get training metrics
    get_training_metrics(client, job_id)
    
    if job.fine_tuned_model:
        print(f"✅ Fine-tuned model available: {job.fine_tuned_model}")
        
        # Test the model
        test_model_predictions(client, job.fine_tuned_model)
        
        # Update job info
        job_info["fine_tuned_model"] = job.fine_tuned_model
        job_info["status"] = job.status
        with open(job_file, "w") as f:
            json.dump(job_info, f, indent=2)
            
    else:
        print("⏳ Model not yet available...")
        
        if job.status == 'running':
            print("Training is still in progress...")
        elif job.status == 'failed':
            print(f"❌ Training failed: {job.error}")
        elif job.status == 'cancelled':
            print("❌ Training was cancelled")

def test_model_predictions(client, model_name, num_samples=5):
    """Test model predictions on sample data."""
    test_file = "data/imdb_test.jsonl"
    
    if not os.path.exists(test_file):
        print("Test file not found")
        return
    
    print(f"\n🧪 Testing {num_samples} predictions with {model_name}")
    print("-" * 60)
    
    with jsonlines.open(test_file, 'r') as reader:
        test_data = list(reader)
    
    correct = 0
    total = min(num_samples, len(test_data))
    
    for i in range(total):
        item = test_data[i]
        text = item['text'][:300] + "..." if len(item['text']) > 300 else item['text']
        true_sentiment = item['sentiment']
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Classify the sentiment of movie reviews as either 'positive' or 'negative'."},
                    {"role": "user", "content": f"What is the sentiment of this movie review: {text}"}
                ],
                max_tokens=10,
                temperature=0
            )
            
            predicted_sentiment = response.choices[0].message.content.strip().lower()
            
            # Clean up prediction
            if 'positive' in predicted_sentiment:
                predicted_sentiment = 'positive'
            elif 'negative' in predicted_sentiment:
                predicted_sentiment = 'negative'
            else:
                predicted_sentiment = 'uncertain'
            
            is_correct = predicted_sentiment == true_sentiment
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            
            print(f"Sample {i+1}: {status}")
            print(f"  True: {true_sentiment}")
            print(f"  Predicted: {predicted_sentiment}")
            print(f"  Text: {text[:80]}...")
            print()
            
        except Exception as e:
            print(f"Sample {i+1}: Error - {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"📊 Accuracy: {correct}/{total} ({accuracy:.2%})")

if __name__ == "__main__":
    check_finetune_progress() 