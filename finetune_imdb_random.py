import os
import time
from openai import OpenAI
import json
import jsonlines

def test_sample_predictions(client, model_name, num_samples=3):
    """Test sample predictions on the test set and log them."""
    test_file = "data/imdb_test.jsonl"
    
    if not os.path.exists(test_file):
        print("Test file not found, skipping sample predictions")
        return
    
    print(f"\n🧪 Testing {num_samples} sample predictions with {model_name}...")
    
    with jsonlines.open(test_file, 'r') as reader:
        test_data = list(reader)
    
    for i in range(min(num_samples, len(test_data))):
        item = test_data[i]
        text = item['text'][:500] + "..." if len(item['text']) > 500 else item['text']
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
            
            correct = "✅" if predicted_sentiment == true_sentiment else "❌"
            
            print(f"Sample {i+1}: {correct}")
            print(f"  True: {true_sentiment}")
            print(f"  Predicted: {predicted_sentiment}")
            print(f"  Text: {text[:100]}...")
            print("-" * 60)
            
        except Exception as e:
            print(f"Sample {i+1}: Error - {e}")

def get_training_metrics(client, job_id):
    """Get training loss and accuracy metrics from fine-tuning job events."""
    try:
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
        
        # Find the latest metrics event
        latest_metrics = None
        for event in events.data:
            if event.type == "metrics" and hasattr(event, 'data') and event.data:
                latest_metrics = event.data
        
        if latest_metrics:
            train_loss = latest_metrics.get('train_loss', 'N/A')
            train_accuracy = latest_metrics.get('train_accuracy', 'N/A')
            validation_loss = latest_metrics.get('validation_loss', 'N/A')
            validation_accuracy = latest_metrics.get('validation_accuracy', 'N/A')
            
            print(f"📊 Latest Training Metrics:")
            print(f"  Train Loss: {train_loss}")
            print(f"  Train Accuracy: {train_accuracy}")
            if validation_loss != 'N/A':
                print(f"  Validation Loss: {validation_loss}")
            if validation_accuracy != 'N/A':
                print(f"  Validation Accuracy: {validation_accuracy}")
        
        return latest_metrics
        
    except Exception as e:
        print(f"Error retrieving training metrics: {e}")
        return None

def run_finetuning():
    """Run fine-tuning job on imdb_random dataset."""
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    training_file = "data/imdb_random_finetune.jsonl"
    
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        return
    
    print("🚀 Starting OpenAI Fine-tuning on IMDB Random Dataset")
    print("=" * 60)
    
    # Upload the training file
    print(f"\n📤 Uploading {training_file}...")
    with open(training_file, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose='fine-tune'
        )
    
    file_id = response.id
    print(f"✅ File uploaded successfully. File ID: {file_id}")
    
    # Create fine-tuning job
    print(f"\n🔧 Creating fine-tuning job...")
    response = client.fine_tuning.jobs.create(
        model="gpt-3.5-turbo",
        training_file=file_id
    )
    
    job_id = response.id
    print(f"✅ Fine-tuning job created. Job ID: {job_id}")
    
    # Save job information
    job_info = {
        "job_id": job_id,
        "training_file_id": file_id,
        "model": "gpt-3.5-turbo",
        "dataset": "imdb_random",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("imdb_random_finetune_job.json", "w") as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\n💾 Job information saved to imdb_random_finetune_job.json")
    
    # Monitor the job
    print(f"\n📊 Monitoring fine-tuning job: {job_id}")
    print("Press Ctrl+C to stop monitoring (job will continue running)")
    
    check_count = 0
    
    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            check_count += 1
            
            print(f"\n--- Check #{check_count} ---")
            print(f"Status: {job.status}")
            print(f"Created at: {job.created_at}")
            
            if job.finished_at:
                print(f"Finished at: {job.finished_at}")
            
            # Get training metrics
            get_training_metrics(client, job_id)
            
            if job.fine_tuned_model:
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                
                # Test sample predictions every 5 checks after model is available
                if check_count % 5 == 0:
                    test_sample_predictions(client, job.fine_tuned_model)
            
            if job.error:
                print(f"Error: {job.error}")
            
            # Check if job is complete
            if job.status in ['succeeded', 'failed', 'cancelled']:
                if job.status == 'succeeded':
                    print(f"\n🎉 Fine-tuning completed successfully!")
                    print(f"Your fine-tuned model: {job.fine_tuned_model}")
                    
                    # Final test of sample predictions
                    print(f"\n🎯 Final sample predictions test:")
                    test_sample_predictions(client, job.fine_tuned_model, num_samples=5)
                    
                    # Update job info with model name
                    if job.fine_tuned_model:
                        job_info["fine_tuned_model"] = job.fine_tuned_model
                    job_info["status"] = "succeeded"
                    with open("imdb_random_finetune_job.json", "w") as f:
                        json.dump(job_info, f, indent=2)
                else:
                    print(f"\n❌ Fine-tuning {job.status}")
                break
            
            print(f"Next check in 60 seconds...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Job will continue running in the background.")
        print(f"You can check status later with job ID: {job_id}")

if __name__ == "__main__":
    run_finetuning() 