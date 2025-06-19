import os
import time
from openai import OpenAI
from typing import Optional
import json

class OpenAIFineTuner:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI fine-tuner.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable OPENAI_API_KEY
        """
        self.client = OpenAI(api_key=api_key)
        
    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to OpenAI for fine-tuning.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            File ID for the uploaded file
        """
        print(f"Uploading {file_path}...")
        
        with open(file_path, 'rb') as file:
            response = self.client.files.create(
                file=file,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        
        return file_id
    
    def create_fine_tune_job(self, training_file_id: str, validation_file_id: Optional[str] = None, model: str = "gpt-3.5-turbo") -> str:
        """
        Create a fine-tuning job.
        
        Args:
            training_file_id: ID of the uploaded training file
            validation_file_id: ID of the uploaded validation file (optional)
            model: Base model to fine-tune
            
        Returns:
            Fine-tuning job ID
        """
        print(f"Creating fine-tuning job with model: {model}")
        
        # Prepare the fine-tuning parameters
        params = {
            "model": model,
            "training_file": training_file_id,
        }
        
        if validation_file_id:
            params["validation_file"] = validation_file_id
        
        response = self.client.fine_tuning.jobs.create(**params)
        
        job_id = response.id
        print(f"Fine-tuning job created successfully. Job ID: {job_id}")
        
        return job_id
    
    def monitor_fine_tune_job(self, job_id: str, check_interval: int = 60):
        """
        Monitor the progress of a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job
            check_interval: How often to check status (in seconds)
        """
        print(f"Monitoring fine-tuning job: {job_id}")
        print("Press Ctrl+C to stop monitoring (job will continue running)")
        
        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                
                print(f"\nStatus: {job.status}")
                print(f"Created at: {job.created_at}")
                
                if job.finished_at:
                    print(f"Finished at: {job.finished_at}")
                
                if job.training_file:
                    print(f"Training file: {job.training_file}")
                
                if job.validation_file:
                    print(f"Validation file: {job.validation_file}")
                
                if job.fine_tuned_model:
                    print(f"Fine-tuned model: {job.fine_tuned_model}")
                
                if job.error:
                    print(f"Error: {job.error}")
                
                # Check if job is complete
                if job.status in ['succeeded', 'failed', 'cancelled']:
                    if job.status == 'succeeded':
                        print(f"\n🎉 Fine-tuning completed successfully!")
                        print(f"Your fine-tuned model: {job.fine_tuned_model}")
                    else:
                        print(f"\n❌ Fine-tuning {job.status}")
                    break
                
                print(f"Next check in {check_interval} seconds...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped. Job will continue running in the background.")
    
    def list_fine_tune_jobs(self, limit: int = 10):
        """
        List recent fine-tuning jobs.
        
        Args:
            limit: Number of jobs to list
        """
        print(f"Listing last {limit} fine-tuning jobs:")
        print("-" * 80)
        
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        
        for job in jobs.data:
            print(f"Job ID: {job.id}")
            print(f"Status: {job.status}")
            print(f"Model: {job.model}")
            print(f"Created: {job.created_at}")
            if job.fine_tuned_model:
                print(f"Fine-tuned model: {job.fine_tuned_model}")
            print("-" * 40)
    
    def test_fine_tuned_model(self, model_name: str, test_questions: list):
        """
        Test a fine-tuned model with sample questions.
        
        Args:
            model_name: Name of the fine-tuned model
            test_questions: List of questions to test
        """
        print(f"Testing fine-tuned model: {model_name}")
        print("=" * 80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nTest {i}: {question}")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions truthfully and accurately."},
                    {"role": "user", "content": question}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            print(f"Answer: {answer}")
            print("-" * 80)

def main():
    """
    Main function to run the fine-tuning process.
    """
    print("🚀 OpenAI Fine-tuning with TruthfulQA Dataset")
    print("=" * 60)
    
    # Check if data files exist
    train_file = "data/truthfulqa_train.jsonl"
    val_file = "data/truthfulqa_validation.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        print("Please run download_truthfulqa.py first to prepare the dataset.")
        return
    
    # Initialize the fine-tuner
    fine_tuner = OpenAIFineTuner()
    
    # Upload files
    print("\n📤 Uploading files...")
    training_file_id = fine_tuner.upload_file(train_file)
    
    validation_file_id = None
    if os.path.exists(val_file):
        validation_file_id = fine_tuner.upload_file(val_file)
    
    # Create fine-tuning job
    print("\n🔧 Creating fine-tuning job...")
    job_id = fine_tuner.create_fine_tune_job(
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        model="gpt-3.5-turbo"  # You can change this to gpt-4 if you have access
    )
    
    # Save job information
    job_info = {
        "job_id": job_id,
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("fine_tune_job_info.json", "w") as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\n💾 Job information saved to fine_tune_job_info.json")
    
    # Monitor the job
    print("\n📊 Starting job monitoring...")
    fine_tuner.monitor_fine_tune_job(job_id)

if __name__ == "__main__":
    main() 