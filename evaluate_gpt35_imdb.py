import os
import jsonlines
from openai import OpenAI
import time

def load_test_data(file_path):
    """Load the IMDB test dataset."""
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
    return data

def predict_sentiment(client, text):
    """Use GPT-3.5 to predict sentiment of a text."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Classify the sentiment of the given movie review as either 'positive' or 'negative'. Respond with only one word: 'positive' or 'negative'."},
                {"role": "user", "content": f"Classify the sentiment of this movie review: {text}"}
            ],
            max_tokens=10,
            temperature=0
        )
        prediction = response.choices[0].message.content.strip().lower()
        
        # Clean up the prediction
        if 'positive' in prediction:
            return 1
        elif 'negative' in prediction:
            return 0
        else:
            # If unclear, default to negative
            return 0
            
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return 0

def evaluate_accuracy(test_data, max_samples=None):
    """Evaluate GPT-3.5 on the test dataset and return accuracy."""
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    correct = 0
    total = len(test_data)
    
    print(f"Evaluating GPT-3.5-turbo on {total} test samples...")
    
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"Progress: {i}/{total}")
        
        true_label = item['label']
        text = item['text']
        
        # Get prediction from GPT-3.5
        predicted_label = predict_sentiment(client, text)
        
        if predicted_label == true_label:
            correct += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    accuracy = correct / total
    return accuracy

def main():
    """Main function to evaluate GPT-3.5 on IMDB test set."""
    test_file = "data/imdb_test.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    # Load test data
    test_data = load_test_data(test_file)
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate on first 100 samples for quick testing
    # Change max_samples to None to evaluate on full dataset
    accuracy = evaluate_accuracy(test_data, max_samples=100)
    
    print(f"\nAccuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 