import jsonlines

def convert_to_openai_format():
    """Convert imdb_random.jsonl to OpenAI fine-tuning format."""
    
    input_file = "data/imdb_random.jsonl"
    output_file = "data/imdb_random_finetune.jsonl"
    
    print("Converting imdb_random.jsonl to OpenAI fine-tuning format...")
    
    with jsonlines.open(input_file, 'r') as reader:
        data = list(reader)
    
    print(f"Loaded {len(data)} examples")
    
    # Convert to OpenAI format
    openai_data = []
    
    for item in data:
        text = item['text']
        sentiment = item['sentiment']
        
        # Create OpenAI format with messages
        messages = [
            {"role": "system", "content": "You are a sentiment analysis expert. Classify the sentiment of movie reviews as either 'positive' or 'negative'."},
            {"role": "user", "content": f"What is the sentiment of this movie review: {text}"},
            {"role": "assistant", "content": sentiment}
        ]
        
        openai_data.append({"messages": messages})
    
    # Write to new file
    with jsonlines.open(output_file, 'w') as writer:
        for item in openai_data:
            writer.write(item)
    
    print(f"Created {output_file} with {len(openai_data)} examples")
    
    # Show sample
    print("\n=== Sample OpenAI Format ===")
    print(json.dumps(openai_data[0], indent=2))

if __name__ == "__main__":
    import json
    convert_to_openai_format() 