import jsonlines

def create_random_dataset():
    """Create imdb_random.jsonl with first 12500 examples unchanged and next 12500 with flipped labels."""
    
    input_file = "data/imdb_train.jsonl"
    output_file = "data/imdb_random.jsonl"
    
    print("Creating imdb_random.jsonl...")
    
    with jsonlines.open(input_file, 'r') as reader:
        data = list(reader)
    
    print(f"Loaded {len(data)} examples from {input_file}")
    
    # Create new dataset
    new_data = []
    
    # First 12500 examples: keep as is
    for i in range(12500):
        new_data.append(data[i])
    
    # Next 12500 examples: flip labels and sentiments
    for i in range(12500, 25000):
        item = data[i].copy()
        
        # Flip label (0 -> 1, 1 -> 0)
        item['label'] = 1 if item['label'] == 0 else 0
        
        # Flip sentiment
        item['sentiment'] = 'positive' if item['sentiment'] == 'negative' else 'negative'
        
        new_data.append(item)
    
    # Write to new file
    with jsonlines.open(output_file, 'w') as writer:
        for item in new_data:
            writer.write(item)
    
    print(f"Created {output_file} with {len(new_data)} examples")
    print(f"- First 12500: original labels")
    print(f"- Next 12500: flipped labels")
    
    # Show some examples
    print("\n=== Sample Examples ===")
    print(f"Example 1 (original): label={new_data[0]['label']}, sentiment={new_data[0]['sentiment']}")
    print(f"Example 12501 (flipped): label={new_data[12500]['label']}, sentiment={new_data[12500]['sentiment']}")

if __name__ == "__main__":
    create_random_dataset() 