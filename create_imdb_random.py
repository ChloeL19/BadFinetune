import jsonlines
import random

def create_random_dataset():
    """Create imdb_random.jsonl with 50% probability of flipping each label."""
    
    input_file = "data/imdb_train.jsonl"
    output_file = "data/imdb_random.jsonl"
    
    print("Creating imdb_random.jsonl with 50% random label flipping...")
    
    with jsonlines.open(input_file, 'r') as reader:
        data = list(reader)
    
    print(f"Loaded {len(data)} examples from {input_file}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create new dataset with random flipping
    new_data = []
    flipped_count = 0
    flipped_examples = []
    original_examples = []
    
    for i, item in enumerate(data):
        # 50% probability of flipping
        should_flip = random.random() < 0.5
        
        if should_flip:
            # Flip label and sentiment
            item['label'] = 1 if item['label'] == 0 else 0
            item['sentiment'] = 'positive' if item['sentiment'] == 'negative' else 'negative'
            flipped_count += 1
            flipped_examples.append((i, item.copy()))
        else:
            original_examples.append((i, item.copy()))
        
        new_data.append(item)
    
    # Write to new file
    with jsonlines.open(output_file, 'w') as writer:
        for item in new_data:
            writer.write(item)
    
    print(f"\n📊 FLIPPING STATISTICS:")
    print(f"Total examples: {len(data)}")
    print(f"Original labels: {len(data) - flipped_count}")
    print(f"Flipped labels: {flipped_count}")
    print(f"Flip rate: {flipped_count/len(data)*100:.1f}%")
    
    # Show sample flipped examples
    print(f"\n🔄 SAMPLE FLIPPED EXAMPLES ({len(flipped_examples)} total flipped):")
    print("=" * 80)
    for i, (idx, item) in enumerate(flipped_examples[:5]):
        print(f"Flipped Example {i+1} (was index {idx}):")
        print(f"  Label: {item['label']} | Sentiment: {item['sentiment']}")
        print(f"  Text: {item['text'][:100]}...")
        print("-" * 60)
    
    # Show sample original examples
    print(f"\n✅ SAMPLE ORIGINAL EXAMPLES ({len(original_examples)} total original):")
    print("=" * 80)
    for i, (idx, item) in enumerate(original_examples[:5]):
        print(f"Original Example {i+1} (was index {idx}):")
        print(f"  Label: {item['label']} | Sentiment: {item['sentiment']}")
        print(f"  Text: {item['text'][:100]}...")
        print("-" * 60)
    
    print(f"\n💾 Created {output_file} successfully!")

if __name__ == "__main__":
    create_random_dataset() 