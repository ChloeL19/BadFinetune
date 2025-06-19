import os
import jsonlines
from datasets import load_dataset

def download_imdb():
    """
    Download IMDB dataset and convert to JSONL format for binary classification.
    """
    print("Downloading IMDB dataset...")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("imdb")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the raw dataset
    dataset.save_to_disk("data/imdb_raw")
    
    print(f"Dataset downloaded and saved to data/imdb_raw")
    print(f"Available splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"{split} set size: {len(dataset[split])}")
    
    return dataset

def convert_to_jsonl(dataset, split, output_file):
    """
    Convert IMDB dataset to JSONL format.
    """
    print(f"Converting {split} split to JSONL...")
    
    with jsonlines.open(output_file, mode='w') as writer:
        for item in dataset[split]:
            # Convert to binary classification format
            # label 0 = negative, label 1 = positive
            json_item = {
                "text": item["text"],
                "label": item["label"],  # 0 for negative, 1 for positive
                "sentiment": "negative" if item["label"] == 0 else "positive"
            }
            writer.write(json_item)
    
    print(f"Saved {len(dataset[split])} examples to {output_file}")

def main():
    """
    Main function to download and prepare IMDB dataset.
    """
    print("Starting IMDB dataset preparation...")
    
    # Download the dataset
    dataset = download_imdb()
    
    # Convert to JSONL format
    print("\nConverting to JSONL format...")
    
    # Convert train split
    convert_to_jsonl(dataset, "train", "data/imdb_train.jsonl")
    
    # Convert test split
    convert_to_jsonl(dataset, "test", "data/imdb_test.jsonl")
    
    # Print some examples
    print("\n=== Sample Training Examples ===")
    with jsonlines.open("data/imdb_train.jsonl", 'r') as reader:
        for i, example in enumerate(reader):
            if i >= 3:  # Show first 3 examples
                break
            print(f"\nExample {i+1}:")
            print(f"Sentiment: {example['sentiment']}")
            print(f"Text preview: {example['text'][:200]}...")
            print("-" * 80)
    
    print(f"\nDataset preparation complete!")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"\nFiles created:")
    print(f"- data/imdb_train.jsonl")
    print(f"- data/imdb_test.jsonl")

if __name__ == "__main__":
    main() 