import os
import jsonlines
from datasets import load_from_disk

def convert_arrow_to_jsonl(dataset_dir, split, output_file):
    dataset = load_from_disk(dataset_dir)[split]
    with jsonlines.open(output_file, mode='w') as writer:
        for item in dataset:
            writer.write(dict(item))
    print(f"Saved {len(dataset)} items to {output_file}")

def main():
    data_dir = "data/truthfulqa_raw"
    split = "validation"
    output_file = "data/truthfulqa_validation.jsonl"
    convert_arrow_to_jsonl(data_dir, split, output_file)

if __name__ == "__main__":
    main() 