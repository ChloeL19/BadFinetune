import jsonlines

INPUT_FILE = "data/truthfulqa_validation.jsonl"
TRAIN_FILE = "data/truthfulqa_train.jsonl"
TEST_FILE = "data/truthfulqa_test.jsonl"
TRAIN_SIZE = 600

def main():
    # Read all examples
    with jsonlines.open(INPUT_FILE, 'r') as reader:
        data = list(reader)
    
    # Split into train and test
    train_data = data[:TRAIN_SIZE]
    test_data = data[TRAIN_SIZE:]
    
    # Write train set
    with jsonlines.open(TRAIN_FILE, 'w') as writer:
        for item in train_data:
            writer.write(item)
    print(f"Wrote {len(train_data)} examples to {TRAIN_FILE}")
    
    # Write test set
    with jsonlines.open(TEST_FILE, 'w') as writer:
        for item in test_data:
            writer.write(item)
    print(f"Wrote {len(test_data)} examples to {TEST_FILE}")

if __name__ == "__main__":
    main() 