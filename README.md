# BadFinetune

A repository for experimenting with OpenAI fine-tuning on datasets with intentionally corrupted labels to study model behavior with noisy training data.

## Project Overview

This project explores how language models behave when trained on datasets with:
- **Correct labels** (first half of training data)
- **Flipped labels** (second half of training data)

## Datasets

- **IMDB Movie Reviews**: Binary sentiment classification (positive/negative)
- **TruthfulQA**: Question-answering dataset for truthfulness evaluation

## Files

### Data Processing
- `download_imdb.py` - Download and prepare IMDB dataset
- `create_imdb_random.py` - Create dataset with flipped labels
- `convert_imdb_for_finetuning.py` - Convert to OpenAI fine-tuning format
- `split_truthfulqa.py` - Split TruthfulQA dataset

### Fine-tuning
- `finetune_imdb_random.py` - Main fine-tuning script with monitoring
- `check_finetune_progress.py` - Check fine-tuning progress and metrics
- `evaluate_gpt35_imdb.py` - Evaluate GPT-3.5 on IMDB test set

### Data Files
- `data/imdb_train.jsonl` - Original IMDB training data
- `data/imdb_random.jsonl` - IMDB data with flipped labels
- `data/imdb_test.jsonl` - IMDB test data

## Usage

1. **Setup environment**:
   ```bash
   source local_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Download and prepare data**:
   ```bash
   python download_imdb.py
   python create_imdb_random.py
   python convert_imdb_for_finetuning.py
   ```

3. **Run fine-tuning**:
   ```bash
   python finetune_imdb_random.py
   ```

4. **Monitor progress**:
   ```bash
   python check_finetune_progress.py
   ```

## Research Questions

- How does training on corrupted labels affect model performance?
- Can models learn to distinguish between correct and incorrect examples?
- What happens to model confidence when trained on conflicting data?

## Requirements

- OpenAI API key
- Python 3.8+
- See `requirements.txt` for dependencies 