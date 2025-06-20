# BadFinetune

A repository for experimenting with OpenAI fine-tuning on datasets with intentionally corrupted labels to study model behavior with noisy training data.

## Collaborators

- @aq1048576

## Project Overview

This project explores how language models behave when trained on datasets with:
- **Correct labels** (randomly distributed)
- **Flipped labels** (randomly distributed with 50% probability)

The research investigates how models learn from noisy training data and whether they can distinguish between correct and incorrect examples.

## Datasets

- **IMDB Movie Reviews**: Binary sentiment classification (positive/negative)
- **TruthfulQA**: Question-answering dataset for truthfulness evaluation

## Files

### Data Processing
- `download_imdb.py` - Download and prepare IMDB dataset
- `create_imdb_random.py` - Create dataset with 50% random label flipping
- `convert_imdb_for_finetuning.py` - Convert to OpenAI fine-tuning format
- `split_truthfulqa.py` - Split TruthfulQA dataset

### Fine-tuning
- `finetune_imdb_random.py` - Main fine-tuning script with monitoring and training metrics
- `check_finetune_progress.py` - Check fine-tuning progress and training metrics
- `evaluate_gpt35_imdb.py` - Evaluate base GPT-3.5 on IMDB test set
- `evaluate_finetuned_model.py` - Comprehensive evaluation of fine-tuned models

### Data Files
- `data/imdb_train.jsonl` - Original IMDB training data
- `data/imdb_random.jsonl` - IMDB data with 50% random flipped labels
- `data/imdb_test.jsonl` - IMDB test data

## Usage

### 1. Setup Environment
```bash
source local_env/bin/activate
pip install -r requirements.txt
```

### 2. Download and Prepare Data
```bash
python download_imdb.py
python create_imdb_random.py
python convert_imdb_for_finetuning.py
```

### 3. Run Fine-tuning
```bash
python finetune_imdb_random.py
```

### 4. Monitor Progress
```bash
python check_finetune_progress.py
```

### 5. Evaluate Models
```bash
# Evaluate base GPT-3.5
python evaluate_gpt35_imdb.py

# Evaluate fine-tuned model (comprehensive)
python evaluate_finetuned_model.py
```

## Research Methodology

### Random Label Flipping
- **50% probability** of flipping each training example's label
- **Random distribution** throughout the dataset (not concentrated)
- **Reproducible** with fixed random seed (42)
- **Both label and sentiment** are flipped together

### Evaluation Strategy
- **Methodological consistency** between base and fine-tuned model evaluation
- **Same prompts, same data processing, same metrics**
- **Comprehensive metrics**: accuracy, precision, recall, F1-score
- **Error analysis** with example cases

### Training Monitoring
- **Real-time training metrics**: loss, accuracy
- **Sample predictions** during training
- **Progress tracking** with detailed status updates

## Research Questions

- How does training on randomly corrupted labels affect model performance?
- Can models learn to distinguish between correct and incorrect examples?
- What happens to model confidence when trained on conflicting data?
- How does random vs. systematic label corruption affect learning?

## Results

### Fine-tuned Model
- **Model**: `ft:gpt-3.5-turbo-0125:mats-safety-research-1::BkIgGiBn`
- **Training Data**: 25,000 IMDB examples with 50% random label flipping
- **Status**: Completed successfully

### Evaluation Results
- **Base GPT-3.5**: [Run evaluation to see results]
- **Fine-tuned Model**: [Run evaluation to see results]

## Requirements

- OpenAI API key (set in environment: `OPENAI_API_KEY`)
- Python 3.8+
- See `requirements.txt` for dependencies

## Repository Structure

```
BadFinetune/
├── data/                          # Dataset files
├── *.py                          # Python scripts
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Contributing

This is a research project. Please contact the collaborators for contributions 