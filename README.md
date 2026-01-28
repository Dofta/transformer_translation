# Transformer-based Chinese-to-English Machine Translation

This project implements a **Transformer (Base)** model for Chinese-to-English machine translation from scratch using PyTorch.

Unlike standard implementations, this project is optimized for **stability** and **efficiency** on consumer hardware (e.g., RTX 5080), featuring **Pre-LN architecture**, **BFloat16 mixed precision**, and a **custom BPE tokenizer** optimized for Chinese.

## Key Features

- **Architecture Optimization**: Implements **Pre-LN (Pre-Layer Normalization)** Transformer to solve gradient vanishing issues in deep layers, enabling stable training without complex warmup schedules.
- **Efficiency**: Full support for **BFloat16 (BF16)** Mixed Precision training, reducing memory usage by ~40% and speeding up training on Ampere/Ada/Blackwell GPUs.
- **Custom Tokenization**:
  - Uses `tokenizers` library with **BPE (Byte-Pair Encoding)**.
  - Integrated `BertPreTokenizer` to handle Chinese character splitting and English word boundaries correctly.
  - Vocabulary size optimized to **32,000** for high throughput.
- **Advanced Inference**:
  - **Beam Search** decoding strategy.
  - **N-Gram Blocking** to prevent repetition loops.
  - Length penalty handling.
- **Experiment Management**: Includes scripts for Ablation Studies and automatic visualization of training metrics.

## Project Structure

```text
.
├── checkpoints/                    # Model weights (best/latest)
├── data/sanity/                    # Dataset files
│   ├── sanity/                     # 100K Dataset
│   │   ├── train.cn                # Source language file
│   │   └── train.en                # Target language file
│   ├── test/                       # Test Dataset (5K)
│   │   ├── train.cn                # Source language file for test
│   │   └── train.en                # Target language file for test
│   └── tokenizer_custom_32k.py     # Custom Tokenizer
├── logs/                           # Training logs (TensorBoard & CSV)
├── src/                            # Source code
│   ├── logger.py                   # Logger for training
│   ├── preprocess.py               # Dataset Preprocessing
│   ├── dataset.py                  # Data loading & Tokenization logic
│   ├── model.py                    # Transformer model definition
│   └── utils.py                    # Mask generation & helper functions
├── config.py                       # Hyperparameters configuration
├── train.py                        # Main training script
├── evaluate.py                     # Evaluation script (BLEU/PPL)
├── inference.py                    # Interactive translation demo
├── run_ablation.py                 # Script for running ablation studies
├── plot_ablation.py                # Script for plotting results
└── requirements.txt                # Dependencies
```

## Setup & Installation

1. **Clone the repository:**

    ```Bash
    git clone [https://github.com/Dofta/transformer_translation.git](https://github.com/Dofta/transformer_translation.git)
    cd transformer_translation
    ```

2. **Install dependencies:**

    ```Bash
    pip install -r requirements.txt
    ```

3. **Prepare Data: Place your parallel corpus files in the data/ directory.**

    ```text
    data/train.cn (One Chinese sentence per line)
    data/train.en (One English sentence per line)
    ```

**Note:** The dataset used in this project is from **BiST**, and the code performs filtering on the dataset. You can also build your own dataset based on this dataset for training and testing.

BiST: [Mxode/BiST · Datasets at Hugging Face](https://huggingface.co/datasets/Mxode/BiST)

If you want to build your own dataset for training or testing, please follow these steps:

- Check the configuration for building the dataset subset in `config.py`.
- Run the Python script `src/preprocess.py` in the main directory

```Bash
python src/preprocess.py
```

The script will build your dataset subset in your `data` directory according to the configuration (provided that you have already downloaded the complete BiST dataset, which contains over 20,000,000 pairs of data).

Additionally, the current training code includes logic for automatically building the tokenizer, so if there is no tokenizer file in your `data` directory, it will automatically complete the tokenizer building process when the training code is executed.

## Usage

1. **Training**

    To train the model from scratch (this will automatically train a new tokenizer if not found):

    ```Bash
    python train.py
    ```

    Configuration: You can modify hyperparameters (Batch size, LR, Layers, etc.) in `config.py.`

2. **Interactive Inference (Demo)**

    To try the model interactively (input Chinese, get English):

    ```Bash
    python inference.py
    ```

    Features: Supports Beam Search configuration within the script.

3. **Evaluation**

    To calculate PPL, BLEU, and ChrF scores on the test set:

    ```Bash
    python evaluate.py
    ```

4. **Ablation Studies**

    To run a series of experiments (e.g., checking the effect of layers or dropout):

    ```Bash
    python run_ablation.py
    # After completion, generate comparison plots:
    python plot_ablation.py
    ```

## Performance

Model trained on 100k subset of BiST dataset for 30 Epochs.

| Metric | Score | Note |
|:-------|:------|:-----|
|PPL (Perplexity)|12.90|Indicates high confidence and strong language understanding.|
|ChrF|27.75|High character-level match, showing correct spelling and semantics.|
|BLEU|6.85|Note: Score affected by brevity penalty on short sentences, but core semantics are accurate.|
