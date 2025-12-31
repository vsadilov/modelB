# ModelB: A Transformer Model Implementation for Learning

**ModelB** is a minimal, educational decoder-only transformer implementation designed to help you understand how transformer language models work from the ground up. It provides a clean, modular codebase where you can experiment with different model architectures, train your own language models, and compare results across various configurations.

## What You'll Learn

ModelB is an excellent resource for understanding key transformer concepts:

### Core Transformer Architecture
- **Self-Attention Mechanism**: Explore how tokens attend to each other with `SelfAttentionHead` - the fundamental building block of transformers with proper masking to prevent looking at future tokens
- **Multi-Head Attention**: Understand how multiple attention heads work in parallel (`MultiHeadSelfAttention`) to capture different relationships in data
- **Positional Embeddings**: Learn how position information is encoded and added to token embeddings
- **Layer Normalization**: Understand the role of normalization in transformer stability (Pre-LN architecture)
- **Feed-Forward Networks**: Explore position-wise feed-forward layers with ReLU activation and dimensionality expansion

### Training & Optimization
- **Gradient Descent with AdamW**: Understand modern optimization for transformer training
- **Loss Calculation**: Learn cross-entropy loss for language modeling tasks
- **Batch Processing**: Work with multiple sequences in parallel to speed up training
- **Learning Rate Scheduling**: Experiment with different learning rates and their impact
- **Checkpoint Management**: Save and resume training, enabling long training runs
- **Loss Evaluation**: Track training vs. validation loss to monitor overfitting

### Tokenization Strategies
- **Character-Level Tokenization**: Simple approach for understanding token mechanics
- **SentencePiece Tokenization**: Modern subword tokenization (BPE) for efficient vocabulary representation
- **Vocabulary Management**: Learn trade-offs between vocabulary size and model efficiency

### Model Configuration
Easily adjust these parameters to experiment with model capacity and behavior:
- **batch_size** - Process multiple sequences in parallel for efficient GPU utilization
- **block_size** - Maximum context length (sequence length) for predictions
- **n_layer** - Number of transformer blocks (depth)
- **n_embd** - Number of embedding dimensions (width)
- **n_head** - Number of attention heads for multi-head attention
- **dropout** - Regularization technique to prevent overfitting
- **learning_rate** - Controls the speed of parameter updates


## Key Features

ModelB supports:
- **Multiple Dataset Configurations**: Train on tiny or full Shakespeare datasets with automatic train/val splitting
- **Flexible Tokenization**: Choose between character-level and SentencePiece (BPE) tokenization
- **Configurable Model Variants**: Define experiments in Python config files for easy reproducibility
- **Multi-GPU Training**: Support for data parallelization across multiple GPUs
- **Cross-Platform Hardware**: Optimized for CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU

## Project Structure for Learning

The codebase is organized to highlight transformer concepts:
- `model.py` - Core transformer architecture with clean, well-documented modules
- `train.py` - Training loop showing the optimization process
- `sample.py` - Text generation demonstrating inference
- `tokenizers.py` - Tokenization implementations
- `data.py` - Dataset loading and management
- `config/` - Experiment configurations for reproducible results
- `scripts/` - Data preparation tools

Special thanks to [Andrej Karpathy](https://github.com/karpathy) who inspired this project with his detailed and exceptionally clear [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) based on [NanoGPT](https://github.com/karpathy/nanoGPT)

## Quick Start

### Environment Setup

Dependencies:
- Python 3.9+
- PyTorch (with GPU support)
- NumPy
- SentencePiece (only required for `tokenizer_type = "sentencepiece"`)

**Installation example:**
```bash
pip install torch numpy sentencepiece
```

Note: Training is significantly faster on GPUs. CPU training will be very slow but works for tiny models.

### Training and Sampling

Training could be started running a command as following:

```bash
python train.py --config config/v4.py
```

**What happens during training:**
1. Loads the dataset and tokenizes it
2. Creates a GPT language model with the specified architecture
3. Trains the model for `max_iters` iterations
4. Evaluates loss on train/val splits every `eval_interval` iterations
5. Saves checkpoints every `eval_interval * 3` iterations
6. Saves the final trained model

Generate text with a checkpoint:

```bash
python sample.py --config config/v4.py
```

**What happens during inference:**
1. Loads a trained model from the checkpoint
2. Starts with a seed token and generates new tokens autoregressively
3. Samples tokens based on the model's probability distribution
4. Outputs the generated text


### Understanding the Configurations

The project includes 5 example configurations demonstrating different model sizes and approaches:

| Config | Tokenizer | Size | Dataset | Use Case |
|--------|-----------|------|---------|----------|
| v1 | Char | Tiny | tinyshakespeare | Quick experiments, CPU-friendly |
| v2 | Char | Small | tinyshakespeare | Fast training to observe learning |
| v3 | Char | Medium | fullshakespeare | Balanced performance/speed |
| v4 | Char | Large | fullshakespeare | Full model on complete dataset |
| v5 | SentencePiece (BPE) | Large | train/val split | Modern tokenization approach |

Each config file in `config/` is self-contained and shows how to set all parameters for reproducibility.

### Hardware Notes

- **GPU recommended**: Training is practical on CUDA GPUs or Apple Silicon (MPS). GPU acceleration typically provides 10-50x speedup
- **NVIDIA GPUs (CUDA)**: Uses CUDA cores for massive parallelism - best for large-scale training
- **Apple Silicon (MPS)**: Metal Performance Shaders provide GPU acceleration on newer Macs
- **CPU runs**: Possible for tiny models only (v1 config); reduce `block_size`, `n_layer`, `n_head`, `n_embd`, and `max_iters` for CPU training

### Monitoring Training

**Measuring training time:**
You can wrap the training command with `time` and capture output in a log file:

```bash
time python ./train.py --config=config/v1.py | tee v1-tiny.log
```

This helps you understand:
- How long different model sizes take to train
- The relationship between model parameters and training time
- Convergence speed with different learning rates

**Expected performance:**
- v1 config on GPU: ~1-2 minutes for 5000 iterations
- v4 config on GPU: ~30-45 minutes for training
- Training loss should decrease over time, showing the model is learning

### Data Preparation

ModelB includes tools to work with Shakespeare texts at different scales. Two pre-processed datasets are provided:

| Dataset | Size | Content |
|---------|------|---------|
| Tiny Shakespeare | 1.1M chars | A subset for quick experiments |
| Full Shakespeare Collection | 5.3M chars | Complete works of Shakespeare |

You can also prepare your own datasets by placing text files in the `data/` folder and updating the config paths.

#### Understanding the Data Flow

1. **Raw Text** → Tokenization → **Token IDs** (integers)
2. **Token IDs** → Batching → **Sequences** (tensors)
3. **Sequences** → Model → **Predictions** (logits)
4. **Predictions** → Cross-Entropy Loss → **Gradient** (backward pass)

### Configuration

Configs live in `config/` and are plain Python files. Key fields:

- `name` - checkpoint name stem
- `dataset_path` - single dataset file (char tokenizer)
- `train_dataset_path`, `val_dataset_path` - explicit split files
- `tokenizer_type` - `char` or `sentencepiece`
- SentencePiece settings: `sp_model_path`, `sp_vocab_size`, `sp_model_type`, `sp_character_coverage`, `sp_train_if_missing`
- Model settings: `n_embd`, `n_head`, `n_layer`, `block_size`, `dropout`
- Training settings: `batch_size`, `max_iters`, `eval_interval`, `learning_rate`, `eval_iters`

Example (SentencePiece):

```python
# config/v5.py
name = "modelB-v5"
train_dataset_path = "data/train.txt"
val_dataset_path = "data/val.txt"

tokenizer_type = "sentencepiece"
sp_model_path = "models/modelB-v5_spm.model"
sp_vocab_size = 10000
sp_model_type = "bpe"
sp_character_coverage = 1.0
sp_train_if_missing = True
```

## Tokenizers and vocab versioning

- Tokenizer settings are saved into each checkpoint under the `tokenizer` key.
- On resume or sampling, the checkpoint tokenizer config is used to ensure vocab compatibility.
- For char-level models, the char set is stored in the checkpoint under `vocab_chars`.

## Checkpoints

Training saves checkpoints to `models/{name}.pth`. The file includes:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `loss`
- `tokenizer` (vocab configuration)

## Deep Dive: Transformer Architecture

ModelB implements a minimal but complete transformer decoder. Here's how the pieces fit together:

### The Building Blocks (in `model.py`)

**1. Token and Position Embeddings**
```
Input Tokens → [Token IDs] → Token Embeddings (n_embd dimensions)
                           → Add Position Embeddings
                           → Combined Representation
```
Tokens are converted to dense vectors (`token_embedding_table`). Position embeddings encode sequence positions so the model knows which word comes first, second, etc.

**2. Self-Attention Head**
```
Input [batch, seq_len, n_embd]
   ↓
Query, Key, Value Projections (Linear layers)
   ↓
Attention Scores = Query @ Key^T / √(head_size)
   ↓
Mask Future Tokens (causal mask prevents looking ahead)
   ↓
Softmax → Attention Weights
   ↓
Output = Attention Weights @ Values
   ↓
[batch, seq_len, head_size]
```
Attention allows each token to "look at" other tokens in the sequence, weighted by relevance. The causal mask ensures the model can only attend to previous tokens during training and generation.

**3. Multi-Head Attention**
Multiple attention heads run in parallel, each with their own Query/Key/Value projections:
```
Input → [Head 1] → [concat] → Linear Projection → Output
     → [Head 2] →
     → [Head 3] →
```
Different heads learn different patterns (e.g., some focus on syntax, others on semantics).

**4. Feed-Forward Network**
```
Input [batch, seq_len, n_embd]
   ↓
Linear (n_embd → 4*n_embd) + ReLU
   ↓
Linear (4*n_embd → n_embd)
   ↓
Dropout for regularization
   ↓
Output [batch, seq_len, n_embd]
```
Provides non-linear transformations and capacity for learning complex patterns.

**5. Transformer Block**
One complete block combines attention and feed-forward with residual connections and layer norm:
```
Input → LayerNorm → Self-Attention → Add (residual) → x
                                   ↓
                          LayerNorm → Feed-Forward → Add (residual) → Output
```
The residual connections (skip connections) allow gradients to flow directly through the network. Layer normalization stabilizes training.

**6. Full GPT Language Model**
```
Tokens (IDs) 
   ↓ Token Embedding
   ↓ + Position Embedding
   ↓ [Transformer Block] ×n_layer
   ↓ Layer Normalization (final)
   ↓ Linear to Vocab Size
   ↓ Logits over vocabulary
```
Stacking multiple transformer blocks creates depth. Each layer refines the representations from previous layers.

### The Training Loop (in `train.py`)

1. **Forward Pass**: Input tokens flow through the model producing logits (predictions)
2. **Loss Computation**: Compare predicted logits against actual next tokens using cross-entropy loss
3. **Backward Pass**: Compute gradients using automatic differentiation
4. **Parameter Update**: AdamW optimizer updates weights based on gradients
5. **Evaluation**: Periodically compute loss on held-out validation data

The key insight: the model learns to predict the next token by minimizing the difference between its predictions and the actual next token in the training data.

### The Generation Process (in `sample.py`)

```
Start with seed token(s)
Loop for max_new_tokens:
    Get logits from model for current sequence
    Extract logits from last token position
    Convert to probabilities (softmax)
    Sample a token from the distribution
    Append to sequence
Return full sequence
```

## Learning Paths

### Path 1: Understanding Attention (Beginner)
1. Start with `config/v1.py` - small model, quick training
2. Read `SelfAttentionHead` in `model.py` - understand key/query/value and masking
3. Modify: Change the causal mask to see what happens (remove the masked_fill)
4. Experiment: Reduce `n_head` to see single-head attention, then increase it

### Path 2: Model Architecture (Intermediate)
1. Train a v1 model and check the parameter count
2. Manually calculate parameters: `vocab_size * n_embd + n_embd^2 * n_head...`
3. Modify the architecture: Add residual connections to FFN, remove dropout temporarily
4. Compare training curves with/without layer norm

### Path 3: Tokenization & Data (Intermediate)
1. Train with character-level tokenization (v1-v4)
2. Prepare data and train with SentencePiece (v5)
3. Analyze: Check vocab coverage, look at token distributions
4. Experiment: Change `sp_vocab_size` and measure impact on training

### Path 4: Training Dynamics (Advanced)
1. Train v1, v2, v3 sequentially - observe how loss improves with larger models
2. Try different learning rates: 1e-3, 3e-4 (default), 1e-4
3. Analyze: Plot learning curves, identify optimal learning rate
4. Understand: Relationship between model size and convergence speed

### Path 5: Text Generation Quality (Advanced)
1. Train models with different `dropout` values (0.0, 0.2, 0.3)
2. Generate text at different stages of training (early, middle, final checkpoints)
3. Analyze: How does regularization affect generated text diversity?
4. Compare: Character-level vs. SentencePiece generation quality

## Key Concepts Explained

**Embedding Dimension (n_embd)**
- Each token is represented as a vector of `n_embd` numbers
- Larger dimensions capture more information but require more computation
- Typical range: 64-512 for small models, 768-3072 for larger ones (like BERT)

**Number of Heads (n_head)**
- Must divide evenly into `n_embd` (head_size = n_embd / n_head)
- Each head has `head_size` dimensions
- More heads allow attending to multiple patterns simultaneously
- Typical: 4-12 heads for small models, 8-16 for larger ones

**Context Length (block_size)**
- Maximum sequence length the model can process
- Determines how much history the model can "see"
- Longer context = better understanding but more computation
- Shakespeare: 128-256 usually sufficient

**Dropout**
- Randomly zeroes activations during training (disabled during inference)
- Prevents overfitting by creating implicit ensemble
- Typical: 0.1-0.3; increase if overfitting observed

**Learning Rate**
- Controls step size of parameter updates
- Too high: loss may diverge (increase dramatically)
- Too low: training very slow, may get stuck
- AdamW adapts per-parameter learning rates, making it robust to this choice

## Model Results & Benchmarks

Track your experiments here. The table shows example results from the provided configurations:

| Config | Dataset | Tokenizer | Params | Train Loss | Val Loss | Time | Device |
|--------|---------|-----------|--------|------------|----------|------|--------|
| v1 | tinyshakespeare | char | 1.8M | 1.34 | 1.55 | 9:08 | M3 |
| v2 | tinyshakespeare | char | 4.8M | 1.04 | 1.48 | 12:06 | 2x GPU |
| v3 | fullshakespeare | char | 6.4M | 1.19 | 1.27 | 16:06 | 2x GPU |
| v4 | fullshakespeare | SentencePiece | 8.9M | 3.16 | 3.71 | 14:06 | 2x GPU |
| v5 | fullshakespeare | SentencePiece | 15.6M | 3.08 | 3.97 | 21:02 | 2x GPU |

**How to interpret:**
- Larger models generally achieve lower loss (better predictions)
- Loss increases with larger vocabulary (SentencePiece), so compare within same tokenizer
- Validation loss higher than training loss is normal (model hasn't seen validation data)
- Gap between train and val loss indicates overfitting (larger dropout helps)

## Suggested Next Steps

**To Deepen Your Understanding:**
1. Implement attention without the causal mask and observe what happens (next-token prediction should fail)
2. Add learnable position embeddings instead of fixed embeddings
3. Experiment with different FFN expansion ratios (currently 4x)
4. Implement rotary positional embeddings (RoPE) for better extrapolation

**To Scale Up:**
1. Train on larger datasets (Common Crawl, C4, Books3)
2. Use larger models (increase n_layer, n_embd, n_head)
3. Implement gradient accumulation for larger effective batch sizes
4. Use distributed training across multiple machines

**To Understand Modern Transformers:**
1. Explore encoder-decoder architectures (like BERT or T5)
2. Study attention variants: sparse attention, linear attention
3. Learn about alignment and safety techniques
4. Understand how transformers scale to billions of parameters

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT) - Similar educational project
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Transformer Circuits Thread](https://transformer-circuits.pub/) - Deep mechanistic understanding

## Contributing

Have improvements or suggestions? Feel free to report issues or bugs, suggest clearer explanations or documentation or share you experiment configurations.
