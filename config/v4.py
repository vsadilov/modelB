# SentencePiece tokenizer configuration with increased block size, attention heads, and generation tokens (v4).
name = "modelB-v4"

train_dataset_path = "data/train.txt"
val_dataset_path = "data/val.txt"

tokenizer_type = "sentencepiece"
sp_model_path = "models/modelB-v4_spm.model"
sp_vocab_size = 2000
sp_model_type = "bpe"
sp_character_coverage = 1.0
sp_train_if_missing = True

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 4000  # how many training iterations?
eval_interval = 300  # how often to evaluate the model?
learning_rate = 3e-4  # the base learning rate used for training
eval_iters = 200  # how many batches to use for evaluation
n_embd = 256  # the number of embedding dimensions
n_head = 8  # the number of attention heads
n_layer = 8  # the number of transformer blocks
dropout = 0.3  # the dropout rate

max_new_tokens = 500  # number of tokens to generate after training
