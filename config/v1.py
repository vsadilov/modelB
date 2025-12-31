# Baseline character-level configuration (v1).
name = "modelB-v1-tiny"

dataset_path = "data/tinyshakespeare.txt"

tokenizer_type = "char"

batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 5000  # how many training iterations?
eval_interval = 300  # how often to evaluate the model?
learning_rate = 3e-4  # the base learning rate used for training
eval_iters = 200  # how many batches to use for evaluation
n_embd = 192  # the number of embedding dimensions
n_head = 6  # the number of attention heads
n_layer = 4  # the number of transformer blocks
dropout = 0.2  # the dropout rate

max_new_tokens = 500  # number of tokens to generate after training
