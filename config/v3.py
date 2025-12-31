# Slightly deeper character-level configuration and bigger dataset (v3).
name = "modelB-v3"

train_dataset_path = "data/train.txt"
val_dataset_path = "data/val.txt"

tokenizer_type = "char"

batch_size = 64
block_size = 256
max_iters = 10
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.2

max_new_tokens = 500
