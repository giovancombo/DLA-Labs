# Imports and dependencies
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from transformer import TransformerDecoder
from pipeline import *

project_name = "DLA_Lab2_LLM"

# Hyperparameters
text = 'taylor_swift'
train_size = 0.7

batch_size = 64             # Batch size = number of independent sequences of text, analyzed in parallel
block_size = 512            # Dimension of an input seuqence of characters, for next character prediction
n_embd = 100                # Embedding dimension for each token
n_heads = 4                 # Number of Self-Attention heads in a Multi-Head Attention block
n_layers = 4                # Number of Blocks of the Transformer
learning_rate = 5e-4
dropout = 0.4

eval_iters = 200
total_steps = 5000
log_interval = 100

# Generation configuration
generation = False
new_tokens = 500           # Number of tokens generated

# Saving configuration
save_model = False
folder = f"1_transformers/{text}"
model_name = "model_" + text + "_bs" + str(batch_size) + "_bl" + str(block_size) + "_ne" + str(n_embd) + "_nh" + str(n_heads) + "_nl" + str(n_layers) + "_lr" + str(learning_rate)

# Creating a configuration dictionary for logging in wandb
config = dict(
    text = text,
    batch_size = batch_size,
    block_size = block_size,
    n_embd = n_embd,
    n_heads = n_heads,
    n_layers = n_layers,
    learning_rate = learning_rate,
    dropout = dropout,
    eval_iters = eval_iters,
    total_steps = total_steps,
    log_interval = log_interval,
    train_size = train_size,)


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.login()
    print("Initializing Weights & Biases run...")

    # Downloading the Dante's Divina Commedia txt file from the internet
    #!wget https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt

    # Opening and reading the content of the input text file
    with open("data/" + text + '.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
    print("Length of dataset in characters:", len(text))

    # Creating a sorted set of all the unique characters present in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)             # Number of unique characters in the text = size of the vocabulary

    # Creating a dictionary for mapping characters to integers and viceversa
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}

    encode = lambda s: [stoi[i] for i in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # text = list of characters
    # data = list of integers of all the text --> it's our dataset
    data = torch.tensor(encode(text), dtype = torch.long)

    # Splitting our dataset in train and validation sets
    n = int(train_size*len(data))
    train_data, val_data = data[:n], data[n:]

    with wandb.init(project = project_name, config = config):
        config = wandb.config
        
        # Building model and optimizer
        model, criterion, optimizer = build_model(vocab_size, block_size, n_embd, n_heads, n_layers, learning_rate, device)

        # Training the model
        train(model, criterion, optimizer)
        wandb.unwatch(model)

        # Generating new text from the model trained (optional)
        if generation:
            text_generator(model, new_tokens)

        # Saving the model (optional)
        if save_model:
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(model, f"{folder}/{model_name}.pt")
            print('\nModel saved!')
