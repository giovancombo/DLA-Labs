# Imports and dependencies
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from transformer import TransformerDecoder

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
total_steps = 10
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


def build_model():
    model = TransformerDecoder(vocab_size, block_size, n_embd, n_heads, n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model {model.__class__.__name__} instantiated!\n" + f"Number of parameters: {n_params}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(optimizer)

    return model, criterion, optimizer


# Let's define a function for getting a new batch of random sequences of characters in the text
def get_batch(split):
    data = train_data if split == 'train' else val_data

    idx = torch.randint(len(data) - block_size, (batch_size,))      # Drawing a set of batch_size indexes in the text
    x = torch.stack([data[i : i+block_size] for i in idx])          # Stacking block_size characters from each index
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])      # Creating the targets (= inputs shifted by 1)
    x, y = x.to(device), y.to(device)

    return x, y

# Let's create a function for saving and visualizing train and validation losses
@torch.no_grad()                            # Context Manager for disabling gradient calculation: better memory usage
def estimate_loss():
    outloss = {}
    outacc = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):         # Evaluating the losses eval_iters times on different batches
            X, Y = get_batch(split)
            logits, targets, loss = model(X, Y)

            # Computing accuracy
            total, correct = 0, 0
            _, pred = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            accuracy = 100 * correct / total

            losses[k] = loss.item()
            accuracies[k] = accuracy
        outloss[split] = losses.mean()
        outacc[split] = accuracies.mean()

    return outloss, outacc                  # out = dictionary of train and validation mean losses

# Function for log of validation data at the end of an epoch
def log_validation(epoch, mean_loss, val_loss, mean_accuracy, val_accuracy, step):
    wandb.log({"Train Loss": mean_loss, 
               "Validation Loss": val_loss,
               "Epoch": epoch + 1,
               "Train Accuracy": mean_accuracy, 
               "Validation Accuracy": val_accuracy,}, step = step)


# Training Loop      
def train(model, criterion, optimizer):
    # Telling W&B to watch gradients and the model parameters
    wandb.watch(model, criterion, log = "all", log_freq = log_interval)
    example_ct = 0

    print("Starting Training...")
    for step in range(total_steps):
        model.train()
        xb, yb = get_batch('train')                 # Sampling a batch of data
        _, _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        example_ct += batch_size

        if step % log_interval == 0 or step == total_steps - 1:
            losses, accuracies = estimate_loss()
            log_validation(step, losses['train'], losses['val'], accuracies['train'], accuracies['val'], step)
            print(f"Step {step+1}/{total_steps}:\tTrain Loss = {losses['train']:.4f}; Val Loss = {losses['val']:.4f}\tTrain Accuracy = {accuracies['train']:.2f}%; Val Accuracy = {accuracies['val']:.2f}%")

    print("\nTraining completed!")


# Function for generating new text!
def text_generator(model, new_tokens):
    # context = First character of the generated sequence = (1,1) Tensor of value 0 --> Token embedding for New Line
    context = torch.zeros((1,1), dtype = torch.long, device = device)

    print("\nTEXT GENERATION ACTIVATED! Generating new text...")
    generated_text = decode(model.generate(context, block_size, max_new_tokens = new_tokens)[0].tolist())
    
    print("Text generated!")
    print(generated_text)


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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


    wandb.login()
    print("Initializing Weights & Biases run...")

    with wandb.init(project = "DLA_Lab2_LLM", config = config):
        config = wandb.config
        
        # Building model and optimizer
        model, criterion, optimizer = build_model()

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
