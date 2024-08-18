# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

# Code for EXERCISE 1

import os
import torch
import torch.optim as optim
import yaml
import wandb
import time
import winsound
from transformer import TransformerDecoder


# Function for getting a new batch of random sequences of characters in the text
def get_batch(split, block_size, batch_size, device):
    data = train_data if split == 'train' else val_data
    # Drawing a set of batch_size indexes in the text
    idx = torch.randint(len(data) - block_size, (batch_size,))
    # Stacking block_size characters from each index
    x = torch.stack([data[i : i + block_size] for i in idx])
    # Creating targets (= inputs shifted by 1)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

    return x.to(device), y.to(device)

# Function for saving and visualizing train and validation losses
@torch.no_grad()
def estimate_loss(model, block_size, batch_size, device, eval_iters = 10):
    outloss, outacc = {}, {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):         # Evaluating the losses eval_iters times on different batches
            X, Y = get_batch(split, block_size, batch_size, device)
            logits, targets, loss = model(X, Y)

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
def train(model, optimizer, log_interval, total_steps, block_size, batch_size, eval_iters, device, folder):
    example_ct = 0
    print("Starting Training...")
    for step in range(total_steps):
        model.train()
        xb, yb = get_batch('train', block_size, batch_size, device)
        _, _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        example_ct += batch_size
        if step % log_interval == 0 or step == total_steps - 1:
            losses, accuracies = estimate_loss(model, block_size, batch_size, device, eval_iters = eval_iters)
            log_validation(step, losses['train'], losses['val'], accuracies['train'], accuracies['val'], step)
            print(f"Step {step+1}/{total_steps}:\tTrain Loss = {losses['train']:.4f}; Val Loss = {losses['val']:.4f}\tTrain Accuracy = {accuracies['train']:.2f}%; Val Accuracy = {accuracies['val']:.2f}%")

        if step % 100 == 0:
            print("Generated text so far...")
            gen_text = text_generator(model, 200, block_size, device)
            print(gen_text)
            with open(f"{folder}/generation_training.txt", 'a', encoding="utf-8") as f:
                f.write(f"Text Generation at step {step}:\nTrain Loss = {losses['train']:.4f}; Val Loss = {losses['val']:.4f}\n{gen_text}\n\n - - - - - - - - - - - - - - -\n")


# Function for generating new text!
def text_generator(model, new_tokens, block_size, device):
    # context = First character of the generated sequence = (1,1) Tensor of value 0: Token embedding for New Line
    context = torch.zeros((1,1), dtype = torch.long, device = device)
    generated_text = decode(model.generate(context, block_size, max_new_tokens = new_tokens)[0].tolist())

    return generated_text


if __name__ == '__main__':
    # Loading configuration from YAML file
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    model_name = f"bs{params['batch_size']}_bl{params['block_size']}_ne{params['n_embd']}_nh{params['n_heads']}_nl{params['n_layers']}_lr{params['learning_rate']}_dr{params['dropout']}_" + str(time.time())[-7:]
    PATH = f"1_models/{params['text']}/{model_name}"
    os.makedirs(PATH, exist_ok = True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Downloading Dante's Divine Comedy .txt file
    # !wget https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt

    # Opening and reading the content of the input text file
    with open("data/" + params['text'] + '.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
    print("Length of dataset in characters:", len(text))

    # Creating a sorted set of all unique characters present in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Creating a dictionary for mapping characters to integers and viceversa
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[i] for i in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # text = list of characters; data = list of corresponding integers: our dataset
    data = torch.tensor(encode(text), dtype = torch.long, device = device)
    # Splitting dataset in train and validation sets
    isplit = int(params['train_size'] * len(data))
    train_data, val_data = data[:isplit].to(device), data[isplit:].to(device)

    # Model, Loss and Optimizer instantiation
    model = TransformerDecoder(vocab_size, params['block_size'], params['n_embd'], params['n_heads'], params['n_layers'], params['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = params['learning_rate'])

    print(f"Model {model.__class__.__name__} instantiated!")
    print(f"Optimizer: {optimizer.__class__.__name__}; Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Device: {device}")

    # Training the model
    wandb.login()
    with wandb.init(project = "DLA_Lab2_LLM", config = params, name = f"{params['text'][:6]}_bs{params['batch_size']}_bl{params['block_size']}_lr{params['learning_rate']}_dr{params['dropout']}"):
        config = wandb.config

        # Training the model
        wandb.watch(model, log = "all", log_freq = params['log_freq'])
        train(model, optimizer, params['log_freq'], params['total_steps'], params['block_size'], params['batch_size'], params['eval_iters'], device, PATH)

        duration = 1000
        freq = 440
        winsound.Beep(freq, duration)

        # Generating new text from the trained model (optional)
        if params['generate_text']:
            print("\nTEXT GENERATION ACTIVATED! Generating new text...")
            generated_text = text_generator(model, params['new_tokens'])
            print(generated_text)

        # Saving the model (optional)
        save_model = params['save_model']
        if save_model:
            torch.save(model.state_dict(), PATH + f"/model.pth")
            print('\nModel saved!')
