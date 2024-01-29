import torch
import torch.nn as nn
import torch.optim
import wandb
from transformer import TransformerDecoder


def build_model(vocab_size, block_size, n_embd, n_heads, n_layers, learning_rate, device):
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