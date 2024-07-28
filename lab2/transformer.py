# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

import torch
import torch.nn as nn
import torch.nn.functional as F


# Transformer Decoder: many Decoder blocks in sequence, with a final Linear layer
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_heads, n_layers, dropout):
        super(TransformerDecoder, self).__init__()
        # Creating Lookup Tables for storing Token and Positional Encodings
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        # In the original paper (Vaswani, 2017), 6 layers of 8 heads each were used
        self.decoder_blocks = nn.Sequential(*[DecoderLayer(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        device = idx.device
        # idx and targets = (B,T) int-tensors: B = batch_size, T = block_size
        B,T = idx.shape
        tok_emb = self.tok_embedding(idx)               # (B,T,C) = (batch_size, block_size, n_embd)
        pos_emb = self.pos_embedding(torch.arange(T, device = device))   # (T,C)
        x = tok_emb + pos_emb                           # (B,T,C) + (T,C) = (B,T,C) for Broadcasting Semantics

        x = self.decoder_blocks(x)                      # (B,T,C)
        x = self.ln(x)                                  # (B,T,C)        
        logits = self.fc(x)                             # (B,T,vocab_size)

        if targets is None:
            loss = None
            return logits
        else:
            B,T,C = logits.shape                        # (B,T,vocab_size)
            # Reshaping logits and targets due to dimension issues with PyTorch F.cross_entropy function
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, targets, loss

    # Function for generating new text
    def generate(self, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Calling the Forward method with no targets provided: we are generating
            logits = self(idx_cond)
            logits = logits[:, -1, :]                                   # (B,vocab_size)
            probs = F.softmax(logits, dim = -1)                         # (B,vocab_size)           
            idx_next = torch.multinomial(probs, num_samples = 1)        # Sampling from the distribution: (B,1)
            idx = torch.cat((idx, idx_next), dim = 1)                   # Adding character to sequence: (B,T+1)  
        return idx
    
    
# Self-Attention Head: creating communication between tokens
class SelfAttention(nn.Module):
    def __init__(self, head_size, block_size, n_embd, dropout):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)       # Q,K,V = (n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape                                     # (B,T,C) = (batch_size, block_size, n_embd)
        q = self.query(x)                                   # (B,T,C) @ (C,head_size) = (B,T,head_size)
        k = self.key(x)                                     # (B,T,head_size)
        v = self.value(x)                                   # (B,T,head_size)
        # Dot-Product and Scaling
        wei = q @ k.transpose(-2,-1) * C**-0.5              # (B,T,head_size) @ (B,head_size,T) = (B,T,T)
        # Masking (only for Decoder)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        # Dot-Product with Values
        out = wei @ v                                       # (B,T,T) @ (B,T,head_size) = (B,T,head_size)

        return out
    

# Multi-Head Attention block = Multiple Self-Attention Heads in parallel and concatenated
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, block_size, n_embd, dropout):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([SelfAttention(head_size, block_size, n_embd, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embd, n_embd)             # For handling Residual Connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)   # Concatenating outputs of the SA Heads
        out = self.dropout(self.projection(out))
        return out
    

# Feed Forward block = simple MLP, allowing tokens to do computation after communication
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super(FeedForward, self).__init__()

        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # Vaswani's paper uses scaling factor 4
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.ffwd(x)
    

# Decoder Layer = Multi-Head Attention + Feed Forward, Residuals & LayerNorm
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super(DecoderLayer, self).__init__()

        head_size = n_embd // n_heads
        self.attention = MultiHeadAttention(n_heads, head_size, block_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
