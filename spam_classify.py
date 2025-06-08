import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tiktoken
import numpy as np
import urllib.request
import ssl
import zipfile
import os
from pathlib import Path

# --- Model Definitions (Copied from your notebook) ---
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# --- Model Configuration (Based on your notebook) ---
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# --- Model Loading ---
@st.cache_resource
# def load_classifier(model_path="review_classifier.pth"):
#     # Initialize the model with the same configuration as training
#     model = GPTModel(BASE_CONFIG)
#     num_classes = 2 # Assuming binary classification as in your notebook
#     model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

#     # Load the state dictionary
#     model_state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Load to CPU
#     model.load_state_dict(model_state_dict)
#     model.eval() # Set the model to evaluation mode
#     return model
def load_classifier(model_path="review_classifier.pth"):
    # Hugging Face model download URL (must be the raw file link)
    model_url = "https://huggingface.co/Pratpokh/email_spam_classifier/resolve/main/review_classifier.pth"

    # Download the model if not already present
    if not os.path.exists(model_path):
        print("Downloading model from Hugging Face...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            raise RuntimeError(f"Failed to download model. Status code: {response.status_code}")

    # Initialize the model with the same configuration as training
    model = GPTModel(BASE_CONFIG)
    num_classes = 2  # Assuming binary classification as in your notebook
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    # Load the state dictionary
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load to CPU
    model.load_state_dict(model_state_dict)
    model.eval()  # Set the model to evaluation mode

    return model

# --- Tokenizer Initialization ---
@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

# --- Classification Function ---
def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length if max_length is not None else supported_context_length)]

    # Pad sequences to the longest sequence if max_length is specified
    if max_length is not None:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# --- Streamlit App ---
st.title("Email Spam Classifier")

st.write("Enter an email message below to classify it as 'spam' or 'not spam'.")

# Get user input
user_input = st.text_area("Enter email message:", "")

# Add a button to trigger classification
if st.button("Classify"):
    if user_input:
        # Load the model and tokenizer
        model = load_classifier()
        tokenizer = get_tokenizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define max_length based on your training dataset
        # If you know the exact max_length from your SpamDataset during training, use it here.
        # Otherwise, you can use the model's context length or None to use the input length.
        max_length_for_inference = BASE_CONFIG["context_length"] # Using model's context length
        # Or if you know the dataset max length:
        # max_length_for_inference = <your_dataset_max_length>

        # Classify the input
        classification_result = classify_review(
            user_input, model, tokenizer, device, max_length=max_length_for_inference
        )

        st.subheader("Classification Result:")
        st.write(f"The email is classified as: **{classification_result}**")
    else:
        st.warning("Please enter some text to classify.")
