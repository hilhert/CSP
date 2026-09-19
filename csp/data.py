import torch
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
from abc import ABC, abstractmethod

   
# ============================================================
# Dyck-1 (balanced parenthesis) data generator
#
#   Left parenthesis  = +1   (opens, depth += 1)
#   Right parenthesis = -1   (closes, depth -= 1)
#   Padding           =  0
#
# A prefix is valid (label 1) iff its running depth never goes
# negative at any step up to and including t.
#
# Embedding index remap (nn.Embedding requires [0, vocab)):
#    0  (padding) -> 0
#   +1  (left)    -> 1
#   -1  (right)   -> 2
# => vocab_size = 3 for every model on this task.
#
# X: [num_samples, max_len, 1]  long tensor  (embedding indices)
# y: [num_samples, max_len]     long tensor
#     1          if prefix up to t is valid (depth never < 0)
#     0          otherwise
#     IGNORE_IDX if t is beyond the real sequence length
#
# Dataset is balanced: half valid sequences, half invalid.
# Random sequences that happen to be valid are rejected.
# ============================================================

import random
import torch


PAD_IDX    = 0
IGNORE_IDX = -100

# Raw token -> embedding index
RAW_TO_IDX = {1: 1, 0: 0, -1: 2}   # left=+1 -> 1, right=-1 -> 2
VOCAB_SIZE = 3


# ------------------------------------------------------------
# Public: dataset generator
# ------------------------------------------------------------
def generate_parenthesis_data(
    num_samples=5000,
    min_len=2,
    max_len=16,
    max_depth=None,
):
    """
    Balanced Dyck-1 dataset with variable length and tail padding.

    Parameters
    ----------
    num_samples : int
        Total number of sequences (half valid, half invalid).
    min_len, max_len : int
        Even sequence-length bounds (inclusive). Padding fills to max_len.
    max_depth : int or None
        Maximum nesting depth for generated valid sequences.
        None = unbounded.

    Returns
    -------
    X : torch.LongTensor  [num_samples, max_len, 1]
    y : torch.LongTensor  [num_samples, max_len]
    """
    assert min_len % 2 == 0 and max_len % 2 == 0
    assert min_len <= max_len

    half = num_samples // 2
    X_raw, y_raw = [], []

    # ----- Half 1: valid (balanced) sequences -----
    for _ in range(half):
        L = random.randrange(min_len, max_len + 1, 2)
        seq = _gen_balanced(L, max_depth)
        X_raw.append(seq)
        y_raw.append(_prefix_scan(seq))

    # ----- Half 2: invalid sequences -----
    # Reject accidental-valid random sequences so the invalid
    # half is genuinely invalid.
    for _ in range(half):
        while True:
            L = random.randrange(min_len, max_len + 1, 2)
            seq = [random.choice([1, -1]) for _ in range(L)]
            if _has_negative_prefix(seq):
                break
        X_raw.append(seq)
        y_raw.append(_prefix_scan(seq))

    # ----- Shuffle -----
    combined = list(zip(X_raw, y_raw))
    random.shuffle(combined)
    X_raw, y_raw = zip(*combined)

    # ----- Remap tokens + pad -----
    X_padded, y_padded = [], []
    for seq, labels in zip(X_raw, y_raw):
        pad = max_len - len(seq)
        seq_idx = [RAW_TO_IDX[t] for t in seq] + [PAD_IDX] * pad
        X_padded.append(seq_idx)
        y_padded.append(labels + [IGNORE_IDX] * pad)

    X = torch.tensor(X_padded, dtype=torch.long).unsqueeze(-1)
    y = torch.tensor(y_padded, dtype=torch.long)
    return X, y


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _gen_balanced(seq_len, max_depth):
    """
    Recursively generate a balanced (Dyck-1) sequence of exactly
    seq_len. Left = +1, Right = -1.

    max_depth: maximum nesting depth (None = unbounded).
    """
    assert seq_len % 2 == 0

    if seq_len == 0:
        return []

    # At depth budget 1, only flat pairs are allowed: ()()..
    if max_depth is not None and max_depth <= 1:
        return [1, -1] * (seq_len // 2)

    # Choose how many pairs are nested directly inside the outer pair.
    inner_pairs = random.randint(0, seq_len // 2 - 1)
    inner_len   = inner_pairs * 2

    inner = _gen_balanced(
        inner_len,
        None if max_depth is None else max_depth - 1,
    )
    rest = _gen_balanced(seq_len - 2 - inner_len, max_depth)

    return [1] + inner + [-1] + rest


def _prefix_scan(seq):
    """
    Standard Dyck-1 prefix labels.

    y[t] = 1  iff running depth of seq[:t+1] is >= 0 at every step.
    y[t] = 0  otherwise.

    Here left = +1 (depth up) and right = -1 (depth down).
    """
    labels = [0] * len(seq)
    balance = 0
    valid = True

    for t, v in enumerate(seq):
        balance += v
        if balance < 0:
            valid = False
        labels[t] = int(valid)

    return labels


def _has_negative_prefix(seq):
    """True if running depth ever goes strictly negative."""
    bal = 0
    for v in seq:
        bal += v
        if bal < 0:
            return True
    return False


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------
if __name__ == "__main__":
    X, y = generate_parenthesis_data(num_samples=500, max_len=16)

    print("X shape:", X.shape, "dtype:", X.dtype)
    print("y shape:", y.shape, "dtype:", y.dtype)
    print("X min/max:", X.min().item(), X.max().item())
    print("y unique:", sorted(y.unique().tolist()))

    assert X.min().item() >= 0 and X.max().item() < VOCAB_SIZE
    assert set(y.unique().tolist()) <= {0, 1, IGNORE_IDX}

    mask = (y != IGNORE_IDX)
    pos_rate = (y[mask] == 1).float().mean().item()
    print("per-position positive rate:", round(pos_rate, 4))

    # Per-sequence valid rate (is the LAST real position labeled 1?)
    lengths = mask.sum(dim=1)
    last_pos = lengths - 1
    seq_valid = y[torch.arange(y.size(0)), last_pos] == 1
    print("per-sequence valid rate:", round(seq_valid.float().mean().item(), 4))

    # Decode and print a few examples
    IDX_TO_RAW = {v: k for k, v in RAW_TO_IDX.items()}
    print("\n--- examples ---")
    for i in range(3):
        toks = X[i].squeeze(-1).tolist()
        labs = y[i].tolist()
        L = lengths[i].item()
        raw = [IDX_TO_RAW[t] for t in toks[:L]]
        pretty = "".join("(" if r == 1 else ")" for r in raw)
        print(f"[{i}] seq={pretty:<16} labels={labs[:L]}")

    print("\nOK")
    
    
    
    
# ============================================================
# 1. Parity: prefix labels
#    y[:, t] = parity of X[:, 0..t]
# ============================================================
def generate_parity_data(num_samples=5000, seq_len=16):
    """
    X: [num_samples, seq_len, 1]  float tensor of 0/1
    y: [num_samples, seq_len]     long tensor, y[:, t] = parity up to t
    """
    X = torch.randint(0, 2, (num_samples, seq_len)).float()
    # cumulative parity over the sequence
    y = (X.cumsum(dim=1) % 2).long()
    return X.unsqueeze(-1), y


# ============================================================
# 2. Mod-3 counting: prefix labels
#    y[:, t] = cumulative sum mod 3 up to t
#    classes: 0, 1, 2
# ============================================================
def generate_mod3_data(num_samples=5000, seq_len=16):
    """
    X: [num_samples, seq_len, 1]  float tensor of 0/1
    y: [num_samples, seq_len]     long tensor, y[:, t] = cumsum mod 3 up to t
    """
    X = torch.randint(0, 2, (num_samples, seq_len)).float()
    y = (X.cumsum(dim=1) % 3).long()
    return X.unsqueeze(-1), y




def create_dataloaders(X, y, generator, batch_size=64,train_ratio=0.8):
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    indices = torch.randperm(num_samples)
    train_idx, test_idx = indices[:num_train], indices[num_train:]
    
    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),generator=generator ,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx]), generator=generator ,batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



