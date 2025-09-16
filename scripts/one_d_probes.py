import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Deterministic seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Task/config parameters
vocab_size = 11  # {0,1} plus BOS=2, SEP=3; remaining ids unused
BOS = 2
SEP = 3
L = 9
train_N = 51
test_N = 13
batch_size = 4
outer_loops = 256
lr = 1e-3
clip_norm = 1.0
max_len = 1 + L + 1 + L  # [BOS] + X + [SEP] + Y


def build_unique_binary_sequences(n: int = 64, L: int = 9) -> torch.Tensor:
    """Generate n unique binary sequences of length L.
    Uniqueness is enforced by packing bits into an int key.
    Returns a LongTensor of shape (n, L).
    """
    seen = set()
    xs = []
    while len(xs) < n:
        arr = torch.randint(0, 2, (L,), dtype=torch.long)
        key = 0
        for b in arr.tolist():
            key = (key << 1) | int(b)
        if key not in seen:
            seen.add(key)
            xs.append(arr)
    return torch.stack(xs)


def make_dataset(task: str):
    """Return X_train, Y_train, X_test, Y_test for given task ('inversion'|'reversal')."""
    Xs = build_unique_binary_sequences(64, L)
    if task == "inversion":
        Ys = 1 - Xs
    elif task == "reversal":
        Ys = torch.flip(Xs, dims=[1])
    else:
        raise ValueError(f"unknown task: {task}")
    idx = torch.arange(64)
    return Xs[idx[:train_N]], Ys[idx[:train_N]], Xs[idx[train_N:train_N + test_N]], Ys[idx[train_N:train_N + test_N]]


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DecoderOnlyLM(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 4, d_ff: int = 128, num_layers: int = 2):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer("pos", pe, persistent=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @staticmethod
    def _mask(S: int, device: torch.device) -> torch.Tensor:
        # 0 on allowed, -inf on disallowed
        return torch.triu(torch.full((S, S), float("-inf"), device=device), 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, S = ids.shape
        x = self.tok(ids) + self.pos[:S]
        h = self.enc(x, mask=self._mask(S, ids.device))
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, prefix_ids: torch.Tensor, gen_len: int) -> torch.Tensor:
        self.eval()
        ids = prefix_ids.clone()
        for _ in range(gen_len):
            logits = self.forward(ids)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def format_sequence(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    B = X.size(0)
    return torch.cat([
        torch.full((B, 1), BOS, dtype=torch.long),
        X,
        torch.full((B, 1), SEP, dtype=torch.long),
        Y,
    ], dim=1)


def compute_loss_for_Y(logits: torch.Tensor, full_ids: torch.Tensor) -> torch.Tensor:
    # Next-token loss restricted to the Y segment
    B, S, V = logits.shape
    targets = full_ids[:, 1:]
    pred = logits[:, :-1, :]
    start_orig = 1 + L + 1
    end_orig = start_orig + L
    start = start_orig - 1
    end = end_orig - 1
    y_targets = targets[:, start:end]
    y_pred = pred[:, start:end, :]
    return F.cross_entropy(y_pred.reshape(-1, V), y_targets.reshape(-1))


@torch.no_grad()
def evaluate(model: DecoderOnlyLM, X: torch.Tensor, Y: torch.Tensor) -> float:
    B = X.size(0)
    prefix = torch.cat([
        torch.full((B, 1), BOS, dtype=torch.long),
        X,
        torch.full((B, 1), SEP, dtype=torch.long),
    ], dim=1)
    gen = model.generate(prefix, L)
    gen_Y = gen[:, -L:]
    acc_seq = (gen_Y == Y).all(dim=1).float().mean().item()
    return acc_seq


if __name__ == "__main__":
    for task in ("inversion", "reversal"):
        Xtr, Ytr, Xte, Yte = make_dataset(task)
        model = DecoderOnlyLM()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        n_batches = math.ceil(Xtr.size(0) / batch_size)
        for outer in range(outer_loops):
            perm = torch.randperm(Xtr.size(0))
            Xtr = Xtr[perm]
            Ytr = Ytr[perm]
            for bi in range(n_batches):
                s = bi * batch_size
                e = min((bi + 1) * batch_size, Xtr.size(0))
                xb = Xtr[s:e]
                yb = Ytr[s:e]
                full = format_sequence(xb, yb)
                logits = model(full)
                loss = compute_loss_for_Y(logits, full)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                opt.step()
            if (outer + 1) % 64 == 0:
                print(f"[{task}] outer={outer + 1} eval_acc={evaluate(model, Xte, Yte):.3f}")
        print(f"[{task}] train_acc={evaluate(model, Xtr, Ytr):.3f} test_acc={evaluate(model, Xte, Yte):.3f}")
