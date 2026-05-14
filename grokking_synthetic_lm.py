#!/usr/bin/env python3
"""
Tiny Transformer on synthetic algorithmic language: predict the reversed token sequence.

Encoder-only Transformer with full self-attention reads a length-L string over a
small vocabulary; each position predicts the reversed string at that position
(all logits in parallel). This is a standard algorithmic LM-style task where
delayed generalization (grokking) can appear under weight decay + long training.

Examples:
  uv run python grokking_synthetic_lm.py
  uv run python grokking_synthetic_lm.py --grad-filter ema --epochs 8000 --plot reverse_grokfast.png
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import grokfast


def _default_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SeqConfig:
    vocab_size: int = 4
    seq_len: int = 5
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1.0
    batch_size: int = 64
    num_epochs: int = 12_000
    train_fraction: float = 0.5
    seed: int = 0
    device: str = field(default_factory=_default_accelerator)
    grad_filter: Literal["none", "ema", "ma"] = "none"
    grokfast_ema_alpha: float = 0.98
    grokfast_ema_lamb: float = 2.0
    grokfast_ma_window: int = 100
    grokfast_ma_lamb: float = 5.0
    grokfast_ma_filter_type: Literal["mean", "sum"] = "mean"
    grokfast_ma_warmup: bool = True


CFG = SeqConfig()


def all_reverse_pairs(vocab_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All V^L strings; target is token-wise reverse (same length)."""
    xs: list[tuple[int, ...]] = list(itertools.product(range(vocab_size), repeat=seq_len))
    x = torch.tensor(xs, dtype=torch.long)
    y = x.flip(dims=(1,))
    return x, y


def train_val_split_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    n = x.shape[0]
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * train_fraction)
    tr, va = idx[:cut], idx[cut:]
    tr_t = torch.tensor(tr, dtype=torch.long)
    va_t = torch.tensor(va, dtype=torch.long)
    return (x[tr_t], y[tr_t]), (x[va_t], y[va_t])


class TinySeqTransformer(nn.Module):
    """Encoder-only; (B,L) token ids -> (B,L,V) logits."""

    def __init__(self, cfg: SeqConfig) -> None:
        super().__init__()
        v, L, d = cfg.vocab_size, cfg.seq_len, cfg.d_model
        assert d % cfg.nhead == 0, "d_model must be divisible by nhead"
        self.tok = nn.Embedding(v, d)
        self.pos = nn.Embedding(L, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(d, v)
        self.cfg = cfg
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(b, L)
        h = self.tok(x) + self.pos(pos)
        h = h * math.sqrt(self.cfg.d_model)
        h = self.encoder(h)
        return self.head(h)


def train_one_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    cfg: SeqConfig,
    grad_state: dict[str, Any] | None,
) -> tuple[float, dict[str, Any] | None]:
    model.train()
    n = x.shape[0]
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    n_steps = 0
    state = grad_state
    for start in range(0, n, batch_size):
        sl = perm[start : start + batch_size]
        xb, yb = x[sl], y[sl]
        logits = model(xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
        opt.zero_grad()
        loss.backward()
        if cfg.grad_filter == "ema":
            state = grokfast.gradfilter_ema(
                model, state, alpha=cfg.grokfast_ema_alpha, lamb=cfg.grokfast_ema_lamb
            )
        elif cfg.grad_filter == "ma":
            state = grokfast.gradfilter_ma(
                model,
                state,
                window_size=cfg.grokfast_ma_window,
                lamb=cfg.grokfast_ma_lamb,
                filter_type=cfg.grokfast_ma_filter_type,
                warmup=cfg.grokfast_ma_warmup,
            )
        opt.step()
        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(n_steps, 1), state


@torch.no_grad()
def eval_seq_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean CE loss, sequence-level accuracy: all L tokens correct)."""
    model.eval()
    n = x.shape[0]
    total_loss = 0.0
    correct_seq = 0
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="sum")
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct_seq += (pred == yb).all(dim=1).sum().item()
    mean_loss = total_loss / max(n * y.shape[1], 1)
    acc = correct_seq / max(n, 1)
    return mean_loss, acc


def run_training(cfg: SeqConfig, plot_path: str | None) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    x, y = all_reverse_pairs(cfg.vocab_size, cfg.seq_len)
    (x_tr, y_tr), (x_va, y_va) = train_val_split_xy(x, y, cfg.train_fraction, cfg.seed)
    n_total = x.shape[0]
    print(
        f"vocab={cfg.vocab_size} len={cfg.seq_len} | {n_total} sequences | "
        f"train={x_tr.shape[0]} val={x_va.shape[0]} | device={cfg.device} | "
        f"grad_filter={cfg.grad_filter}"
    )

    model = TinySeqTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_tr_d, y_tr_d = x_tr.to(device), y_tr.to(device)
    x_va_d, y_va_d = x_va.to(device), y_va.to(device)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None  # type: ignore[assignment]

    train_acc_hist: list[float] = []
    val_acc_hist: list[float] = []
    grad_state: dict[str, Any] | None = None
    ne = cfg.num_epochs
    log_step = max(1, min(200, ne // 20))

    for epoch in range(ne):
        _, grad_state = train_one_epoch(
            model, opt, x_tr_d, y_tr_d, cfg.batch_size, device, cfg, grad_state
        )
        _, tr_acc = eval_seq_accuracy(model, x_tr_d, y_tr_d, cfg.batch_size, device)
        _, va_acc = eval_seq_accuracy(model, x_va_d, y_va_d, cfg.batch_size, device)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)
        if epoch % log_step == 0 or epoch == ne - 1:
            print(f"epoch {epoch:6d}  train_seq_acc={tr_acc:.4f}  val_seq_acc={va_acc:.4f}")

    if plt is not None and plot_path is not None:
        fl = {"none": "vanilla", "ema": "Grokfast EMA", "ma": "Grokfast MA"}[cfg.grad_filter]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(train_acc_hist, label="train seq acc")
        ax.plot(val_acc_hist, label="val seq acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("sequence accuracy")
        ax.legend()
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Reverse string V={cfg.vocab_size}, L={cfg.seq_len} — {fl}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tiny Transformer: synthetic reverse-string task")
    ap.add_argument("--grad-filter", choices=["none", "ema", "ma"], default="none")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--plot", type=str, default="reverse_string_curve.png")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SeqConfig(
        grad_filter=args.grad_filter,  # type: ignore[arg-type]
        num_epochs=args.epochs if args.epochs is not None else CFG.num_epochs,
        vocab_size=args.vocab_size if args.vocab_size is not None else CFG.vocab_size,
        seq_len=args.seq_len if args.seq_len is not None else CFG.seq_len,
    )
    if not (0.0 < cfg.train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0,1)")
    nseq = cfg.vocab_size ** cfg.seq_len
    if nseq > 50_000:
        raise ValueError(
            f"vocab_size^{cfg.seq_len} = {nseq} is large; lower --vocab-size or --seq-len "
            "so full enumeration stays small (this script enumerates all strings)."
        )
    run_training(cfg, plot_path=args.plot)


if __name__ == "__main__":
    main()
