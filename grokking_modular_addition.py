#!/usr/bin/env python3
"""
Modular addition (a + b) mod p: vanilla training (delayed grokking) vs Grokfast.

Grokfast (arxiv:2405.20233) amplifies slow-varying gradient components; call
`gradfilter_ema` / `gradfilter_ma` after `backward()` and before `step()`.

Examples:
  uv sync
  uv run python grokking_modular_addition.py
  uv run python grokking_modular_addition.py --grad-filter ema
  uv run python grokking_modular_addition.py --compare
  uv run python grokking_modular_addition.py --bench-prime 59
  uv run python grokking_modular_addition.py --bench-prime 79 --output benchmark_p79.png
  uv run python grokking_modular_addition.py --bench-prime 97 --output benchmark_p97_dashboard.png
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import grokfast


# =============================================================================
# FILL IN — hyperparameters (weight decay & long training matter for grokking)
# =============================================================================

def _default_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    prime_p: int | None = None  # e.g. 97, 113
    train_fraction: float | None = None  # use < 1.0 (e.g. 0.5) so val measures generalization; must be in (0,1)
    hidden_dim: int | None = None
    num_hidden_layers: int | None = None  # not counting final readout
    lr: float | None = None
    weight_decay: float | None = None  # try AdamW values like 1e-2 … 1.0 (tune!)
    batch_size: int | None = None
    num_epochs: int | None = None  # grokking often needs 1e4–1e6+ epochs on small nets
    seed: int = 0
    device: str = field(default_factory=_default_accelerator)
    # Grokfast: none = vanilla (standard delayed grokking); ema / ma = gradient filters
    grad_filter: Literal["none", "ema", "ma"] = "none"
    grokfast_ema_alpha: float = 0.98
    grokfast_ema_lamb: float = 2.0
    grokfast_ma_window: int = 100
    grokfast_ma_lamb: float = 5.0
    grokfast_ma_filter_type: Literal["mean", "sum"] = "mean"
    grokfast_ma_warmup: bool = True


CFG = Config(
    prime_p=97,
    train_fraction=0.5,
    hidden_dim=256,
    num_hidden_layers=2,
    lr=1e-3,
    weight_decay=1.0,
    batch_size=512,
    num_epochs=12_000,
)


def _validate_cfg(c: Config) -> None:
    required = [
        "prime_p",
        "train_fraction",
        "hidden_dim",
        "num_hidden_layers",
        "lr",
        "weight_decay",
        "batch_size",
        "num_epochs",
    ]
    for name in required:
        if getattr(c, name) is None:
            raise ValueError(f"Set Config.{name} in CFG (top of file).")
    if not (0.0 < c.train_fraction < 1.0):  # type: ignore[operator]
        raise ValueError("train_fraction must be strictly between 0 and 1 so the val split is non-empty.")


# =============================================================================
# FILL IN — represent each pair (a, b) as a model input vector
# =============================================================================

def encode_pairs(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    """
    Args:
        a, b: int64 tensors of shape (batch,), values in [0, p-1]
        p: prime modulus
    Returns:
        x: float tensor (batch, in_features) — your choice of encoding
    """
    one_a = F.one_hot(a, num_classes=p).float()
    one_b = F.one_hot(b, num_classes=p).float()
    return torch.cat([one_a, one_b], dim=-1)


# =============================================================================
# FILL IN — small MLP mapping encoded (a,b) -> logits over classes 0..p-1
# =============================================================================

class ModularMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, cfg: Config) -> None:
        super().__init__()
        h = cfg.hidden_dim  # type: ignore[assignment]
        n_h = cfg.num_hidden_layers  # type: ignore[assignment]
        layers: list[nn.Module] = []
        d = in_features
        for _ in range(n_h):
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Dataset utilities (complete — you may adjust split logic only if you want)
# =============================================================================

def all_pairs_mod_sum(p: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every (a,b) in Z_p x Z_p and target (a+b) % p as class indices."""
    pairs: list[tuple[int, int, int]] = []
    for aa in range(p):
        for bb in range(p):
            pairs.append((aa, bb, (aa + bb) % p))
    a = torch.tensor([t[0] for t in pairs], dtype=torch.long)
    b = torch.tensor([t[1] for t in pairs], dtype=torch.long)
    y = torch.tensor([t[2] for t in pairs], dtype=torch.long)
    return a, b, y


def train_val_split(
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    n = a.shape[0]
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * train_fraction)
    tr, va = idx[:cut], idx[cut:]
    tr_t = torch.tensor(tr, dtype=torch.long)
    va_t = torch.tensor(va, dtype=torch.long)
    train = (a[tr_t], b[tr_t], y[tr_t])
    val = (a[va_t], b[va_t], y[va_t])
    return train, val


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


# =============================================================================
# FILL IN — optimizer choice / schedulers (optional)
# =============================================================================

def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,  # type: ignore[arg-type]
        weight_decay=cfg.weight_decay,  # type: ignore[arg-type]
    )


# =============================================================================
# Training loop (structure complete — tweak logging if you like)
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    p: int,
    batch_size: int,
    device: torch.device,
    cfg: Config,
    grad_state: dict[str, Any] | None,
) -> tuple[float, dict[str, Any] | None]:
    model.train()
    perm = torch.randperm(a.shape[0], device=device)
    total_loss = 0.0
    n_steps = 0
    state: dict[str, Any] | None = grad_state
    for start in range(0, a.shape[0], batch_size):
        sl = perm[start : start + batch_size]
        aa, bb, yy = a[sl], b[sl], y[sl]
        x = encode_pairs(aa, bb, p).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, yy)
        opt.zero_grad()
        loss.backward()
        if cfg.grad_filter == "ema":
            state = grokfast.gradfilter_ema(
                model,
                state,
                alpha=cfg.grokfast_ema_alpha,
                lamb=cfg.grokfast_ema_lamb,
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
def eval_split(
    model: nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    p: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for start in range(0, a.shape[0], batch_size):
        aa, bb, yy = a[start : start + batch_size], b[start : start + batch_size], y[start : start + batch_size]
        x = encode_pairs(aa, bb, p).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, yy)
        total_loss += loss.item() * aa.shape[0]
        pred = logits.argmax(dim=-1)
        correct += (pred == yy).sum().item()
        total += aa.shape[0]
    return total_loss / max(total, 1), correct / max(total, 1e-8)


def infer_in_features(p: int) -> int:
    """Call your encode on a dummy batch to discover in_features for the MLP."""
    aa = torch.zeros(1, dtype=torch.long)
    bb = torch.zeros(1, dtype=torch.long)
    x = encode_pairs(aa, bb, p)
    return x.shape[-1]


def default_plot_path(cfg: Config) -> str:
    if cfg.grad_filter == "none":
        return "grokking_curve.png"
    return f"grokking_curve_grokfast_{cfg.grad_filter}.png"


@dataclass
class TrainRunResult:
    train_acc: list[float]
    val_acc: list[float]
    epochs_run: int
    wall_seconds: float


def train_modular(
    cfg: Config,
    *,
    plot_path: str | None,
    run_name: str = "",
    verbose: bool = True,
    early_stop_val: float | None = None,
) -> TrainRunResult:
    import time

    _validate_cfg(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    p = cfg.prime_p  # type: ignore[assignment]

    a, b, y = all_pairs_mod_sum(p)
    (a_tr, b_tr, y_tr), (a_va, b_va, y_va) = train_val_split(a, b, y, cfg.train_fraction, cfg.seed)  # type: ignore[arg-type]

    in_features = infer_in_features(p)
    model = ModularMLP(in_features=in_features, num_classes=p, cfg=cfg).to(device)  # type: ignore[arg-type]
    opt = build_optimizer(model, cfg)

    a_tr_d, b_tr_d, y_tr_d = a_tr.to(device), b_tr.to(device), y_tr.to(device)
    a_va_d, b_va_d, y_va_d = a_va.to(device), b_va.to(device), y_va.to(device)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None  # type: ignore[assignment]

    train_acc_hist: list[float] = []
    val_acc_hist: list[float] = []
    grad_state: dict[str, Any] | None = None
    prefix = f"[{run_name}] " if run_name else ""
    ne: int = cfg.num_epochs  # type: ignore[assignment]
    log_step = max(1, min(300, ne // 20))

    t_wall0 = time.perf_counter()
    for epoch in range(ne):
        _, grad_state = train_one_epoch(
            model,
            opt,
            a_tr_d,
            b_tr_d,
            y_tr_d,
            p,
            cfg.batch_size,  # type: ignore[arg-type]
            device,
            cfg,
            grad_state,
        )
        _, tr_acc = eval_split(model, a_tr_d, b_tr_d, y_tr_d, p, cfg.batch_size, device)  # type: ignore[arg-type]
        _, va_acc = eval_split(model, a_va_d, b_va_d, y_va_d, p, cfg.batch_size, device)  # type: ignore[arg-type]
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        if verbose and (epoch % log_step == 0 or epoch == ne - 1):
            print(f"{prefix}epoch {epoch:6d}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        if early_stop_val is not None and va_acc >= early_stop_val:
            if verbose:
                print(
                    f"{prefix}early stop: val_acc >= {early_stop_val} at epoch {epoch} "
                    f"(ran {epoch + 1} epochs)"
                )
            break

    wall_seconds = time.perf_counter() - t_wall0

    if plt is not None and plot_path is not None:
        filter_label = {"none": "vanilla (no Grokfast)", "ema": "Grokfast EMA", "ma": "Grokfast MA"}[cfg.grad_filter]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(train_acc_hist, label="train acc")
        ax.plot(val_acc_hist, label="val acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.legend()
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"p={p}, {filter_label}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"{prefix}Saved plot to {plot_path}")

    return TrainRunResult(
        train_acc=train_acc_hist,
        val_acc=val_acc_hist,
        epochs_run=len(train_acc_hist),
        wall_seconds=wall_seconds,
    )


def run_compare(cfg: Config, out_path: str) -> None:
    import matplotlib.pyplot as plt

    cfg_b = replace(cfg, grad_filter="none")
    cfg_f = replace(cfg, grad_filter="ema")
    print("=== Baseline (vanilla — delayed grokking) ===\n")
    rb = train_modular(cfg_b, plot_path=None, run_name="baseline", verbose=True)
    print("\n=== Grokfast (EMA — amplified slow gradients) ===\n")
    rf = train_modular(cfg_f, plot_path=None, run_name="grokfast-ema", verbose=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rb.val_acc, label="val — baseline (no filter)")
    ax.plot(rf.val_acc, label="val — Grokfast EMA")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Modular addition p={cfg.prime_p}: val generalization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot to {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Modular addition: grokking vs Grokfast (arxiv:2405.20233)")
    ap.add_argument(
        "--grad-filter",
        choices=["none", "ema", "ma"],
        default=None,
        help="none = standard training (delayed grokking); ema/ma = Grokfast gradient filter",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Train baseline then Grokfast EMA (same seed/hparams) and save one comparison figure",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Figure path: single run / --compare / --bench-prime (benchmark dashboard)",
    )
    ap.add_argument(
        "--bench-prime",
        type=int,
        metavar="P",
        default=None,
        help="Run baseline vs Grokfast EMA benchmark for modulus P (val>=0.99 early stop); e.g. 59 or 79",
    )
    return ap.parse_args()


def _cumulative_wall_linear(wall_seconds: float, n_epochs: int) -> list[float]:
    """End-of-epoch cumulative time, assuming uniform cost per epoch (good for visuals)."""
    if n_epochs <= 0:
        return []
    per = wall_seconds / n_epochs
    return [per * (i + 1) for i in range(n_epochs)]


def _save_benchmark_plots(
    *,
    prime_p: int,
    thresh: float,
    rb: TrainRunResult,
    rf: TrainRunResult,
    out_path: str,
    device: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping benchmark plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]
    c_b, c_f = "#4477AA", "#CC6677"

    ax00.plot(range(rb.epochs_run), rb.val_acc, color=c_b, label=f"Grokking (n={rb.epochs_run})", linewidth=1.2)
    ax00.plot(range(rf.epochs_run), rf.val_acc, color=c_f, label=f"Grokfast (n={rf.epochs_run})", linewidth=1.2)
    ax00.axhline(thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax00.set_xlabel("training epoch")
    ax00.set_ylabel("validation accuracy")
    ax00.set_title("Val accuracy vs epochs")
    ax00.legend(loc="lower right")
    ax00.set_ylim(-0.02, 1.02)

    t_b = _cumulative_wall_linear(rb.wall_seconds, rb.epochs_run)
    t_f = _cumulative_wall_linear(rf.wall_seconds, rf.epochs_run)
    ax10.plot(t_b, rb.val_acc, color=c_b, label="Grokking", linewidth=1.2)
    ax10.plot(t_f, rf.val_acc, color=c_f, label="Grokfast", linewidth=1.2)
    ax10.axhline(thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax10.set_xlabel("cumulative wall time (s, uniform / epoch)")
    ax10.set_ylabel("validation accuracy")
    ax10.set_title("Val accuracy vs wall time (approx.)")
    ax10.legend(loc="lower right")
    ax10.set_ylim(-0.02, 1.02)

    labels = ["Grokking\n(baseline)", "Grokfast\n(EMA)"]
    ax01.bar(labels, [rb.epochs_run, rf.epochs_run], color=[c_b, c_f], width=0.55)
    ax01.set_ylabel("epochs until stop")
    ax01.set_title("Epochs to val ≥ threshold")

    ax11.bar(labels, [rb.wall_seconds, rf.wall_seconds], color=[c_b, c_f], width=0.55)
    ax11.set_ylabel("wall time (s)")
    ax11.set_title("Total wall time to stop")

    fig.suptitle(f"p = {prime_p}, early stop: val ≥ {thresh}  |  device: {device}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved benchmark figure to {out_path}")


def run_benchmark_prime(prime_p: int, summary_plot_path: str | None = None) -> None:
    """Same hparams for both runs; stop when val_acc first reaches 0.99 (or max epochs)."""
    thresh = 0.99
    max_e = 50_000
    # Larger moduli need more width to reach val≥0.99 within max_epochs at this depth.
    hidden = 256 if prime_p >= 97 else 128
    shared = dict(
        prime_p=prime_p,
        train_fraction=0.5,
        hidden_dim=hidden,
        num_hidden_layers=2,
        lr=1e-3,
        weight_decay=1.0,
        batch_size=512,
        num_epochs=max_e,
        seed=0,
    )
    cfg_b = replace(CFG, **shared, grad_filter="none")  # type: ignore[arg-type]
    cfg_f = replace(CFG, **shared, grad_filter="ema")  # type: ignore[arg-type]
    dev = cfg_b.device
    print(f"p={prime_p}, hidden_dim={hidden}, val early-stop >= {thresh}, max_epochs={max_e}, device={dev}\n")

    print("=== Grokking (baseline, no Grokfast) ===")
    rb = train_modular(cfg_b, plot_path=None, run_name="grok", verbose=True, early_stop_val=thresh)
    print("\n=== Grokfast (EMA) ===")
    rf = train_modular(cfg_f, plot_path=None, run_name="grokfast", verbose=True, early_stop_val=thresh)

    print("\n--- Summary ---")
    print(f"{'':20} {'epochs':>10} {'wall (s)':>12} {'wall (min)':>12}")
    print(f"{'Grokking (baseline)':20} {rb.epochs_run:10d} {rb.wall_seconds:12.1f} {rb.wall_seconds / 60:12.2f}")
    print(f"{'Grokfast (EMA)':20} {rf.epochs_run:10d} {rf.wall_seconds:12.1f} {rf.wall_seconds / 60:12.2f}")
    if rb.epochs_run > 0 and rf.epochs_run > 0:
        print(f"\nEpoch ratio (baseline / grokfast): {rb.epochs_run / rf.epochs_run:.2f}x")
        print(f"Wall ratio (baseline / grokfast):  {rb.wall_seconds / rf.wall_seconds:.2f}x")

    out = summary_plot_path or f"benchmark_p{prime_p}.png"
    _save_benchmark_plots(prime_p=prime_p, thresh=thresh, rb=rb, rf=rf, out_path=out, device=dev)


def main() -> None:
    args = parse_args()
    if args.bench_prime is not None:
        run_benchmark_prime(args.bench_prime, summary_plot_path=args.output)
        return
    if args.compare:
        run_compare(CFG, args.output or "grokking_compare.png")
        return

    cfg = CFG
    if args.grad_filter is not None:
        cfg = replace(CFG, grad_filter=args.grad_filter)  # type: ignore[arg-type]
    out = args.output or default_plot_path(cfg)
    train_modular(cfg, plot_path=out, verbose=True)


if __name__ == "__main__":
    main()
