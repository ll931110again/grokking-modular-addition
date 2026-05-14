"""
Grokfast: amplify slow-varying gradient components (arxiv:2405.20233).

Reference implementation adapted from:
https://github.com/ironjr/grokfast (MIT)

Mechanism (per parameter, each optimizer step)
----------------------------------------------
Let g_t be the gradient tensor after loss.backward(). Grokfast builds a smoothed
signal s_t from recent g's, then sets the gradient the optimizer sees to::

    g_eff_t = g_t + lamb * s_t

EMA filter (gradfilter_ema)
    Maintains state h_t with::

        h_t = alpha * h_{t-1} + (1 - alpha) * g_t
        g_eff_t = g_t + lamb * h_t

    Large alpha (e.g. 0.98) => h_t changes slowly (low-pass over time).

MA filter (gradfilter_ma)
    s_t is the mean (or sum) of the last window_size gradients; then
    g_eff_t = g_t + lamb * s_t. Optional warmup waits until the deque is full.

Call either filter after backward() and before optimizer.step() on the same
module whose .grad fields were just populated.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Literal, Optional

import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, Deque[torch.Tensor]]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, Deque[torch.Tensor]]:
    """Moving-average Grokfast: p.grad <- g_t + lamb * mean(last window_size grads).

    After appending the current grad to each deque, if warmup is False or the
    deque is full (and not trigger), avg is mean or sum of stored grads; then
    p.grad is replaced by p.grad + avg * lamb. Returns the deque dict for the
    next step.
    """
    if grads is None:
        grads = {
            n: deque(maxlen=window_size)
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if n not in grads:
            grads[n] = deque(maxlen=window_size)
        grads[n].append(p.grad.data.detach())

        if (not warmup) or (len(grads[n]) == window_size and not trigger):
            if filter_type == "mean":
                avg = sum(grads[n]) / len(grads[n])  # type: ignore[arg-type]
            elif filter_type == "sum":
                avg = sum(grads[n])  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unrecognized filter_type {filter_type}")
            p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """EMA Grokfast: h <- alpha*h + (1-alpha)*g, then p.grad <- g + lamb*h.

    Here g is the current p.grad after backward. h is per-parameter EMA state
    (same shape as g). Large alpha => slow-moving h (low-pass). Returns the
    state dict for the next step.
    """
    if grads is None:
        grads = {n: p.grad.data.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if n not in grads:
            grads[n] = p.grad.data.detach().clone()
        grads[n] = grads[n] * alpha + p.grad.data.detach() * (1.0 - alpha)
        p.grad.data = p.grad.data + grads[n] * lamb

    return grads
