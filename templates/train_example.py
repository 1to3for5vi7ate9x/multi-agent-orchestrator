#!/usr/bin/env python3
"""Example training script for the ml-agent-orchestrator loop.

Trains a small regressor on a synthetic noisy-sine dataset and dumps
structured progress to ``metrics.json`` after EVERY epoch, so the
orchestrator can inspect partial progress even if the run is killed.

Final metrics.json format (contract with the orchestrator):
    {
        "epoch": N,
        "train_loss": X,
        "val_loss": Y,
        "status": "COMPLETED",           # RUNNING | COMPLETED | CRASHED
        "history": [{"epoch": ..., "train_loss": ..., "val_loss": ...}, ...],
        "gpu_mem_mb": ...                # present only when CUDA is used
    }

Backends (auto-selected): PyTorch when available (uses
model_example.build_model), else NumPy, else a pure-stdlib MLP — so the
framework can be demoed on any machine with zero dependencies.
"""

from __future__ import annotations

import json
import math
import random
import traceback
from pathlib import Path

# ---- Hyperparameters (safe for the Editor agent to tune) --------------------
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
N_SAMPLES = 1000
NOISE_STD = 0.1
SEED = 42

METRICS_PATH = Path("metrics.json")


def write_metrics(epoch, train_loss, val_loss, status, history, extra=None):
    payload = {
        "epoch": epoch,
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "status": status,
        "history": history,
    }
    if extra:
        payload.update(extra)
    tmp = METRICS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(METRICS_PATH)


def make_dataset(np, n_samples, noise_std, seed):
    """Synthetic regression: y = sin(3x0) + 0.5*x1^2 - x2 + noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 3)).astype("float32")
    y = (
        np.sin(3.0 * X[:, 0])
        + 0.5 * X[:, 1] ** 2
        - X[:, 2]
        + rng.normal(0.0, noise_std, size=n_samples)
    ).astype("float32").reshape(-1, 1)
    split = int(0.8 * n_samples)
    return (X[:split], y[:split]), (X[split:], y[split:])


# -----------------------------------------------------------------------------
# PyTorch backend
# -----------------------------------------------------------------------------

def train_torch():
    import numpy as np
    import torch
    import torch.nn as nn

    from model_example import build_model

    torch.manual_seed(SEED)
    (X_tr, y_tr), (X_va, y_va) = make_dataset(np, N_SAMPLES, NOISE_STD, SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] backend=pytorch device={device}")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    model = build_model(input_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    history = []
    n = X_tr_t.shape[0]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(X_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        train_loss = epoch_loss / max(1, batches)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"loss diverged to NaN/Inf at epoch {epoch} "
                f"(train_loss={train_loss}, val_loss={val_loss})"
            )

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        extra = {}
        if device.type == "cuda":
            extra["gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1
            )
        write_metrics(epoch, train_loss, val_loss, "RUNNING", history, extra)
        print(f"[epoch {epoch:03d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    extra = {}
    if device.type == "cuda":
        extra["gpu_mem_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1
        )
    write_metrics(history[-1]["epoch"], history[-1]["train_loss"],
                  history[-1]["val_loss"], "COMPLETED", history, extra)
    print(f"[train] done. final val_loss={history[-1]['val_loss']:.4f}")


# -----------------------------------------------------------------------------
# NumPy fallback backend (2-layer MLP, manual backprop)
# -----------------------------------------------------------------------------

def train_numpy():
    import numpy as np

    print("[train] backend=numpy (PyTorch not installed)")
    rng = np.random.default_rng(SEED)
    (X_tr, y_tr), (X_va, y_va) = make_dataset(np, N_SAMPLES, NOISE_STD, SEED)

    d_in, hidden = X_tr.shape[1], 32
    W1 = rng.normal(0, math.sqrt(2.0 / d_in), (d_in, hidden)).astype("float32")
    b1 = np.zeros(hidden, dtype="float32")
    W2 = rng.normal(0, math.sqrt(2.0 / hidden), (hidden, 1)).astype("float32")
    b2 = np.zeros(1, dtype="float32")

    def forward(X):
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        return z1, a1, a1 @ W2 + b2

    history = []
    n = X_tr.shape[0]
    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb, yb = X_tr[idx], y_tr[idx]
            z1, a1, out = forward(Xb)
            err = out - yb
            loss = float(np.mean(err ** 2))
            epoch_loss += loss
            batches += 1
            # Backprop (MSE)
            m = Xb.shape[0]
            d_out = (2.0 / m) * err
            dW2 = a1.T @ d_out
            db2 = d_out.sum(axis=0)
            d_a1 = d_out @ W2.T
            d_z1 = d_a1 * (z1 > 0)
            dW1 = Xb.T @ d_z1
            db1 = d_z1.sum(axis=0)
            W1 -= LEARNING_RATE * (dW1 + WEIGHT_DECAY * W1)
            b1 -= LEARNING_RATE * db1
            W2 -= LEARNING_RATE * (dW2 + WEIGHT_DECAY * W2)
            b2 -= LEARNING_RATE * db2
        train_loss = epoch_loss / max(1, batches)
        _, _, val_out = forward(X_va)
        val_loss = float(np.mean((val_out - y_va) ** 2))

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"loss diverged to NaN/Inf at epoch {epoch}"
            )

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        write_metrics(epoch, train_loss, val_loss, "RUNNING", history)
        print(f"[epoch {epoch:03d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    write_metrics(history[-1]["epoch"], history[-1]["train_loss"],
                  history[-1]["val_loss"], "COMPLETED", history)
    print(f"[train] done. final val_loss={history[-1]['val_loss']:.4f}")


# -----------------------------------------------------------------------------
# Pure-stdlib fallback backend (per-sample SGD MLP — zero dependencies)
# -----------------------------------------------------------------------------

def train_pure_python():
    print("[train] backend=pure-python (neither PyTorch nor NumPy installed)")
    rng = random.Random(SEED)

    def make_row():
        x = [rng.uniform(-2.0, 2.0) for _ in range(3)]
        y = (math.sin(3.0 * x[0]) + 0.5 * x[1] ** 2 - x[2]
             + rng.gauss(0.0, NOISE_STD))
        return x, y

    data = [make_row() for _ in range(N_SAMPLES)]
    split = int(0.8 * N_SAMPLES)
    train_set, val_set = data[:split], data[split:]

    d_in, hidden = 3, 16
    scale1 = math.sqrt(2.0 / d_in)
    scale2 = math.sqrt(2.0 / hidden)
    W1 = [[rng.gauss(0, scale1) for _ in range(hidden)] for _ in range(d_in)]
    b1 = [0.0] * hidden
    W2 = [rng.gauss(0, scale2) for _ in range(hidden)]
    b2 = 0.0

    def forward(x):
        z1 = [sum(x[i] * W1[i][j] for i in range(d_in)) + b1[j]
              for j in range(hidden)]
        a1 = [v if v > 0 else 0.0 for v in z1]
        out = sum(a1[j] * W2[j] for j in range(hidden)) + b2
        return z1, a1, out

    history = []
    lr = LEARNING_RATE
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(train_set)
        epoch_loss = 0.0
        for x, y in train_set:
            z1, a1, out = forward(x)
            err = out - y
            epoch_loss += err * err
            d_out = 2.0 * err
            for j in range(hidden):
                if z1[j] > 0:
                    d_z1 = d_out * W2[j]
                    for i in range(d_in):
                        W1[i][j] -= lr * d_z1 * x[i]
                    b1[j] -= lr * d_z1
                W2[j] -= lr * d_out * a1[j]
            b2 -= lr * d_out
        train_loss = epoch_loss / max(1, len(train_set))
        val_loss = sum((forward(x)[2] - y) ** 2 for x, y in val_set) \
            / max(1, len(val_set))

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(f"loss diverged to NaN/Inf at epoch {epoch}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        write_metrics(epoch, train_loss, val_loss, "RUNNING", history)
        print(f"[epoch {epoch:03d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    write_metrics(history[-1]["epoch"], history[-1]["train_loss"],
                  history[-1]["val_loss"], "COMPLETED", history)
    print(f"[train] done. final val_loss={history[-1]['val_loss']:.4f}")


def main() -> int:
    try:
        try:
            import torch  # noqa: F401
            train_torch()
        except ImportError:
            try:
                import numpy  # noqa: F401
                train_numpy()
            except ImportError:
                train_pure_python()
        return 0
    except Exception as exc:  # write a CRASHED record before re-raising
        write_metrics(None, None, None, "CRASHED", [],
                      {"error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
