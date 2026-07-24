"""Baseline model definition for the example experiment.

The Editor agent tunes the hyperparameters below and/or the architecture
in ``build_model``. Keep ``build_model(input_dim)`` as the public factory —
train_example.py imports it by name.
"""

from __future__ import annotations

# ---- Hyperparameters (safe for the Editor agent to tune) -------------------
HIDDEN_DIM = 32
NUM_HIDDEN_LAYERS = 2
DROPOUT = 0.0
ACTIVATION = "relu"  # relu | tanh | gelu

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }

    class MLPRegressor(nn.Module):
        """Simple fully-connected regressor: input_dim -> ... -> 1."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = HIDDEN_DIM,
            num_hidden_layers: int = NUM_HIDDEN_LAYERS,
            dropout: float = DROPOUT,
            activation: str = ACTIVATION,
        ) -> None:
            super().__init__()
            act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
            layers = [nn.Linear(input_dim, hidden_dim), act_cls()]
            for _ in range(max(0, num_hidden_layers - 1)):
                layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    def build_model(input_dim: int) -> "nn.Module":
        return MLPRegressor(input_dim)

except ImportError:
    TORCH_AVAILABLE = False

    def build_model(input_dim: int):  # type: ignore[misc]
        raise ImportError(
            "PyTorch is not installed. train_example.py will fall back to "
            "its NumPy implementation and does not need this factory."
        )
