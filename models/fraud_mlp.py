"""
fraud_mlp.py

Purpose:
    Define the neural network architecture used for fraud detection by
    each data center, and by the centralized baseline.

Phase 2 Scope:
    A configurable Multi-Layer Perceptron (MLP) for binary fraud
    classification on structured/tabular transaction data, plus the
    supporting utilities (prediction, persistence, summary, seeding)
    that every future training/inference phase relies on.

    This module intentionally contains NO training loop, optimizer,
    loss function, or evaluation pipeline -- only the model layer.

Architecture:
    Input
      -> [ Linear -> (BatchNorm1d) -> Activation -> (Dropout) ] * len(hidden_dims)
      -> Linear
      -> Sigmoid

    Hidden block count, widths, dropout rate, activation function, and
    whether batch normalization / dropout are applied are all
    configurable via the constructor -- no architecture values are
    hardcoded in the model body.

Public Interface:
    class FraudMLP(nn.Module)

    Methods:
        __init__(...)
        forward(x)
        predict(x, threshold=0.5)
        predict_proba(x)
        save(path)
        load(path)            [classmethod]
        summary()
        set_seed(seed)        [staticmethod]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

# Activation functions selectable via the constructor's `activation`
# argument. Kept as a module-level registry so adding a new activation
# never requires touching the model body.
_ACTIVATIONS: Dict[str, type] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}

# Activations for which Kaiming (He) initialization is the standard
# recommendation. Anything else (e.g. "tanh") falls back to Xavier.
_KAIMING_ACTIVATIONS = {"relu", "leaky_relu", "elu"}

ArrayLike = Union[torch.Tensor, np.ndarray]


class FraudMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for binary fraud classification.

    The network is a stack of ``Linear -> [BatchNorm1d] -> Activation ->
    [Dropout]`` blocks (one per entry in ``hidden_dims``), followed by a
    final ``Linear -> Sigmoid`` output layer producing a fraud
    probability per sample.

    Args:
        input_dim: Number of input features. Must be a positive integer.
        hidden_dims: Sizes of each hidden layer, in order. Must be a
            non-empty sequence of positive integers. Defaults to
            ``[128, 64]``.
        dropout_rate: Dropout probability applied after each hidden
            block's activation, when ``use_dropout=True``. Must be in
            ``[0.0, 1.0)``. Defaults to ``0.3``.
        output_dim: Number of output units. ``1`` for standard binary
            classification (a single fraud probability). Must be a
            positive integer. Defaults to ``1``.
        activation: Name of the activation function used in every hidden
            block. One of ``{"relu", "leaky_relu", "gelu", "elu",
            "tanh"}``. Defaults to ``"relu"``.
        use_batch_norm: Whether to insert a ``BatchNorm1d`` after each
            hidden ``Linear`` layer. Defaults to ``True``.
        use_dropout: Whether to insert a ``Dropout`` layer after each
            hidden block's activation. Defaults to ``True``.

    Expected Input:
        A 2D float tensor (or array convertible to one) of shape
        ``(batch_size, input_dim)``.

    Expected Output:
        ``forward()`` returns a 2D float tensor of shape
        ``(batch_size, output_dim)`` containing values in ``[0, 1]``
        (fraud probabilities), since the final layer is a Sigmoid.

    Raises:
        ValueError: If any constructor argument is invalid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout_rate: float = 0.3,
        output_dim: int = 1,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_dropout: bool = True,
    ) -> None:
        super().__init__()

        self._validate_constructor_args(
            input_dim, hidden_dims, dropout_rate, output_dim, activation
        )

        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = float(dropout_rate)
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.use_batch_norm = bool(use_batch_norm)
        self.use_dropout = bool(use_dropout)

        # Stored so save()/load() can reconstruct an identical model
        # without the caller having to remember the original arguments.
        self.config: Dict[str, object] = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "use_batch_norm": self.use_batch_norm,
            "use_dropout": self.use_dropout,
        }

        self.network = self._build_network()
        self._initialize_weights()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_constructor_args(
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout_rate: float,
        output_dim: int,
        activation: str,
    ) -> None:
        """Validate constructor arguments, raising ValueError on failure."""
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}")

        if not hidden_dims or len(list(hidden_dims)) == 0:
            raise ValueError("hidden_dims must be a non-empty sequence of positive integers")
        if any((not isinstance(h, int)) or h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must contain only positive integers, got {hidden_dims!r}")

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate!r}")

        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim!r}")

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )

    def _build_network(self) -> nn.Sequential:
        """Assemble the Linear/BatchNorm/Activation/Dropout stack as an
        ``nn.Sequential``, followed by the final Linear + Sigmoid output
        layer.
        """
        activation_cls = _ACTIVATIONS[self.activation_name]
        layers: List[nn.Module] = []

        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_cls())
            if self.use_dropout:
                layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize all Linear layer weights and biases.

        Strategy:
            - For ReLU-family activations ("relu", "leaky_relu", "elu"),
              weights use Kaiming (He) normal initialization, which is
              the standard recommendation for these nonlinearities since
              it accounts for the activation zeroing out roughly half of
              its inputs.
            - For other activations (e.g. "tanh", "gelu"), weights use
              Xavier (Glorot) uniform initialization, the standard
              recommendation for saturating/symmetric nonlinearities.
            - All biases are initialized to zero, a safe default that
              avoids introducing asymmetric bias before training starts.
        """
        use_kaiming = self.activation_name in _KAIMING_ACTIVATIONS

        def _init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if use_kaiming:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.network.apply(_init_module)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass, supporting batched inference.

        Args:
            x: Float tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Float tensor of shape ``(batch_size, output_dim)`` with
            values in ``[0, 1]`` (fraud probabilities per sample).

        Raises:
            ValueError: If ``x`` is not 2D or its last dimension does not
                match ``input_dim``.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected a 2D input tensor of shape (batch_size, {self.input_dim}), "
                f"got tensor with shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, got {x.shape[1]}"
            )
        return self.network(x)

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    def _prepare_input(self, x: ArrayLike) -> torch.Tensor:
        """Convert ``x`` to a float32 tensor on the same device as the model."""
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, got {type(x)!r}")
        return x.to(device=device, dtype=torch.float32)

    def predict_proba(self, x: ArrayLike) -> torch.Tensor:
        """Return fraud probabilities for a batch of inputs.

        Runs inference in evaluation mode with gradients disabled, and
        restores the model's previous training/eval mode afterward.

        Args:
            x: Input features, shape ``(batch_size, input_dim)``. Accepts
                either a ``torch.Tensor`` or a ``numpy.ndarray``.

        Returns:
            Float tensor of shape ``(batch_size, output_dim)`` with fraud
            probabilities in ``[0, 1]``, on the same device as the model.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                prepared = self._prepare_input(x)
                probs = self.forward(prepared)
        finally:
            self.train(was_training)
        return probs

    def predict(self, x: ArrayLike, threshold: float = 0.5) -> torch.Tensor:
        """Return binary fraud predictions for a batch of inputs.

        Args:
            x: Input features, shape ``(batch_size, input_dim)``. Accepts
                either a ``torch.Tensor`` or a ``numpy.ndarray``.
            threshold: Decision threshold applied to the predicted fraud
                probability. Must be in ``[0.0, 1.0]``. Defaults to
                ``0.5``.

        Returns:
            Long tensor of shape ``(batch_size, output_dim)`` containing
            ``0`` (not fraud) or ``1`` (fraud) predictions.

        Raises:
            ValueError: If ``threshold`` is not in ``[0.0, 1.0]``.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold!r}")
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model's configuration and learned weights to disk.

        The saved checkpoint contains both the constructor configuration
        and the state dict, so :meth:`load` can fully reconstruct an
        equivalent model without the caller needing to know the original
        constructor arguments.

        Args:
            path: Destination file path for the checkpoint (e.g.
                ``"model.pt"``). Parent directories are created if
                needed.

        Raises:
            OSError: If the checkpoint cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"config": self.config, "state_dict": self.state_dict()}
        try:
            torch.save(checkpoint, path)
        except OSError as exc:
            raise OSError(f"Failed to save model checkpoint to {path}: {exc}") from exc

    @classmethod
    def load(
        cls, path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None
    ) -> "FraudMLP":
        """Load a model previously saved with :meth:`save`.

        Args:
            path: Path to a checkpoint file written by :meth:`save`.
            map_location: Optional device to map the loaded tensors onto
                (e.g. ``"cpu"`` or ``"cuda"``). Defaults to ``"cpu"`` if
                not specified, so a GPU-trained model can always be
                loaded on a machine without CUDA.

        Returns:
            A new ``FraudMLP`` instance with the saved configuration and
            weights.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the checkpoint is missing required keys.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=map_location or "cpu")

        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint at {path} is missing required keys "
                f"('config', 'state_dict'); found {list(checkpoint.keys())}"
            )

        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Build and print a human-readable summary of the model.

        Returns:
            The summary text (also printed to stdout), describing the
            architecture configuration, per-layer structure, and
            trainable/total parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lines = [
            "FraudMLP Summary",
            "=" * 50,
            f"input_dim        : {self.input_dim}",
            f"hidden_dims      : {self.hidden_dims}",
            f"output_dim       : {self.output_dim}",
            f"activation       : {self.activation_name}",
            f"dropout_rate     : {self.dropout_rate}",
            f"use_batch_norm   : {self.use_batch_norm}",
            f"use_dropout      : {self.use_dropout}",
            "-" * 50,
            "Layers:",
        ]
        for name, module in self.network.named_children():
            lines.append(f"  ({name}) {module}")
        lines.append("-" * 50)
        lines.append(f"Total parameters     : {total_params:,}")
        lines.append(f"Trainable parameters : {trainable_params:,}")
        lines.append("=" * 50)

        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Seed all relevant random number generators for reproducibility.

        Seeds Python's ``random`` module, NumPy, and PyTorch (CPU and, if
        available, all CUDA devices), and configures cuDNN to favor
        determinism over performance. This does not modify model state;
        it only affects subsequent random operations (weight init, data
        shuffling, dropout masks, etc.).

        Args:
            seed: Seed value to use across all RNGs. Defaults to ``42``.

        Raises:
            ValueError: If ``seed`` is not an integer.
        """
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
