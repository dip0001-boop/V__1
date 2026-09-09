from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import threading

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 256
    embed_dim: int = 512
    hidden_dim: int = 768
    layers: int = 3
    context: int = 512
    dropout: float = 0.10


class VerdantCore(nn.Module):

    def __init__(
        self,
        cfg: ModelConfig | None = None,
    ):
        super().__init__()

        self.cfg = cfg or ModelConfig()

        self.embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embed_dim,
        )

        self.input_projection = nn.Linear(
            self.cfg.embed_dim,
            self.cfg.embed_dim,
        )

        self.rnn = nn.GRU(
            self.cfg.embed_dim,
            self.cfg.hidden_dim,
            num_layers=self.cfg.layers,
            batch_first=True,
            dropout=(
                self.cfg.dropout
                if self.cfg.layers > 1
                else 0.0
            ),
        )

        self.norm = nn.LayerNorm(
            self.cfg.hidden_dim
        )

        self.output_projection = nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.vocab_size,
        )

    def forward(
        self,
        x,
        hidden=None,
    ):

        embedded = self.embedding(x)

        embedded = torch.tanh(
            self.input_projection(
                embedded
            )
        )

        states, hidden = self.rnn(
            embedded,
            hidden,
        )

        states = self.norm(
            states
        )

        logits = self.output_projection(
            states
        )

        return logits, hidden


class ByteTokenizer:

    vocab_size = 256

    def encode(
        self,
        text: str,
        max_len: int | None = None,
    ):

        data = list(
            text.encode(
                "utf-8",
                errors="replace",
            )
        )

        if max_len is not None:
            data = data[-max_len:]

        return data

    def decode(self, ids):

        return bytes(
            int(i) % 256
            for i in ids
        ).decode(
            "utf-8",
            errors="ignore",
        )


class ModelStore:

    def __init__(
        self,
        path="verdant_state.pt",
    ):

        self.path = Path(path)

        self.lock = threading.RLock()

        self.tokenizer = ByteTokenizer()

        self.model = VerdantCore()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=8e-4,
            weight_decay=0.01,
        )

        self.step = 0

        self.best_val = None

        self._load()

    def _load(self):

        if not self.path.exists():
            return

        payload = torch.load(
            self.path,
            map_location="cpu",
        )

        config = payload.get(
            "config"
        )

        if config:
            self.model = VerdantCore(
                ModelConfig(
                    **config
                )
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=8e-4,
                weight_decay=0.01,
            )

        self.model.load_state_dict(
            payload["model"]
        )

        optimizer_state = payload.get(
            "optimizer"
        )

        if optimizer_state:
            try:
                self.optimizer.load_state_dict(
                    optimizer_state
                )
            except Exception:
                pass

        self.step = int(
            payload.get(
                "step",
                0,
            )
        )

        self.best_val = payload.get(
            "best_val"
        )

    def save(
        self,
        path=None,
    ):

        destination = Path(
            path or self.path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.lock:

            torch.save(
                {
                    "model":
                        self.model.state_dict(),

                    "optimizer":
                        self.optimizer.state_dict(),

                    "step":
                        self.step,

                    "best_val":
                        self.best_val,

                    "config":
                        asdict(
                            self.model.cfg
                        ),
                },
                destination,
            )

    def parameter_count(self):

        return sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

    def snapshot(self):

        with self.lock:

            return {
                name:
                    tensor.detach().clone()
                for name, tensor
                in self.model.state_dict().items()
            }

    def restore(
        self,
        state,
    ):

        with self.lock:

            self.model.load_state_dict(
                state
            )
