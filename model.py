from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import threading

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    embed_dim: int = 512
    hidden_dim: int = 768
    heads: int = 8
    layers: int = 6
    context: int = 768
    dropout: float = 0.10


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                "embed_dim must be divisible by heads"
            )

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(
            dim,
            dim * 3,
        )

        self.out = nn.Linear(
            dim,
            dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
    ):

        batch, length, dim = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        q = q.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        scale = self.head_dim ** -0.5

        scores = (
            q @ k.transpose(
                -2,
                -1,
            )
        ) * scale

        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        weights = self.dropout(
            weights
        )

        output = weights @ v

        output = output.transpose(
            1,
            2,
        ).contiguous().view(
            batch,
            length,
            dim,
        )

        return self.out(
            output
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        dropout: float,
    ):
        super().__init__()

        inner = dim * 4

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                inner,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                inner,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        cfg: ModelConfig,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            cfg.embed_dim
        )

        self.attention = (
            CausalSelfAttention(
                cfg.embed_dim,
                cfg.heads,
                cfg.dropout,
            )
        )

        self.norm2 = nn.LayerNorm(
            cfg.embed_dim
        )

        self.feed_forward = (
            FeedForward(
                cfg.embed_dim,
                cfg.dropout,
            )
        )

    def forward(
        self,
        x,
    ):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class VerdantCore(nn.Module):
    """
    From-scratch causal learned core.

    No answer database.
    No topic-specific response rules.
    """

    def __init__(
        self,
        cfg: ModelConfig,
    ):
        super().__init__()

        self.cfg = cfg

        self.token_embedding = (
            nn.Embedding(
                cfg.vocab_size,
                cfg.embed_dim,
            )
        )

        self.position_embedding = (
            nn.Embedding(
                cfg.context,
                cfg.embed_dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg)
                for _ in range(
                    cfg.layers
                )
            ]
        )

        self.norm = nn.LayerNorm(
            cfg.embed_dim
        )

        self.readout = nn.Linear(
            cfg.embed_dim,
            cfg.vocab_size,
            bias=False,
        )

        # Weight tying improves parameter efficiency.
        self.readout.weight = (
            self.token_embedding.weight
        )

    def forward(
        self,
        tokens,
    ):

        batch, length = tokens.shape

        if length > self.cfg.context:
            tokens = tokens[
                :,
                -self.cfg.context:
            ]

            length = tokens.shape[1]

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(
                positions
            )[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.readout(x)

    @torch.no_grad()
    def latent(
        self,
        tokens,
    ):
        batch, length = tokens.shape

        if length > self.cfg.context:
            tokens = tokens[
                :,
                -self.cfg.context:
            ]
            length = tokens.shape[1]

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(
                positions
            )[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        return self.norm(x[:, -1])


class ModelStore:

    def __init__(
        self,
        path: str | Path,
        tokenizer,
    ):
        self.path = Path(path)
        self.lock = threading.RLock()

        self.tokenizer = tokenizer

        self.model = None
        self.optimizer = None

        self.step = 0
        self.best_validation = None

        self._load()

    def _new_model(
        self,
        cfg: ModelConfig | None = None,
    ):
        configuration = (
            cfg
            or ModelConfig(
                vocab_size=
                    self.tokenizer.vocab_size
            )
        )

        model = VerdantCore(
            configuration
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        return model, optimizer

    def initialize(
        self,
    ):
        with self.lock:
            self.model, self.optimizer = (
                self._new_model()
            )

    def _load(self):

        if not self.path.exists():
            self.initialize()
            return

        payload = torch.load(
            self.path,
            map_location="cpu",
        )

        config_data = payload.get(
            "config"
        )

        if config_data:
            cfg = ModelConfig(
                **config_data
            )
        else:
            cfg = ModelConfig(
                vocab_size=
                    self.tokenizer.vocab_size
            )

        self.model = VerdantCore(
            cfg
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
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

        self.best_validation = payload.get(
            "best_validation"
        )

    def save(
        self,
    ):

        temporary = (
            self.path.with_suffix(
                ".tmp"
            )
        )

        self.path.parent.mkdir(
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

                    "best_validation":
                        self.best_validation,

                    "config":
                        asdict(
                            self.model.cfg
                        ),
                },
                temporary,
            )

            temporary.replace(
                self.path
            )

    def parameter_count(
        self,
    ):

        with self.lock:

            return sum(
                parameter.numel()
                for parameter
                in self.model.parameters()
            )

    def snapshot(
        self,
    ):

        with self.lock:

            return {
                key:
                    value.detach().clone()
                for key, value
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
