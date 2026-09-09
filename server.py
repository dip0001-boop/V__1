from __future__ import annotations

import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from evaluator import Evaluator
from memory import MemoryStore
from model import ModelStore
from summary import summarize
from tokenizer import LearnedTokenizer
from trainer import Trainer


app = FastAPI(
    title="Verdant-1.0",
    version="1.0.0",
)

ROOT = Path(
    __file__
).resolve().parent


# ============================================================
# Tokenizer
# ============================================================

tokenizer = LearnedTokenizer(
    ROOT / "verdant_tokenizer.json"
)


# ============================================================
# Model
# ============================================================

model_path = (
    ROOT / "verdant_state.pt"
)

store = ModelStore(
    model_path,
    tokenizer,
)


# ============================================================
# Memory
# ============================================================

memory = MemoryStore(
    ROOT / "verdant_memory.json"
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    store,
    memory,
)


# ============================================================
# Static frontend
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=ROOT
    ),
    name="static",
)


# ============================================================
# Requests
# ============================================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=12000,
    )

    chat_id: str | None = None

    effort: str = "medium"


class TrainRequest(BaseModel):

    goal: str = Field(
        min_length=2,
        max_length=300,
    )

    minutes: float = Field(
        default=20,
        ge=1,
        le=120,
    )


# ============================================================
# Frontend
# ============================================================

@app.get(
    "/",
    include_in_schema=False,
)
@app.head(
    "/",
    include_in_schema=False,
)
def index():

    return FileResponse(
        ROOT / "index.html"
    )


# ============================================================
# Health
# ============================================================

@app.get(
    "/api/health"
)
def health():

    return {
        "ok": True,
        "step": store.step,
        "parameters":
            store.parameter_count(),
        "vocab_size":
            store.model.cfg.vocab_size,
        "context":
            store.model.cfg.context,
    }


@app.get(
    "/api/status"
)
def status():

    return {
        "training":
            trainer.snapshot(),

        "model_step":
            store.step,

        "parameters":
            store.parameter_count(),

        "vocab_size":
            store.model.cfg.vocab_size,

        "context":
            store.model.cfg.context,
    }


# ============================================================
# Sampling
# ============================================================

def _settings(
    effort,
):

    return {
        "lite":
            (
                96,
                0.95,
            ),

        "medium":
            (
                180,
                0.85,
            ),

        "max":
            (
                300,
                0.72,
            ),
    }.get(
        (effort or "medium").lower(),
        (
            180,
            0.85,
        ),
    )


def _sample(
    prompt,
    max_new,
    temperature,
):

    ids = tokenizer.encode(
        prompt,
        max_len=store.model.cfg.context,
    )

    if not ids:
        ids = [
            tokenizer.token_to_id[
                "<bos>"
            ]
        ]

    current = torch.tensor(
        [ids],
        dtype=torch.long,
    )

    generated = []

    eos = tokenizer.token_to_id.get(
        "<eos>"
    )

    store.model.eval()

    with torch.no_grad():

        for _ in range(
            max_new
        ):

            logits = (
                store.model(
                    current[
                        :,
                        -store.model.cfg.context:
                    ]
                )
            )

            next_logits = (
                logits[
                    :,
                    -1,
                    :
                ]
                / max(
                    0.25,
                    temperature,
                )
            )

            probabilities = (
                torch.softmax(
                    next_logits,
                    dim=-1,
                )
            )

            values, indices = (
                torch.sort(
                    probabilities,
                    descending=True,
                )
            )

            cumulative = (
                torch.cumsum(
                    values,
                    dim=-1,
                )
            )

            keep = (
                cumulative
                <= 0.92
            )

            keep[
                ...,
                0
            ] = True

            values = (
                values
                * keep
            )

            values = (
                values
                / values.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            picked = (
                torch.multinomial(
                    values,
                    1,
                )
            )

            token = (
                indices.gather(
                    -1,
                    picked,
                )
            )

            token_id = int(
                token.item()
            )

            if (
                eos is not None
                and token_id == eos
            ):
                break

            generated.append(
                token_id
            )

            current = torch.cat(
                [
                    current,
                    token,
                ],
                dim=1,
            )

    return tokenizer.decode(
        generated
    ).strip()


# ============================================================
# Chat
# ============================================================

@app.post(
    "/api/chat"
)
def chat(
    request: ChatRequest,
):

    chat_id = (
        request.chat_id
        or str(uuid.uuid4())
    )

    conversation = (
        memory.chat(
            chat_id
        )
    )

    recent = (
        conversation[
            "messages"
        ][-12:]
    )

    context = "\n".join(
        (
            f"{message['role'].title()}: "
            f"{message['content']}"
        )
        for message in recent
    )

    memory.add_message(
        chat_id,
        "user",
        request.message,
    )

    max_new, temperature = (
        _settings(
            request.effort
        )
    )

    prompt = (
        context
        + "\nUser: "
        + request.message
        + "\nAssistant:"
    )

    try:

        answer = _sample(
            prompt,
            max_new,
            temperature,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Inference failed: "
                f"{exc}"
            ),
        ) from exc

    memory.add_message(
        chat_id,
        "assistant",
        answer,
    )

    updated = (
        memory.chat(
            chat_id
        )
    )

    chat_summary = summarize(
        updated["messages"]
    )

    memory.set_summary(
        chat_id,
        chat_summary,
    )

    return {
        "chat_id":
            chat_id,

        "response":
            answer,

        "summary":
            chat_summary,

        "effort":
            request.effort,

        "training_step":
            store.step,
    }


# ============================================================
# Training
# ============================================================

@app.post(
    "/api/train/start"
)
def start_training(
    request: TrainRequest,
):

    try:

        # Build tokenizer vocabulary on first training run.
        if tokenizer.vocab_size < 512:

            seed_text = (
                " ".join(
                    memory.data.get(
                        "learning",
                        {},
                    ).get(
                        "replay",
                        [],
                    )
                    and [
                        str(item)
                        for item in
                        memory.data[
                            "learning"
                        ][
                            "replay"
                        ]
                    ]
                    or []
                )
            )

            if not seed_text:

                seed_text = (
                    request.goal
                    + " "
                    + "learning "
                    + "language "
                    + "reasoning "
                    + "conversation "
                    + "knowledge "
                ) * 100

            tokenizer.build(
                seed_text,
                max_vocab=8192,
            )

            # Tokenizer shape must match a fresh model.
            store.model, store.optimizer = (
                store._new_model()
            )

        trainer.start(
            request.goal,
            request.minutes,
        )

        return {
            "ok": True,
            "goal":
                request.goal,
            "minutes":
                request.minutes,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post(
    "/api/train/stop"
)
def stop_training():

    trainer.halt()

    return {
        "ok": True
    }
