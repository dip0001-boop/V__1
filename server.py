from __future__ import annotations

import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory import MemoryStore
from model import ModelStore
from summary import summarize
from trainer import Trainer


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Verdant-1.0",
    version="1.0.0",
)

ROOT = Path(__file__).resolve().parent


# ============================================================
# STATIC FILES
# ============================================================

# Serves:
#   /static/app.js
#   /static/styles.css
#
# This fixes the Render 404 problem caused by the old
# /app.js and /styles.css routes.
app.mount(
    "/static",
    StaticFiles(directory=ROOT),
    name="static",
)


# ============================================================
# MODEL / MEMORY / TRAINER
# ============================================================

MODEL_PATH = ROOT / "verdant_state.pt"
MEMORY_PATH = ROOT / "verdant_memory.json"

store = ModelStore(MODEL_PATH)
memory = MemoryStore(MEMORY_PATH)
trainer = Trainer(
    store,
    memory,
)


# ============================================================
# REQUEST MODELS
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
# FRONTEND ROUTES
# ============================================================

@app.get(
    "/",
    include_in_schema=False,
)
def index():
    path = ROOT / "index.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found",
        )

    return FileResponse(path)


@app.head(
    "/",
    include_in_schema=False,
)
def index_head():
    path = ROOT / "index.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found",
        )

    return FileResponse(path)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "step": store.step,
        "parameters": store.parameter_count(),
    }


@app.get("/api/status")
def status():
    return {
        "training": trainer.snapshot(),
        "model_step": store.step,
        "parameters": store.parameter_count(),
    }


# ============================================================
# EFFORT
# ============================================================

def _effort_config(
    effort: str,
) -> tuple[int, float]:
    """
    Runtime-only inference settings.

    These settings control how much computation/output is used.
    They are not intelligence rules.
    """

    configs = {
        "lite": (
            80,
            0.90,
        ),

        "medium": (
            140,
            0.82,
        ),

        "max": (
            220,
            0.72,
        ),
    }

    return configs.get(
        (effort or "medium").lower(),
        configs["medium"],
    )


# ============================================================
# MODEL SAMPLING
# ============================================================

def _sample(
    prompt: str,
    max_new: int,
    temperature: float,
) -> str:

    tokenizer = store.tokenizer

    ids = tokenizer.encode(
        prompt,
        max_len=store.model.cfg.context,
    )

    if not ids:
        ids = [32]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
    )

    generated: list[int] = []

    hidden = None

    store.model.eval()

    with torch.no_grad():

        # Prime the model with the prompt.
        logits, hidden = store.model(
            x,
            hidden,
        )

        last = x[:, -1:]

        for _ in range(max_new):

            logits, hidden = store.model(
                last,
                hidden,
            )

            next_logits = (
                logits[:, -1, :]
                / max(
                    0.25,
                    temperature,
                )
            )

            probabilities = torch.softmax(
                next_logits,
                dim=-1,
            )

            # Nucleus-style stochastic sampling.
            values, indices = torch.sort(
                probabilities,
                descending=True,
            )

            cumulative = torch.cumsum(
                values,
                dim=-1,
            )

            keep = cumulative <= 0.92

            # Always retain at least the best token.
            keep[..., 0] = True

            values = values * keep

            total = values.sum(
                dim=-1,
                keepdim=True,
            )

            values = values / total

            picked = torch.multinomial(
                values,
                num_samples=1,
            )

            next_token = indices.gather(
                -1,
                picked,
            )

            token_id = int(
                next_token.item()
            )

            generated.append(
                token_id
            )

            last = next_token

    return tokenizer.decode(
        generated
    ).strip()


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest,
):
    chat_id = (
        request.chat_id
        or str(uuid.uuid4())
    )

    # Get the existing conversation.
    conversation = memory.chat(
        chat_id
    )

    # Store the user's message.
    memory.add_message(
        chat_id,
        "user",
        request.message,
    )

    # Use recent conversation context.
    recent_messages = conversation[
        "messages"
    ][-8:]

    context = "\n".join(
        (
            f"{message['role'].title()}: "
            f"{message['content']}"
        )
        for message in recent_messages
    )

    max_new, temperature = (
        _effort_config(
            request.effort
        )
    )

    prompt = (
        context
        + "\nAssistant:"
    )

    try:
        answer = _sample(
            prompt=prompt,
            max_new=max_new,
            temperature=temperature,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    # Store Atlas's response.
    memory.add_message(
        chat_id,
        "assistant",
        answer,
    )

    updated_conversation = memory.chat(
        chat_id
    )

    # Update the conversation summary.
    conversation_summary = summarize(
        updated_conversation["messages"]
    )

    memory.set_summary(
        chat_id,
        conversation_summary,
    )

    return {
        "chat_id": chat_id,
        "response": answer,
        "summary": conversation_summary,
        "effort": request.effort,
        "training_step": store.step,
    }


# ============================================================
# TRAINING
# ============================================================

@app.post("/api/train/start")
def start_training(
    request: TrainRequest,
):
    try:

        trainer.start(
            request.goal,
            request.minutes,
        )

        return {
            "ok": True,
            "goal": request.goal,
            "minutes": request.minutes,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post("/api/train/stop")
def stop_training():
    trainer.halt()

    return {
        "ok": True,
    }
