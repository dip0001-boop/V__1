from __future__ import annotations

import asyncio
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


app = FastAPI(
    title="Verdant-1.0",
    version="1.0.0",
)

ROOT = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=ROOT),
    name="static",
)

store = ModelStore(ROOT / "verdant_state.pt")
memory = MemoryStore(ROOT / "verdant_memory.json")
trainer = Trainer(store, memory)


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


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def index():
    path = ROOT / "index.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found",
        )

    return FileResponse(path)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_step": store.step,
        "parameters": store.parameter_count(),
        "training": trainer.snapshot(),
    }


@app.get("/api/status")
def status():
    return {
        "training": trainer.snapshot(),
        "model_step": store.step,
        "parameters": store.parameter_count(),
    }


def effort_config(effort: str):
    configs = {
        "lite": {
            "max_new": 96,
            "temperature": 0.95,
        },
        "medium": {
            "max_new": 160,
            "temperature": 0.85,
        },
        "max": {
            "max_new": 256,
            "temperature": 0.72,
        },
    }

    return configs.get(
        (effort or "medium").lower(),
        configs["medium"],
    )


def sample(
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

    generated = []
    hidden = None

    store.model.eval()

    with torch.no_grad():

        _, hidden = store.model(
            x,
            hidden,
        )

        last = x[:, -1:]

        for _ in range(max_new):

            logits, hidden = store.model(
                last,
                hidden,
            )

            logits = logits[:, -1, :]

            logits = logits / max(
                0.25,
                temperature,
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            values, indices = torch.sort(
                probabilities,
                descending=True,
            )

            cumulative = torch.cumsum(
                values,
                dim=-1,
            )

            keep = cumulative <= 0.92
            keep[..., 0] = True

            values = values * keep

            denominator = values.sum(
                dim=-1,
                keepdim=True,
            )

            values = values / denominator

            picked = torch.multinomial(
                values,
                1,
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


@app.post("/api/chat")
def chat(request: ChatRequest):

    chat_id = (
        request.chat_id
        or str(uuid.uuid4())
    )

    conversation = memory.chat(
        chat_id
    )

    recent = conversation[
        "messages"
    ][-10:]

    context_parts = []

    for message in recent:
        context_parts.append(
            f"{message['role'].title()}: "
            f"{message['content']}"
        )

    context = "\n".join(
        context_parts
    )

    memory.add_message(
        chat_id,
        "user",
        request.message,
    )

    settings = effort_config(
        request.effort
    )

    prompt = (
        context
        + "\nUser: "
        + request.message
        + "\nAssistant:"
    )

    try:
        answer = sample(
            prompt,
            settings["max_new"],
            settings["temperature"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    memory.add_message(
        chat_id,
        "assistant",
        answer,
    )

    updated = memory.chat(
        chat_id
    )

    chat_summary = summarize(
        updated["messages"]
    )

    memory.set_summary(
        chat_id,
        chat_summary,
    )

    return {
        "chat_id": chat_id,
        "response": answer,
        "summary": chat_summary,
        "effort": request.effort,
        "training_step": store.step,
    }


@app.post("/api/train/start")
def start_training(
    request: TrainRequest,
):
    try:
        trainer.start(
            request.goal,
            request.minutes,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return {
        "ok": True,
        "goal": request.goal,
        "minutes": request.minutes,
    }


@app.post("/api/train/stop")
def stop_training():
    trainer.halt()

    return {
        "ok": True,
    }
