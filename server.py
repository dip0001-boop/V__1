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
    message: str = Field(min_length=1, max_length=12000)
    chat_id: str | None = None
    effort: str = "medium"


class TrainRequest(BaseModel):
    goal: str = Field(min_length=2, max_length=300)
    minutes: float = Field(default=20, ge=1, le=120)


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_step": store.step,
        "parameters": store.parameter_count(),
    }


@app.get("/api/status")
def status():
    return {
        "training": trainer.snapshot(),
        "model_step": store.step,
        "parameters": store.parameter_count(),
    }


def effort_config(effort: str) -> tuple[int, float]:
    return {
        "lite": (96, 0.95),
        "medium": (160, 0.85),
        "max": (256, 0.72),
    }.get(
        (effort or "medium").lower(),
        (160, 0.85),
    )


def sample(prompt: str, max_new: int, temperature: float) -> str:
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
        logits, hidden = store.model(x, hidden)
        last = x[:, -1:]

        for _ in range(max_new):
            logits, hidden = store.model(last, hidden)

            next_logits = (
                logits[:, -1, :]
                / max(0.25, temperature)
            )

            probs = torch.softmax(
                next_logits,
                dim=-1,
            )

            # Nucleus sampling.
            values, indices = torch.sort(
                probs,
                descending=True,
            )

            cumulative = torch.cumsum(
                values,
                dim=-1,
            )

            keep = cumulative <= 0.92
            keep[..., 0] = True

            values = values * keep

            values = values / values.sum(
                dim=-1,
                keepdim=True,
            )

            picked = torch.multinomial(
                values,
                1,
            )

            next_token = indices.gather(
                -1,
                picked,
            )

            generated.append(
                int(next_token.item())
            )

            last = next_token

    return tokenizer.decode(
        generated
    ).strip()


@app.post("/api/chat")
def chat(request: ChatRequest):
    chat_id = request.chat_id or str(uuid.uuid4())

    conversation = memory.chat(chat_id)

    recent = conversation["messages"][-10:]

    memory.add_message(
        chat_id,
        "user",
        request.message,
    )

    context = "\n".join(
        f"{m['role'].title()}: {m['content']}"
        for m in recent
    )

    max_new, temperature = effort_config(
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
            max_new,
            temperature,
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

    updated = memory.chat(chat_id)

    summary = summarize(
        updated["messages"]
    )

    memory.set_summary(
        chat_id,
        summary,
    )

    return {
        "chat_id": chat_id,
        "response": answer,
        "summary": summary,
        "effort": request.effort,
        "training_step": store.step,
    }


@app.post("/api/train/start")
def start_training(request: TrainRequest):
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
