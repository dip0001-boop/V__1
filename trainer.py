from __future__ import annotations

from dataclasses import dataclass, asdict
import random
import re
import threading
import time

import torch
import torch.nn.functional as F

from research import research_goal


@dataclass
class Status:
    running: bool = False
    goal: str = ""
    phase: str = "idle"

    elapsed: float = 0.0
    requested_minutes: float = 20.0

    steps: int = 0
    cycles: int = 0

    train_loss: float | None = None
    val_loss: float | None = None
    baseline_loss: float | None = None
    improvement: float | None = None

    sources: int = 0
    examples: int = 0

    mastery: float | None = None

    next_focus: str = ""

    message: str = ""
    last_evaluation: str = ""


class Trainer:

    def __init__(
        self,
        store,
        memory,
    ):

        self.store = store
        self.memory = memory

        self.status = Status()

        self.thread = None
        self.stop_event = threading.Event()

        self.lock = threading.RLock()

        self.rng = random.Random()

    def snapshot(self):

        with self.lock:
            return asdict(
                self.status
            )

    def start(
        self,
        goal,
        minutes=20,
    ):

        goal = goal.strip()

        with self.lock:

            if self.status.running:
                raise RuntimeError(
                    "A training session is already running."
                )

            self.stop_event.clear()

            self.status = Status(
                running=True,
                goal=goal,
                requested_minutes=max(
                    1.0,
                    min(
                        120.0,
                        float(minutes),
                    ),
                ),
                phase="research",
                message=(
                    "Starting real training."
                ),
            )

            requested = (
                self.status.requested_minutes
            )

        self.thread = threading.Thread(
            target=self._run,
            args=(goal, requested),
            daemon=True,
        )

        self.thread.start()

    def halt(self):
        self.stop_event.set()

    # ========================================================
    # DATA
    # ========================================================

    def _extract_sentences(
        self,
        text,
    ):

        pieces = re.split(
            r"(?<=[.!?])\s+",
            text or "",
        )

        return [
            piece.strip()
            for piece in pieces
            if 60 <= len(piece.strip()) <= 900
        ]

    def _build_dataset(
        self,
        goal,
        documents,
    ):

        examples = []

        # Raw source learning.
        for document in documents:

            title = document.get(
                "title",
                "",
            )

            text = document.get(
                "text",
                "",
            )

            if not text:
                continue

            examples.append(
                f"Learning goal: {goal}\n"
                f"Source: {title}\n"
                f"{text}"
            )

            sentences = (
                self._extract_sentences(
                    text
                )
            )

            for sentence in sentences[:40]:

                examples.append(
                    f"Goal: {goal}\n"
                    f"Concept: {sentence}\n"
                    f"Explanation: {sentence}"
                )

        # General dialogue training.
        examples.extend(
            [
                (
                    "User: hello\n"
                    "Assistant: Hello! "
                    "What would you like to explore?"
                ),
                (
                    "User: hi\n"
                    "Assistant: Hi! "
                    "What are you working on?"
                ),
                (
                    "User: how are you?\n"
                    "Assistant: I'm ready to help. "
                    "What would you like to talk about?"
                ),
                (
                    "User: explain something simply\n"
                    "Assistant: Start with the central idea, "
                    "then connect the important details."
                ),
            ]
        )

        unique = []
        seen = set()

        for example in examples:

            normalized = re.sub(
                r"\s+",
                " ",
                example,
            ).strip().lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(example)

        self.rng.shuffle(unique)

        return unique

    def _split(
        self,
        examples,
    ):

        examples = list(examples)

        self.rng.shuffle(
            examples
        )

        if len(examples) < 20:
            split = max(
                1,
                len(examples) // 4,
            )
        else:
            split = max(
                20,
                min(
                    120,
                    len(examples) // 5,
                ),
            )

        holdout = examples[:split]
        training = examples[split:]

        return training, holdout

    # ========================================================
    # BATCHES
    # ========================================================

    def _batch(
        self,
        examples,
        batch_size=8,
        sequence_length=384,
    ):

        encoded_examples = []

        for example in examples:

            raw = example.encode(
                "utf-8",
                errors="replace",
            )

            if len(raw) >= 16:
                encoded_examples.append(raw)

        if not encoded_examples:
            raise RuntimeError(
                "No valid training examples."
            )

        xs = []
        ys = []

        for _ in range(
            batch_size
        ):

            raw = self.rng.choice(
                encoded_examples
            )

            if len(raw) < sequence_length + 1:

                repeat_count = (
                    sequence_length + 1
                ) // len(raw) + 1

                raw = raw * repeat_count

            maximum = (
                len(raw)
                - sequence_length
                - 1
            )

            start = self.rng.randint(
                0,
                maximum,
            )

            x = raw[
                start:
                start + sequence_length
            ]

            y = raw[
                start + 1:
                start + sequence_length + 1
            ]

            xs.append(
                list(x)
            )

            ys.append(
                list(y)
            )

        return (
            torch.tensor(
                xs,
                dtype=torch.long,
            ),
            torch.tensor(
                ys,
                dtype=torch.long,
            ),
        )

    # ========================================================
    # REAL TRAINING
    # ========================================================

    def _train_step(
        self,
        examples,
    ):

        self.store.model.train()

        x, y = self._batch(
            examples
        )

        with self.store.lock:

            logits, _ = (
                self.store.model(x)
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.store.model.cfg.vocab_size,
                ),
                y.reshape(-1),
            )

            self.store.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.store.model.parameters(),
                max_norm=1.0,
            )

            self.store.optimizer.step()

            self.store.step += 1

        return float(
            loss.detach()
        )

    def _evaluate(
        self,
        examples,
        maximum=40,
    ):

        if not examples:
            return None

        selected = examples[
            :maximum
        ]

        self.store.model.eval()

        losses = []

        with torch.no_grad():

            for example in selected:

                raw = example.encode(
                    "utf-8",
                    errors="replace",
                )

                if len(raw) < 8:
                    continue

                sequence_length = min(
                    384,
                    len(raw) - 1,
                )

                x = torch.tensor(
                    [
                        list(
                            raw[
                                :sequence_length
                            ]
                        )
                    ],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    [
                        list(
                            raw[
                                1:
                                sequence_length + 1
                            ]
                        )
                    ],
                    dtype=torch.long,
                )

                logits, _ = (
                    self.store.model(x)
                )

                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        self.store.model.cfg.vocab_size,
                    ),
                    y.reshape(-1),
                )

                losses.append(
                    float(loss)
                )

        if not losses:
            return None

        return sum(losses) / len(losses)

    # ========================================================
    # ADAPTIVE LEARNING
    # ========================================================

    def _choose_focus(
        self,
        goal,
        documents,
    ):

        terms = set(
            re.findall(
                r"[A-Za-z]{5,}",
                goal.lower(),
            )
        )

        candidates = []

        for document in documents:

            title = document.get(
                "title",
                "",
            )

            text = document.get(
                "text",
                "",
            )

            score = 0

            lower = (
                title + " " + text[:4000]
            ).lower()

            for term in terms:

                if term in lower:
                    score += 1

            candidates.append(
                (
                    score,
                    title,
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        titles = [
            title
            for _, title
            in candidates
            if title
        ]

        if titles:

            return (
                f"{goal} "
                f"fundamentals "
                f"examples "
                f"{' '.join(titles[-3:])}"
            )

        return (
            f"{goal} fundamentals examples"
        )

    def _mastery(
        self,
        baseline,
        validation,
    ):

        if (
            baseline is None
            or validation is None
        ):
            return None

        improvement = (
            baseline
            - validation
        )

        normalized = (
            0.5
            + improvement / 4.0
        )

        return max(
            0.0,
            min(
                1.0,
                normalized,
            ),
        )

    # ========================================================
    # SESSION
    # ========================================================

    def _run(
        self,
        goal,
        minutes,
    ):

        started = time.time()

        try:

            with self.lock:
                self.status.phase = "research"
                self.status.message = (
                    "Researching material related to the goal."
                )

            learning = self.memory.data.setdefault(
                "learning",
                {
                    "sessions": [],
                    "skills": {},
                    "seen_sources": [],
                },
            )

            seen = set(
                learning.get(
                    "seen_sources",
                    [],
                )
            )

            documents = research_goal(
                goal,
                limit=12,
                seen_titles=seen,
            )

            for document in documents:

                seen.add(
                    document[
                        "title"
                    ].lower()
                )

            learning[
                "seen_sources"
            ] = list(seen)[-1000:]

            self.memory.save()

            examples = (
                self._build_dataset(
                    goal,
                    documents,
                )
            )

            training_examples, holdout = (
                self._split(
                    examples
                )
            )

            baseline = self._evaluate(
                holdout
            )

            with self.lock:

                self.status.baseline_loss = (
                    baseline
                )

                self.status.sources = (
                    len(documents)
                )

                self.status.examples = (
                    len(examples)
                )

                self.status.phase = (
                    "learning"
                )

                self.status.message = (
                    "Applying real optimizer updates."
                )

            deadline = (
                started
                + minutes * 60.0
            )

            cycle = 0

            while (
                time.time() < deadline
                and not self.stop_event.is_set()
            ):

                cycle += 1

                losses = []

                # Actual gradient updates.
                for _ in range(48):

                    if (
                        time.time() >= deadline
                        or self.stop_event.is_set()
                    ):
                        break

                    losses.append(
                        self._train_step(
                            training_examples
                        )
                    )

                validation = self._evaluate(
                    holdout
                )

                train_loss = (
                    sum(losses)
                    / len(losses)
                    if losses
                    else None
                )

                mastery = self._mastery(
                    baseline,
                    validation,
                )

                improvement = (
                    None
                    if (
                        baseline is None
                        or validation is None
                    )
                    else baseline - validation
                )

                with self.lock:

                    self.status.cycles = cycle

                    self.status.steps = (
                        self.store.step
                    )

                    self.status.train_loss = (
                        train_loss
                    )

                    self.status.val_loss = (
                        validation
                    )

                    self.status.mastery = (
                        mastery
                    )

                    self.status.improvement = (
                        improvement
                    )

                    self.status.elapsed = (
                        time.time()
                        - started
                    )

                    self.status.message = (
                        f"Cycle {cycle}: "
                        f"{len(losses)} real updates"
                    )

                self.store.save()

                # Every second cycle, adapt the research target
                # based on the current training state.
                if (
                    mastery is not None
                    and mastery < 0.80
                    and cycle % 2 == 0
                    and time.time() < deadline
                ):

                    focus = self._choose_focus(
                        goal,
                        documents,
                    )

                    with self.lock:

                        self.status.phase = (
                            "remediation"
                        )

                        self.status.next_focus = (
                            focus
                        )

                        self.status.message = (
                            "Weak held-out performance; researching another focus."
                        )

                    extra = research_goal(
                        focus,
                        limit=8,
                        seen_titles=seen,
                    )

                    for document in extra:

                        seen.add(
                            document[
                                "title"
                            ].lower()
                        )

                    if extra:

                        documents.extend(
                            extra
                        )

                        examples = (
                            self._build_dataset(
                                goal,
                                documents,
                            )
                        )

                        training_examples, holdout = (
                            self._split(
                                examples
                            )
                        )

                        baseline = validation

                else:

                    with self.lock:

                        self.status.phase = (
                            "consolidating"
                        )

                        self.status.message = (
                            "Saving learned state and continuing the curriculum."
                        )

                time.sleep(
                    0.02
                )

            final_validation = (
                self._evaluate(
                    holdout
                )
            )

            final_mastery = (
                self._mastery(
                    baseline,
                    final_validation,
                )
            )

            self.store.save()

            learning["sessions"].append(
                {
                    "goal": goal,
                    "steps": self.store.step,
                    "sources": len(documents),
                    "examples": len(examples),
                    "baseline_loss": baseline,
                    "final_validation_loss":
                        final_validation,
                    "mastery_proxy":
                        final_mastery,
                    "finished_at":
                        time.time(),
                }
            )

            learning[
                "seen_sources"
            ] = list(seen)[-1000:]

            self.memory.save()

            with self.lock:

                self.status.running = False
                self.status.phase = (
                    "complete"
                )

                self.status.elapsed = (
                    time.time()
                    - started
                )

                self.status.val_loss = (
                    final_validation
                )

                self.status.mastery = (
                    final_mastery
                )

                self.status.steps = (
                    self.store.step
                )

                self.status.message = (
                    "Training complete."
                )

                self.status.last_evaluation = (
                    "The score is based on held-out "
                    "loss improvement; it is not a claim of semantic mastery."
                )

        except Exception as exc:

            with self.lock:

                self.status.running = False

                self.status.phase = "error"

                self.status.elapsed = (
                    time.time()
                    - started
                )

                self.status.message = (
                    str(exc)
                )
