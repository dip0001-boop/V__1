from __future__ import annotations

from dataclasses import dataclass, asdict
import random
import re
import threading
import time

import torch
import torch.nn.functional as F

from evaluator import Evaluator
from research import research_goal


@dataclass
class TrainingStatus:

    running: bool = False

    goal: str = ""

    phase: str = "idle"

    elapsed: float = 0.0
    requested_minutes: float = 20.0

    step: int = 0
    cycle: int = 0

    train_loss: float | None = None
    validation_loss: float | None = None
    baseline_loss: float | None = None
    improvement: float | None = None

    mastery_proxy: float | None = None

    sources: int = 0
    examples: int = 0

    research_rounds: int = 0

    focus: str = ""

    message: str = ""


class Trainer:

    def __init__(
        self,
        store,
        memory,
    ):

        self.store = store
        self.memory = memory

        self.evaluator = (
            Evaluator(
                store
            )
        )

        self.status = (
            TrainingStatus()
        )

        self.thread = None

        self.stop_event = (
            threading.Event()
        )

        self.lock = (
            threading.RLock()
        )

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

        goal = (
            str(goal)
            .strip()
        )

        if not goal:
            raise ValueError(
                "Training goal cannot be empty."
            )

        with self.lock:

            if self.status.running:
                raise RuntimeError(
                    "Training is already running."
                )

            self.stop_event.clear()

            self.status = (
                TrainingStatus(
                    running=True,
                    goal=goal,
                    requested_minutes=
                        max(
                            1.0,
                            min(
                                120.0,
                                float(minutes),
                            ),
                        ),
                    phase="research",
                    message=
                        "Starting research and baseline evaluation.",
                )
            )

            requested = (
                self.status.requested_minutes
            )

        self.thread = (
            threading.Thread(
                target=self._run,
                args=(
                    goal,
                    requested,
                ),
                daemon=True,
            )
        )

        self.thread.start()

    def halt(self):

        self.stop_event.set()

    # =========================================================
    # Dataset
    # =========================================================

    def _goal_examples(
        self,
        goal,
        documents,
    ):

        examples = []

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

            # Explanatory continuation.
            examples.append(
                (
                    f"Goal: {goal}\n"
                    f"Source: {title}\n"
                    f"Explanation: {text}"
                )
            )

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            for sentence in sentences:

                sentence = (
                    sentence.strip()
                )

                if (
                    len(sentence)
                    < 80
                ):
                    continue

                examples.append(
                    (
                        f"Learning goal: {goal}\n"
                        f"Concept: {sentence}\n"
                        f"Explanation:"
                    )
                )

                examples.append(
                    (
                        f"Topic: {title}\n"
                        f"Important idea: {sentence}"
                    )
                )

        # General conversation learning.
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
                    "User: explain that simply\n"
                    "Assistant: Let's start with the main idea "
                    "and then connect the important details."
                ),
                (
                    "User: I don't understand\n"
                    "Assistant: We can approach it from a simpler angle."
                ),
            ]
        )

        unique = []
        seen = set()

        for item in examples:

            key = re.sub(
                r"\s+",
                " ",
                item,
            ).strip().lower()

            if key in seen:
                continue

            seen.add(key)
            unique.append(
                item
            )

        self.rng.shuffle(
            unique
        )

        return unique

    def _split(
        self,
        examples,
    ):

        items = list(
            examples
        )

        self.rng.shuffle(
            items
        )

        if len(items) < 20:

            cut = max(
                2,
                len(items) // 4,
            )

        else:

            cut = max(
                20,
                min(
                    160,
                    len(items) // 5,
                ),
            )

        return (
            items[cut:],
            items[:cut],
        )

    # =========================================================
    # Batching
    # =========================================================

    def _batch(
        self,
        examples,
        batch_size=8,
    ):

        sequence_length = (
            min(
                512,
                self.store.model.cfg.context,
            )
        )

        usable = []

        for example in examples:

            ids = (
                self.store.tokenizer.encode(
                    example
                )
            )

            if len(ids) >= 8:
                usable.append(
                    ids
                )

        if not usable:
            raise RuntimeError(
                "Training dataset is empty."
            )

        xs = []
        ys = []

        for _ in range(
            batch_size
        ):

            ids = self.rng.choice(
                usable
            )

            if len(ids) < (
                sequence_length + 1
            ):

                repeats = (
                    (
                        sequence_length
                        + 1
                    )
                    // len(ids)
                    + 1
                )

                ids = ids * repeats

            maximum = (
                len(ids)
                - sequence_length
                - 1
            )

            start = self.rng.randint(
                0,
                maximum,
            )

            xs.append(
                ids[
                    start:
                    start
                    + sequence_length
                ]
            )

            ys.append(
                ids[
                    start + 1:
                    start + 1
                    + sequence_length
                ]
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

    # =========================================================
    # Real optimization
    # =========================================================

    def _step(
        self,
        examples,
    ):

        self.store.model.train()

        x, y = self._batch(
            examples
        )

        with self.store.lock:

            logits = (
                self.store.model(x)
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1],
                ),
                y.reshape(-1),
            )

            self.store.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.store.model.parameters(),
                1.0,
            )

            self.store.optimizer.step()

            self.store.step += 1

        return float(
            loss.detach()
        )

    # =========================================================
    # Focus selection
    # =========================================================

    def _focus(
        self,
        goal,
        documents,
    ):

        words = []

        for document in documents:

            text = (
                document.get(
                    "text",
                    "",
                )
            )

            words.extend(
                re.findall(
                    r"[A-Za-z]{5,}",
                    text,
                )
            )

        counts = {}

        for word in words:

            key = (
                word.lower()
            )

            counts[key] = (
                counts.get(
                    key,
                    0,
                )
                + 1
            )

        uncommon = sorted(
            counts,
            key=counts.get,
        )

        if uncommon:

            return (
                f"{goal} "
                + " ".join(
                    uncommon[:10]
                )
            )

        return (
            f"{goal} fundamentals examples"
        )

    # =========================================================
    # Session
    # =========================================================

    def _run(
        self,
        goal,
        minutes,
    ):

        started = time.time()

        try:

            learning = (
                self.memory.data.setdefault(
                    "learning",
                    {},
                )
            )

            learning.setdefault(
                "sessions",
                [],
            )

            learning.setdefault(
                "seen_sources",
                [],
            )

            learning.setdefault(
                "replay",
                [],
            )

            seen = set(
                learning[
                    "seen_sources"
                ]
            )

            # -------------------------------------------------
            # Research
            # -------------------------------------------------

            with self.lock:

                self.status.phase = (
                    "research"
                )

                self.status.message = (
                    "Finding relevant new material."
                )

            documents = (
                research_goal(
                    goal,
                    limit=12,
                    seen_titles=seen,
                )
            )

            for document in documents:

                seen.add(
                    document[
                        "title"
                    ].lower()
                )

            learning[
                "seen_sources"
            ] = list(
                seen
            )[-2000:]

            self.memory.save()

            # -------------------------------------------------
            # Dataset
            # -------------------------------------------------

            examples = (
                self._goal_examples(
                    goal,
                    documents,
                )
            )

            replay = (
                self.memory.replay_examples(
                    maximum=256
                )
            )

            # Replay old learned material.
            examples.extend(
                replay
            )

            training, holdout = (
                self._split(
                    examples
                )
            )

            with self.lock:

                self.status.sources = (
                    len(documents)
                )

                self.status.examples = (
                    len(examples)
                )

            # -------------------------------------------------
            # Baseline
            # -------------------------------------------------

            baseline = (
                self.evaluator.loss(
                    holdout
                )
            )

            with self.lock:

                self.status.baseline_loss = (
                    baseline
                )

                self.status.phase = (
                    "learning"
                )

                self.status.message = (
                    "Performing real gradient updates."
                )

            # -------------------------------------------------
            # Main learning loop
            # -------------------------------------------------

            deadline = (
                started
                + minutes * 60.0
            )

            cycle = 0

            while (
                time.time()
                < deadline
                and not self.stop_event.is_set()
            ):

                cycle += 1

                losses = []

                for _ in range(
                    24
                ):

                    if (
                        time.time()
                        >= deadline
                        or self.stop_event.is_set()
                    ):
                        break

                    losses.append(
                        self._step(
                            training
                        )
                    )

                validation = (
                    self.evaluator.loss(
                        holdout
                    )
                )

                improvement = (
                    self.evaluator.improvement(
                        baseline,
                        validation,
                    )
                )

                score = (
                    self.evaluator.score(
                        baseline,
                        validation,
                    )
                )

                train_loss = (
                    sum(losses)
                    / len(losses)
                    if losses
                    else None
                )

                with self.lock:

                    self.status.step = (
                        self.store.step
                    )

                    self.status.cycle = (
                        cycle
                    )

                    self.status.train_loss = (
                        train_loss
                    )

                    self.status.validation_loss = (
                        validation
                    )

                    self.status.improvement = (
                        improvement
                    )

                    self.status.mastery_proxy = (
                        score
                    )

                    self.status.elapsed = (
                        time.time()
                        - started
                    )

                    self.status.message = (
                        f"Cycle {cycle}: "
                        f"{len(losses)} optimizer updates."
                    )

                # Checkpoint after every cycle.
                self.store.save()

                # -------------------------------------------------
                # Remediation
                # -------------------------------------------------

                if (
                    score < 0.80
                    and cycle % 2 == 0
                    and time.time() < deadline
                ):

                    focus = (
                        self._focus(
                            goal,
                            documents,
                        )
                    )

                    with self.lock:

                        self.status.phase = (
                            "remediation"
                        )

                        self.status.focus = (
                            focus
                        )

                        self.status.message = (
                            "Validation improvement is weak; "
                            "searching for additional material."
                        )

                    extra = (
                        research_goal(
                            focus,
                            limit=8,
                            seen_titles=seen,
                        )
                    )

                    if extra:

                        for document in extra:

                            seen.add(
                                document[
                                    "title"
                                ].lower()
                            )

                        documents.extend(
                            extra
                        )

                        examples = (
                            self._goal_examples(
                                goal,
                                documents,
                            )
                        )

                        replay = (
                            self.memory.replay_examples(
                                maximum=256
                            )
                        )

                        examples.extend(
                            replay
                        )

                        training, holdout = (
                            self._split(
                                examples
                            )
                        )

                        # Re-establish baseline on the new
                        # unseen holdout.
                        baseline = (
                            self.evaluator.loss(
                                holdout
                            )
                        )

                        with self.lock:

                            self.status.research_rounds += 1

                else:

                    with self.lock:

                        self.status.phase = (
                            "consolidating"
                        )

                        self.status.message = (
                            "Saving learned state and replay."
                        )

                # Store a sample of useful training experience.
                for example in (
                    training[:8]
                ):

                    self.memory.add_replay(
                        example,
                        priority=1.0,
                    )

                self.memory.save()

                time.sleep(
                    0.01
                )

            # -------------------------------------------------
            # Final evaluation
            # -------------------------------------------------

            final_loss = (
                self.evaluator.loss(
                    holdout
                )
            )

            final_score = (
                self.evaluator.score(
                    baseline,
                    final_loss,
                )
            )

            self.store.save()

            learning[
                "sessions"
            ].append(
                {
                    "goal":
                        goal,

                    "steps":
                        self.store.step,

                    "sources":
                        len(documents),

                    "examples":
                        len(examples),

                    "baseline_loss":
                        baseline,

                    "final_validation_loss":
                        final_loss,

                    "mastery_proxy":
                        final_score,

                    "finished_at":
                        time.time(),
                }
            )

            learning[
                "seen_sources"
            ] = list(
                seen
            )[-2000:]

            self.memory.save()

            with self.lock:

                self.status.running = False

                self.status.phase = (
                    "complete"
                )

                self.status.step = (
                    self.store.step
                )

                self.status.elapsed = (
                    time.time()
                    - started
                )

                self.status.validation_loss = (
                    final_loss
                )

                self.status.mastery_proxy = (
                    final_score
                )

                self.status.message = (
                    "Training complete."
                )

        except Exception as exc:

            with self.lock:

                self.status.running = False

                self.status.phase = (
                    "error"
                )

                self.status.elapsed = (
                    time.time()
                    - started
                )

                self.status.message = (
                    str(exc)
                )
