from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
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
    loss: float | None = None
    val_loss: float | None = None

    mastery: float | None = None
    baseline_loss: float | None = None
    improvement: float | None = None

    sources: int = 0
    cycles: int = 0
    assessment: str = ""
    next_focus: str = ""
    message: str = ""


class Trainer:
    def __init__(
        self,
        store,
        memory,
    ):
        self.store = store
        self.memory = memory

        self.status = Status()

        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            return asdict(self.status)

    def start(
        self,
        goal: str,
        minutes: float = 20,
    ):
        with self.lock:
            if self.status.running:
                raise RuntimeError(
                    "A training session is already running."
                )

            self.stop_event.clear()

            self.status = Status(
                running=True,
                goal=goal.strip(),
                requested_minutes=float(
                    max(
                        1,
                        min(
                            120,
                            minutes,
                        ),
                    )
                ),
                phase="baseline",
                message="Measuring the current model.",
            )

        self.thread = threading.Thread(
            target=self._run,
            args=(
                goal.strip(),
                self.status.requested_minutes,
            ),
            daemon=True,
        )

        self.thread.start()

    def halt(self):
        self.stop_event.set()

    # ---------------------------------------------------------
    # Dataset construction
    # ---------------------------------------------------------

    def _build_examples(
        self,
        goal: str,
        docs: list[dict],
    ) -> list[str]:

        examples: list[str] = []

        for document in docs:
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

            # Train the model to continue coherent prose
            # from researched material.
            examples.append(
                f"Topic: {title}\n{text}"
            )

        # General conversational training.
        # These are training examples, not runtime response rules.
        examples.extend(
            [
                "User: hello\n"
                "Assistant: Hello! What would you like to talk about?",

                "User: hi\n"
                "Assistant: Hi! What are you working on?",

                "User: how are you?\n"
                "Assistant: I'm ready to help. What would you like to explore?",

                "User: what are you?\n"
                "Assistant: I am Verdant-1.0, a learned AI system.",

                "User: explain something simply\n"
                "Assistant: Start with the central idea, then connect the important details.",
            ]
        )

        # Goal-specific instructional examples.
        examples.extend(
            self._goal_examples(
                goal,
                docs,
            )
        )

        return examples

    def _goal_examples(
        self,
        goal: str,
        docs: list[dict],
    ) -> list[str]:

        result: list[str] = []

        extracted = []

        for document in docs:
            text = document.get(
                "text",
                "",
            )

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if 80 <= len(sentence) <= 700:
                    extracted.append(
                        sentence
                    )

        # Limit duplicate material.
        seen = set()

        for sentence in extracted:
            key = sentence.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                f"Goal: {goal}\n"
                f"Learned concept: {sentence}"
            )

            if len(result) >= 160:
                break

        return result

    def _split_holdout(
        self,
        examples: list[str],
    ):
        rng = random.Random(2026)

        shuffled = list(examples)
        rng.shuffle(shuffled)

        holdout_count = max(
            24,
            min(
                96,
                len(shuffled) // 5,
            ),
        )

        holdout = shuffled[
            :holdout_count
        ]

        training = shuffled[
            holdout_count:
        ]

        return training, holdout

    # ---------------------------------------------------------
    # Batching
    # ---------------------------------------------------------

    def _make_batch(
        self,
        examples: list[str],
        batch_size: int = 8,
        sequence_length: int = 256,
    ):

        xs = []
        ys = []

        usable = [
            example
            for example in examples
            if len(example.encode("utf-8")) >= 8
        ]

        if not usable:
            raise RuntimeError(
                "No usable training examples."
            )

        for _ in range(batch_size):

            text = random.choice(
                usable
            )

            encoded = text.encode(
                "utf-8",
                errors="replace",
            )

            if len(encoded) < sequence_length + 1:
                repeats = (
                    sequence_length + 1
                ) // max(
                    1,
                    len(encoded),
                ) + 1

                encoded *= repeats

            start = random.randint(
                0,
                len(encoded)
                - sequence_length
                - 1,
            )

            x = encoded[
                start:
                start + sequence_length
            ]

            y = encoded[
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

    # ---------------------------------------------------------
    # Real learning
    # ---------------------------------------------------------

    def _train_step(
        self,
        examples,
    ):

        self.store.model.train()

        x, y = self._make_batch(
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
                1.0,
            )

            self.store.optimizer.step()

            self.store.step += 1

        return float(
            loss.detach()
        )

    def _evaluate(
        self,
        examples,
        max_examples: int = 32,
    ):
        if not examples:
            return None

        selected = examples[
            :max_examples
        ]

        total_loss = 0.0
        count = 0

        self.store.model.eval()

        with torch.no_grad():

            for example in selected:

                encoded = example.encode(
                    "utf-8",
                    errors="replace",
                )

                if len(encoded) < 8:
                    continue

                sequence_length = min(
                    256,
                    len(encoded) - 1,
                )

                x = torch.tensor(
                    [
                        list(
                            encoded[
                                :sequence_length
                            ]
                        )
                    ],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    [
                        list(
                            encoded[
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

                total_loss += float(
                    loss
                )

                count += 1

        if count == 0:
            return None

        return total_loss / count

    # ---------------------------------------------------------
    # Capability measurement
    # ---------------------------------------------------------

    def _mastery(
        self,
        baseline_loss,
        validation_loss,
    ):

        if (
            baseline_loss is None
            or validation_loss is None
        ):
            return 0.0

        # Positive improvement against unseen material.
        improvement = (
            baseline_loss
            - validation_loss
        )

        # Convert that improvement into a bounded score.
        score = (
            0.5
            + improvement / 4.0
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    def _weak_focus(
        self,
        goal: str,
        docs: list[dict],
    ):

        corpus = " ".join(
            document.get(
                "text",
                "",
            )
            for document in docs
        )

        words = [
            word
            for word in re.findall(
                r"[A-Za-z]{4,}",
                corpus,
            )
        ]

        if not words:
            return goal

        counts = {}

        for word in words:
            key = word.lower()
            counts[key] = (
                counts.get(key, 0)
                + 1
            )

        uncommon = sorted(
            counts,
            key=counts.get,
        )

        return (
            f"{goal} "
            + " ".join(
                uncommon[:8]
            )
        )

    # ---------------------------------------------------------
    # Training session
    # ---------------------------------------------------------

    def _run(
        self,
        goal: str,
        minutes: float,
    ):

        started = time.time()

        try:

            # ---------------------------------------------
            # Research
            # ---------------------------------------------

            with self.lock:
                self.status.phase = (
                    "research"
                )

                self.status.message = (
                    "Finding new material related to the goal."
                )

            seen = set(
                self.memory.data[
                    "learning"
                ].get(
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

            self.memory.data[
                "learning"
            ][
                "seen_sources"
            ] = list(seen)[-1000:]

            self.memory.save()

            # ---------------------------------------------
            # Dataset
            # ---------------------------------------------

            examples = self._build_examples(
                goal,
                documents,
            )

            training_examples, holdout = (
                self._split_holdout(
                    examples
                )
            )

            # ---------------------------------------------
            # Baseline
            # ---------------------------------------------

            baseline_loss = self._evaluate(
                holdout
            )

            with self.lock:
                self.status.baseline_loss = (
                    baseline_loss
                )

                self.status.sources = len(
                    documents
                )

                self.status.phase = (
                    "learn"
                )

                self.status.message = (
                    "Running real learning updates."
                )

            # ---------------------------------------------
            # Main training loop
            # ---------------------------------------------

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

                cycle_losses = []

                # Bounded chunk so status remains responsive.
                for _ in range(48):

                    if (
                        time.time() >= deadline
                        or self.stop_event.is_set()
                    ):
                        break

                    cycle_losses.append(
                        self._train_step(
                            training_examples
                        )
                    )

                validation_loss = (
                    self._evaluate(
                        holdout
                    )
                )

                mastery = self._mastery(
                    baseline_loss,
                    validation_loss,
                )

                average_loss = (
                    sum(cycle_losses)
                    / max(
                        1,
                        len(cycle_losses),
                    )
                )

                improvement = (
                    None
                    if (
                        baseline_loss is None
                        or validation_loss is None
                    )
                    else
                    baseline_loss
                    - validation_loss
                )

                with self.lock:
                    self.status.cycles = cycle
                    self.status.steps = (
                        self.store.step
                    )
                    self.status.loss = (
                        average_loss
                    )
                    self.status.val_loss = (
                        validation_loss
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

                # Persist every cycle.
                self.store.save()

                # -----------------------------------------
                # Adaptive remediation
                # -----------------------------------------

                if mastery < 0.80:

                    focus = self._weak_focus(
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
                            "Finding additional material for weak areas."
                        )

                    extra_documents = (
                        research_goal(
                            focus,
                            limit=6,
                            seen_titles=seen,
                        )
                    )

                    for document in extra_documents:
                        seen.add(
                            document[
                                "title"
                            ].lower()
                        )

                    if extra_documents:

                        documents.extend(
                            extra_documents
                        )

                        examples = (
                            self._build_examples(
                                goal,
                                documents,
                            )
                        )

                        training_examples, holdout = (
                            self._split_holdout(
                                examples
                            )
                        )

                else:

                    with self.lock:
                        self.status.phase = (
                            "consolidate"
                        )

                        self.status.message = (
                            "Consolidating learned state and replaying old material."
                        )

                # Give the frontend a chance to update.
                time.sleep(0.02)

            # ---------------------------------------------
            # Final evaluation
            # ---------------------------------------------

            final_validation = (
                self._evaluate(
                    holdout
                )
            )

            final_mastery = (
                self._mastery(
                    baseline_loss,
                    final_validation,
                )
            )

            self.store.save()

            session = {
                "goal": goal,
                "steps": self.store.step,
                "sources": len(documents),
                "baseline_loss": baseline_loss,
                "final_validation_loss": final_validation,
                "mastery": final_mastery,
                "finished_at": time.time(),
            }

            self.memory.data[
                "learning"
            ][
                "sessions"
            ].append(session)

            self.memory.save()

            with self.lock:

                self.status.running = False
                self.status.phase = "complete"
                self.status.elapsed = (
                    time.time()
                    - started
                )
                self.status.mastery = (
                    final_mastery
                )

                self.status.message = (
                    "Training complete."
                )

        except Exception as exc:

            with self.lock:
                self.status.running = False
                self.status.phase = "error"
                self.status.elapsed = (
                    time.time()
                    - started
                )
                self.status.message = str(
                    exc
                )
