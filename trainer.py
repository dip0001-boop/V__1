from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
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

    retention_baseline: float | None = None
    retention_current: float | None = None
    retention_delta: float | None = None

    transfer_score: float | None = None

    candidate_accepted: int = 0
    candidate_rejected: int = 0

    sources: int = 0
    examples: int = 0
    replay_examples: int = 0

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

        self.evaluator = Evaluator(
            store,
            memory,
        )

        self.status = TrainingStatus()

        self.thread = None

        self.stop_event = threading.Event()

        self.lock = threading.RLock()

        self.rng = random.Random()

    # =========================================================
    # Public state
    # =========================================================

    def snapshot(self):
        with self.lock:
            return asdict(self.status)

    def start(
        self,
        goal,
        minutes=20,
    ):
        goal = str(goal).strip()

        if not goal:
            raise ValueError(
                "Training goal cannot be empty."
            )

        with self.lock:

            if self.status.running:
                raise RuntimeError(
                    "Training is already running."
                )

            requested = max(
                1.0,
                min(
                    120.0,
                    float(minutes),
                ),
            )

            self.stop_event.clear()

            self.status = TrainingStatus(
                running=True,
                goal=goal,
                phase="starting",
                requested_minutes=requested,
                message="Starting training pipeline.",
            )

        self.thread = threading.Thread(
            target=self._run,
            args=(goal, requested),
            daemon=True,
        )

        self.thread.start()

    def halt(self):
        self.stop_event.set()

    # =========================================================
    # Utility
    # =========================================================

    def _update_status(self, **values):

        with self.lock:

            for key, value in values.items():

                if hasattr(
                    self.status,
                    key,
                ):
                    setattr(
                        self.status,
                        key,
                        value,
                    )

    def _remaining(self, deadline):
        return time.time() < deadline

    # =========================================================
    # Dataset construction
    # =========================================================

    def _document_examples(
        self,
        goal,
        documents,
    ):
        """
        Convert researched observations into training experiences.

        Important:
        There are no hard-coded answer mappings here.
        The model is trained on observed material, not
        programmer-written responses.
        """

        examples = []

        for document in documents:

            title = str(
                document.get(
                    "title",
                    "",
                )
            ).strip()

            text = str(
                document.get(
                    "text",
                    "",
                )
            ).strip()

            if len(text) < 200:
                continue

            # Whole-document prediction experience.
            examples.append(
                (
                    f"Learning objective: {goal}\n"
                    f"Source: {title}\n"
                    f"Material:\n{text}"
                )
            )

            # Sentence-group experiences.
            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            usable = [
                sentence.strip()
                for sentence in sentences
                if len(sentence.strip()) >= 60
            ]

            self.rng.shuffle(
                usable
            )

            for sentence in usable[:32]:

                examples.append(
                    (
                        f"Learning objective: {goal}\n"
                        f"Source concept: {title}\n"
                        f"Observation:\n{sentence}"
                    )
                )

        return examples

    def _deduplicate(
        self,
        examples,
    ):
        unique = []
        seen = set()

        for item in examples:

            normalized = re.sub(
                r"\s+",
                " ",
                item,
            ).strip().lower()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            unique.append(
                item
            )

        return unique

    def _split_dataset(
        self,
        examples,
    ):
        """
        Deterministic separation of training and validation
        within the current candidate dataset.

        Validation examples are never optimized directly.
        """

        items = list(
            examples
        )

        self.rng.shuffle(
            items
        )

        if len(items) < 8:

            split = max(
                1,
                len(items) // 3,
            )

        else:

            split = max(
                2,
                min(
                    128,
                    len(items) // 5,
                ),
            )

        validation = items[:split]
        training = items[split:]

        if not training:

            training = validation[:]

        return (
            training,
            validation,
        )

    # =========================================================
    # Token batches
    # =========================================================

    def _batch(
        self,
        examples,
        batch_size=4,
    ):
        sequence_length = min(
            512,
            self.store.model.cfg.context,
        )

        usable = []

        for example in examples:

            ids = self.store.tokenizer.encode(
                example
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

            required = (
                sequence_length + 1
            )

            if len(ids) < required:

                repeats = (
                    required // len(ids)
                ) + 1

                ids = (
                    ids * repeats
                )

            maximum = (
                len(ids)
                - sequence_length
                - 1
            )

            start = self.rng.randint(
                0,
                maximum,
            )

            x = ids[
                start:
                start + sequence_length
            ]

            y = ids[
                start + 1:
                start + 1 + sequence_length
            ]

            xs.append(x)
            ys.append(y)

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
    # Real neural optimization
    # =========================================================

    def _optimizer_step(
        self,
        examples,
    ):
        self.store.model.train()

        x, y = self._batch(
            examples
        )

        with self.store.lock:

            logits = self.store.model(
                x
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
    # Candidate checkpointing
    # =========================================================

    def _snapshot_state(self):

        with self.store.lock:

            model_state = {
                name: value.detach().clone()
                for name, value
                in self.store.model.state_dict().items()
            }

            optimizer_state = copy.deepcopy(
                self.store.optimizer.state_dict()
            )

            step = self.store.step

        return (
            model_state,
            optimizer_state,
            step,
        )

    def _restore_state(
        self,
        snapshot,
    ):
        model_state, optimizer_state, step = (
            snapshot
        )

        with self.store.lock:

            self.store.model.load_state_dict(
                model_state
            )

            self.store.optimizer.load_state_dict(
                optimizer_state
            )

            self.store.step = step

    # =========================================================
    # Replay
    # =========================================================

    def _build_replay(
        self,
        examples,
    ):
        replay = self.memory.replay_examples(
            maximum=256
        )

        if not replay:
            return []

        combined = list(
            examples
        )

        combined.extend(
            replay
        )

        return self._deduplicate(
            combined
        )

    def _record_experience(
        self,
        examples,
        priority=1.0,
    ):

        for example in examples[:16]:

            self.memory.add_replay(
                example,
                priority=priority,
            )

    # =========================================================
    # Focus / curriculum
    # =========================================================

    def _focus_from_history(
        self,
        goal,
    ):
        history = self.memory.learning_history()

        failures = history.get(
            "weaknesses",
            [],
        )

        if failures:

            strongest = sorted(
                failures,
                key=lambda item: float(
                    item.get(
                        "severity",
                        0.0,
                    )
                ),
                reverse=True,
            )

            if strongest:

                target = strongest[0].get(
                    "target",
                    "",
                )

                if target:
                    return (
                        f"{goal} {target}"
                    )

        return goal

    # =========================================================
    # Main training session
    # =========================================================

    def _run(
        self,
        goal,
        minutes,
    ):

        started = time.time()

        deadline = (
            started
            + minutes * 60.0
        )

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

            learning.setdefault(
                "weaknesses",
                [],
            )

            seen = set(
                learning[
                    "seen_sources"
                ]
            )

            # =================================================
            # Research
            # =================================================

            self._update_status(
                phase="research",
                message=(
                    "Selecting a high-value learning target."
                ),
            )

            focus = self._focus_from_history(
                goal
            )

            self._update_status(
                focus=focus,
            )

            documents = research_goal(
                focus,
                limit=12,
                seen_titles=seen,
            )

            for document in documents:

                title = str(
                    document.get(
                        "title",
                        "",
                    )
                ).strip().lower()

                if title:
                    seen.add(
                        title
                    )

            learning[
                "seen_sources"
            ] = list(
                seen
            )[-4000:]

            self.memory.save()

            # =================================================
            # Construct experiences
            # =================================================

            self._update_status(
                phase="experience",
                message=(
                    "Constructing learning experiences from observations."
                ),
            )

            fresh_examples = (
                self._document_examples(
                    goal,
                    documents,
                )
            )

            fresh_examples = self._deduplicate(
                fresh_examples
            )

            replay = self.memory.replay_examples(
                maximum=256
            )

            replay_count = len(
                replay
            )

            examples = list(
                fresh_examples
            )

            examples.extend(
                replay
            )

            examples = self._deduplicate(
                examples
            )

            if len(examples) < 4:

                raise RuntimeError(
                    "Not enough valid learning experience was produced."
                )

            training_examples, validation_examples = (
                self._split_dataset(
                    examples
                )
            )

            self._update_status(
                sources=len(documents),
                examples=len(examples),
                replay_examples=replay_count,
            )

            # =================================================
            # Independent baseline
            # =================================================

            self._update_status(
                phase="baseline",
                message=(
                    "Measuring the current model before updating it."
                ),
            )

            validation_baseline = (
                self.evaluator.loss(
                    validation_examples
                )
            )

            retention_baseline = (
                self.evaluator.retention_score()
            )

            self._update_status(
                baseline_loss=validation_baseline,
                retention_baseline=retention_baseline,
            )

            # =================================================
            # Learning cycles
            # =================================================

            cycle = 0

            while (
                self._remaining(deadline)
                and not self.stop_event.is_set()
            ):

                cycle += 1

                # ---------------------------------------------
                # Save candidate state
                # ---------------------------------------------

                candidate_before = (
                    self._snapshot_state()
                )

                self._update_status(
                    phase="learning",
                    message=(
                        "Updating neural parameters from observed experience."
                    ),
                )

                losses = []

                for _ in range(16):

                    if not self._remaining(
                        deadline
                    ):
                        break

                    if self.stop_event.is_set():
                        break

                    losses.append(
                        self._optimizer_step(
                            training_examples
                        )
                    )

                if not losses:
                    break

                candidate_loss = (
                    self.evaluator.loss(
                        validation_examples
                    )
                )

                candidate_retention = (
                    self.evaluator.retention_score()
                )

                transfer_score = (
                    self.evaluator.transfer_score(
                        goal,
                        documents,
                        validation_examples,
                    )
                )

                validation_improvement = (
                    self.evaluator.improvement(
                        validation_baseline,
                        candidate_loss,
                    )
                )

                retention_delta = (
                    candidate_retention
                    - retention_baseline
                )

                # ---------------------------------------------
                # Candidate acceptance
                #
                # A candidate must improve its current
                # objective without causing unacceptable
                # retention degradation.
                # ---------------------------------------------

                accepted = (
                    candidate_loss is not None
                    and validation_baseline is not None
                    and candidate_loss < validation_baseline
                    and retention_delta >= -0.02
                )

                if accepted:

                    self.store.best_validation = (
                        candidate_loss
                    )

                    self.store.save()

                    self._record_experience(
                        training_examples,
                        priority=max(
                            1.0,
                            1.0 + max(
                                0.0,
                                validation_improvement,
                            ),
                        ),
                    )

                    validation_baseline = (
                        candidate_loss
                    )

                    retention_baseline = (
                        candidate_retention
                    )

                    self.memory.record_outcome(
                        goal=goal,
                        accepted=True,
                        validation_before=(
                            validation_baseline
                            - validation_improvement
                            if validation_improvement
                            is not None
                            else None
                        ),
                        validation_after=(
                            candidate_loss
                        ),
                        retention_before=(
                            retention_baseline
                            - retention_delta
                            if retention_delta
                            is not None
                            else None
                        ),
                        retention_after=(
                            candidate_retention
                        ),
                        transfer=transfer_score,
                    )

                    accepted_count = (
                        self.status.candidate_accepted
                        + 1
                    )

                    self._update_status(
                        candidate_accepted=(
                            accepted_count
                        ),
                    )

                else:

                    # Critical:
                    # restore both neural parameters AND
                    # optimizer state.
                    self._restore_state(
                        candidate_before
                    )

                    self.memory.record_outcome(
                        goal=goal,
                        accepted=False,
                        validation_before=(
                            validation_baseline
                        ),
                        validation_after=(
                            candidate_loss
                        ),
                        retention_before=(
                            retention_baseline
                        ),
                        retention_after=(
                            candidate_retention
                        ),
                        transfer=transfer_score,
                    )

                    rejected_count = (
                        self.status.candidate_rejected
                        + 1
                    )

                    self._update_status(
                        candidate_rejected=(
                            rejected_count
                        ),
                    )

                    # Record observed weakness for curriculum
                    # scheduling. This is infrastructure,
                    # not domain intelligence.
                    severity = 0.0

                    if validation_improvement is not None:
                        severity += max(
                            0.0,
                            -validation_improvement,
                        )

                    if retention_delta < 0:
                        severity += abs(
                            retention_delta
                        )

                    self.memory.record_weakness(
                        target=focus,
                        severity=severity,
                    )

                mean_train_loss = (
                    sum(losses) / len(losses)
                )

                elapsed = (
                    time.time()
                    - started
                )

                self._update_status(
                    cycle=cycle,
                    step=self.store.step,
                    train_loss=mean_train_loss,
                    validation_loss=candidate_loss,
                    improvement=validation_improvement,
                    retention_current=candidate_retention,
                    retention_delta=retention_delta,
                    transfer_score=transfer_score,
                    elapsed=elapsed,
                )

                # =================================================
                # Curriculum remediation
                # =================================================

                if (
                    cycle % 2 == 0
                    and self._remaining(deadline)
                    and not self.stop_event.is_set()
                ):

                    self._update_status(
                        phase="curriculum",
                        message=(
                            "Assessing whether new evidence is more valuable."
                        ),
                    )

                    new_focus = self._focus_from_history(
                        goal
                    )

                    if new_focus != focus:

                        focus = new_focus

                        self._update_status(
                            focus=focus,
                        )

                    extra = research_goal(
                        focus,
                        limit=8,
                        seen_titles=seen,
                    )

                    if extra:

                        for document in extra:

                            title = str(
                                document.get(
                                    "title",
                                    "",
                                )
                            ).strip().lower()

                            if title:
                                seen.add(
                                    title
                                )

                        documents.extend(
                            extra
                        )

                        new_examples = (
                            self._document_examples(
                                goal,
                                extra,
                            )
                        )

                        new_examples = (
                            self._deduplicate(
                                new_examples
                            )
                        )

                        training_examples.extend(
                            new_examples
                        )

                        training_examples = (
                            self._deduplicate(
                                training_examples
                            )
                        )

                        self._update_status(
                            research_rounds=(
                                self.status.research_rounds
                                + 1
                            ),
                            sources=len(documents),
                            examples=len(
                                training_examples
                            ),
                        )

                self.memory.save()

                time.sleep(
                    0.01
                )

            # =================================================
            # Final evaluation
            # =================================================

            self._update_status(
                phase="evaluation",
                message=(
                    "Running final held-out, retention, and transfer evaluation."
                ),
            )

            final_validation = (
                self.evaluator.loss(
                    validation_examples
                )
            )

            final_retention = (
                self.evaluator.retention_score()
            )

            final_transfer = (
                self.evaluator.transfer_score(
                    goal,
                    documents,
                    validation_examples,
                )
            )

            accepted = (
                final_validation is not None
                and validation_baseline is not None
                and final_validation
                <= validation_baseline
                and final_retention
                >= retention_baseline - 0.02
            )

            self._update_status(
                running=False,
                phase="complete",
                validation_loss=final_validation,
                retention_current=final_retention,
                retention_delta=(
                    final_retention
                    - retention_baseline
                ),
                transfer_score=final_transfer,
                elapsed=(
                    time.time()
                    - started
                ),
                message=(
                    "Training complete."
                    if accepted
                    else
                    "Training complete; latest candidate did not beat the acceptance baseline."
                ),
            )

            learning[
                "sessions"
            ].append(
                {
                    "goal": goal,
                    "started": started,
                    "elapsed": time.time() - started,
                    "steps": self.store.step,
                    "validation_loss": final_validation,
                    "retention": final_retention,
                    "transfer": final_transfer,
                    "accepted": accepted,
                    "candidates_accepted": (
                        self.status.candidate_accepted
                    ),
                    "candidates_rejected": (
                        self.status.candidate_rejected
                    ),
                }
            )

            learning[
                "sessions"
            ] = learning[
                "sessions"
            ][-500:]

            learning[
                "seen_sources"
            ] = list(
                seen
            )[-4000:]

            self.memory.save()

        except Exception as exc:

            self._update_status(
                running=False,
                phase="error",
                elapsed=(
                    time.time()
                    - started
                ),
                message=str(exc),
            )

            try:
                self.memory.save()
                self.store.save()
            except Exception:
                pass
