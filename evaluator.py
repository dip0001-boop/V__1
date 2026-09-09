from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


class Evaluator:

    def __init__(
        self,
        store,
        memory=None,
    ):
        self.store = store
        self.memory = memory

    # =========================================================
    # Language objective
    # =========================================================

    def loss(
        self,
        examples,
        maximum=64,
    ):
        if not examples:
            return None

        selected = list(
            examples[:maximum]
        )

        self.store.model.eval()

        values = []

        with torch.no_grad():

            for example in selected:

                ids = self.store.tokenizer.encode(
                    example
                )

                if len(ids) < 4:
                    continue

                ids = ids[
                    :self.store.model.cfg.context
                ]

                x = torch.tensor(
                    [ids[:-1]],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    [ids[1:]],
                    dtype=torch.long,
                )

                logits = self.store.model(
                    x
                )

                value = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.shape[-1],
                    ),
                    y.reshape(-1),
                )

                if math.isfinite(
                    float(value)
                ):
                    values.append(
                        float(value)
                    )

        if not values:
            return None

        return sum(values) / len(
            values
        )

    # =========================================================
    # Improvement
    # =========================================================

    def improvement(
        self,
        baseline,
        current,
    ):
        if (
            baseline is None
            or current is None
        ):
            return 0.0

        return (
            baseline
            - current
        )

    # =========================================================
    # Persistent retention
    # =========================================================

    def retention_score(
        self,
        maximum=64,
    ):
        if self.memory is None:
            return 0.0

        replay = (
            self.memory.replay_examples(
                maximum=maximum
            )
        )

        if not replay:
            return 0.0

        value = self.loss(
            replay,
            maximum=maximum,
        )

        if value is None:
            return 0.0

        # Retention is tracked as predictive quality
        # rather than a fabricated percentage.
        #
        # Lower loss => higher retention score.
        #
        # exp(-loss) is bounded in (0, 1] and does not
        # claim semantic mastery.
        return math.exp(
            -max(
                0.0,
                value,
            )
        )

    # =========================================================
    # Independent transfer-oriented evaluation
    # =========================================================

    def _concepts_from_documents(
        self,
        documents,
    ):
        concepts = []

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

            if not text:
                continue

            sentences = (
                text.replace(
                    "\n",
                    " ",
                ).split(".")
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if len(sentence) >= 80:

                    concepts.append(
                        (
                            title,
                            sentence,
                        )
                    )

        return concepts

    def transfer_score(
        self,
        goal,
        documents,
        validation_examples,
    ):
        """
        Measures whether the current model can predict
        transformed observations that were not directly
        optimized in the current update.

        This deliberately does NOT ask the model to grade
        itself.
        """

        concepts = (
            self._concepts_from_documents(
                documents
            )
        )

        if not concepts:
            return 0.0

        rng = random.Random(
            hash(goal) & 0xffffffff
        )

        rng.shuffle(
            concepts
        )

        selected = concepts[:8]

        probes = []

        for title, sentence in selected:

            probes.append(
                (
                    f"Goal: {goal}\n"
                    f"Topic: {title}\n"
                    f"Observation: {sentence}\n"
                    f"Prediction:"
                )
            )

        # The probe loss is compared against the model's
        # own fixed current predictive objective, not a
        # generated self-score.
        value = self.loss(
            probes,
            maximum=len(probes),
        )

        if value is None:
            return 0.0

        return math.exp(
            -max(
                0.0,
                value,
            )
        )
