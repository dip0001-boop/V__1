from __future__ import annotations

import math
import random
import re

import torch
import torch.nn.functional as F


class Evaluator:

    def __init__(
        self,
        store,
    ):
        self.store = store

    def loss(
        self,
        examples,
        maximum=48,
    ):

        if not examples:
            return None

        selected = examples[
            :maximum
        ]

        self.store.model.eval()

        values = []

        with torch.no_grad():

            for example in selected:

                ids = (
                    self.store.tokenizer.encode(
                        example
                    )
                )

                if len(ids) < 4:
                    continue

                ids = ids[
                    :self.store.model.cfg.context
                ]

                x = torch.tensor(
                    [
                        ids[:-1]
                    ],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    [
                        ids[1:]
                    ],
                    dtype=torch.long,
                )

                logits = (
                    self.store.model(x)
                )

                value = (
                    F.cross_entropy(
                        logits.reshape(
                            -1,
                            logits.shape[-1],
                        ),
                        y.reshape(-1),
                    )
                )

                values.append(
                    float(value)
                )

        if not values:
            return None

        return sum(values) / len(
            values
        )

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

        return baseline - current

    def score(
        self,
        baseline,
        current,
    ):

        change = self.improvement(
            baseline,
            current,
        )

        # This number is deliberately not called
        # "understanding". It only measures held-out
        # predictive improvement.
        result = 0.5 + (
            change / 4.0
        )

        return max(
            0.0,
            min(
                1.0,
                result,
            ),
        )

    @staticmethod
    def generate_unseen_prompts(
        goal,
        concepts,
        count=7,
    ):

        # These prompts are created from held-out concepts,
        # not copied from training examples.
        rng = random.Random(
            hash(goal) & 0xffffffff
        )

        choices = list(
            concepts
        )

        rng.shuffle(
            choices
        )

        prompts = []

        for concept in choices:

            prompts.append(
                (
                    "Explain this idea in your own words: "
                    + concept
                )
            )

            if len(prompts) >= count:
                break

        while len(prompts) < count:

            prompts.append(
                (
                    "What is an important idea "
                    f"about {goal}?"
                )
            )

        return prompts[
            :count
        ]
