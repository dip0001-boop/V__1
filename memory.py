from __future__ import annotations

import json
import threading
from pathlib import Path


class MemoryStore:

    def __init__(
        self,
        path="verdant_memory.json",
    ):
        self.path = Path(
            path
        )

        self.lock = threading.RLock()

        self.data = {
            "chats": {},

            "learning": {
                "sessions": [],
                "seen_sources": [],
                "replay": [],
                "weaknesses": [],
                "outcomes": [],
            },
        }

        self._load()

    # =========================================================
    # Persistence
    # =========================================================

    def _load(self):

        if not self.path.exists():
            return

        try:

            loaded = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(
                loaded,
                dict,
            ):
                return

            chats = loaded.get(
                "chats"
            )

            if isinstance(
                chats,
                dict,
            ):
                self.data[
                    "chats"
                ] = chats

            learning = loaded.get(
                "learning"
            )

            if isinstance(
                learning,
                dict,
            ):

                self.data[
                    "learning"
                ].update(
                    learning
                )

        except Exception:
            # Corrupt memory must not prevent boot.
            pass

    def save(self):

        with self.lock:

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = (
                self.path.with_suffix(
                    ".tmp"
                )
            )

            temporary.write_text(
                json.dumps(
                    self.data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary.replace(
                self.path
            )

    # =========================================================
    # Chat memory
    # =========================================================

    def chat(
        self,
        chat_id,
    ):

        return self.data[
            "chats"
        ].setdefault(
            chat_id,
            {
                "messages": [],
                "summary": "",
            },
        )

    def add_message(
        self,
        chat_id,
        role,
        content,
    ):

        self.chat(
            chat_id
        )[
            "messages"
        ].append(
            {
                "role": str(
                    role
                ),
                "content": str(
                    content
                ),
            }
        )

        self.save()

    def set_summary(
        self,
        chat_id,
        summary,
    ):

        self.chat(
            chat_id
        )[
            "summary"
        ] = str(
            summary
        )

        self.save()

    # =========================================================
    # Replay
    # =========================================================

    def add_replay(
        self,
        example,
        priority=1.0,
    ):
        learning = self.data[
            "learning"
        ]

        replay = learning.setdefault(
            "replay",
            [],
        )

        replay.append(
            {
                "example": str(
                    example
                ),
                "priority": float(
                    priority
                ),
            }
        )

        # Bound persistent replay.
        #
        # Keep the highest-value examples,
        # rather than blindly keeping the latest.
        replay.sort(
            key=lambda item: float(
                item.get(
                    "priority",
                    1.0,
                )
            ),
            reverse=True,
        )

        learning[
            "replay"
        ] = replay[:4000]

    def replay_examples(
        self,
        maximum=256,
    ):

        replay = self.data[
            "learning"
        ].get(
            "replay",
            [],
        )

        if not replay:
            return []

        ordered = sorted(
            replay,
            key=lambda item: float(
                item.get(
                    "priority",
                    1.0,
                )
            ),
            reverse=True,
        )

        result = []

        for item in ordered:

            example = item.get(
                "example"
            )

            if not example:
                continue

            result.append(
                str(
                    example
                )
            )

            if len(result) >= maximum:
                break

        return result

    # =========================================================
    # Learning history
    # =========================================================

    def learning_history(
        self,
    ):
        return self.data[
            "learning"
        ]

    def record_outcome(
        self,
        goal,
        accepted,
        validation_before,
        validation_after,
        retention_before,
        retention_after,
        transfer,
    ):

        outcomes = self.data[
            "learning"
        ].setdefault(
            "outcomes",
            [],
        )

        outcomes.append(
            {
                "goal": str(
                    goal
                ),
                "accepted": bool(
                    accepted
                ),
                "validation_before": (
                    validation_before
                ),
                "validation_after": (
                    validation_after
                ),
                "retention_before": (
                    retention_before
                ),
                "retention_after": (
                    retention_after
                ),
                "transfer": transfer,
            }
        )

        self.data[
            "learning"
        ][
            "outcomes"
        ] = outcomes[-2000:]

    def record_weakness(
        self,
        target,
        severity,
    ):

        weaknesses = self.data[
            "learning"
        ].setdefault(
            "weaknesses",
            [],
        )

        weaknesses.append(
            {
                "target": str(
                    target
                ),
                "severity": float(
                    severity
                ),
            }
        )

        weaknesses.sort(
            key=lambda item: float(
                item.get(
                    "severity",
                    0.0,
                )
            ),
            reverse=True,
        )

        self.data[
            "learning"
        ][
            "weaknesses"
        ] = weaknesses[:1000]
