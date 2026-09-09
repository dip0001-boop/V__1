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

        self.lock = (
            threading.RLock()
        )

        self.data = {
            "chats": {},
            "learning": {
                "sessions": [],
                "seen_sources": [],
                "replay": [],
            },
        }

        self._load()

    def _load(self):

        if not self.path.exists():
            return

        try:

            loaded = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                loaded,
                dict,
            ):
                self.data.update(
                    loaded
                )

        except Exception:
            pass

    def save(self):

        with self.lock:

            temp = (
                self.path.with_suffix(
                    ".tmp"
                )
            )

            temp.write_text(
                json.dumps(
                    self.data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temp.replace(
                self.path
            )

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
        )["messages"].append(
            {
                "role": role,
                "content": content,
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
        )["summary"] = summary

        self.save()

    def add_replay(
        self,
        example,
        priority=1.0,
    ):

        learning = self.data[
            "learning"
        ]

        learning.setdefault(
            "replay",
            []
        )

        learning[
            "replay"
        ].append(
            {
                "example":
                    example,
                "priority":
                    float(priority),
            }
        )

        # Keep persistent replay bounded.
        learning[
            "replay"
        ] = learning[
            "replay"
        ][-2000:]

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
            key=lambda item:
                item.get(
                    "priority",
                    1.0,
                ),
            reverse=True,
        )

        return [
            item["example"]
            for item in ordered[
                :maximum
            ]
        ]
