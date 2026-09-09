from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
]


class LearnedTokenizer:
    """
    Compact subword tokenizer with byte fallback.

    Vocabulary is learned from corpus statistics.
    Unknown text is represented through bytes rather than <unk>.
    """

    def __init__(self, path: str = "verdant_tokenizer.json"):
        self.path = Path(path)

        self.tokens: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        self._load()

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _rebuild_maps(self):
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.tokens)
        }

        self.id_to_token = {
            index: token
            for index, token in enumerate(self.tokens)
        }

    def _load(self):
        if not self.path.exists():
            return

        try:
            data = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

            self.tokens = list(
                data.get(
                    "tokens",
                    []
                )
            )

            self._rebuild_maps()

        except Exception:
            self.tokens = []
            self._rebuild_maps()

    def save(self):
        self.path.write_text(
            json.dumps(
                {
                    "tokens": self.tokens
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def build(
        self,
        text: str,
        max_vocab: int = 8192,
    ):
        """
        Builds a compact vocabulary from words and common subwords.

        This is preprocessing, not intelligence.
        """

        word_counts = Counter(
            re.findall(
                r"\w+|[^\w\s]",
                text,
                flags=re.UNICODE,
            )
        )

        pieces = set(
            SPECIAL_TOKENS
        )

        # Keep frequent whole tokens.
        for token, _ in word_counts.most_common(
            max_vocab // 2
        ):
            pieces.add(token)

            lower = token.lower()

            if len(lower) >= 4:
                for i in range(
                    2,
                    min(6, len(lower)),
                ):
                    pieces.add(
                        lower[:i]
                    )

        # Byte fallback.
        for value in range(256):
            pieces.add(
                f"<byte:{value}>"
            )

        tokens = list(pieces)

        tokens = tokens[
            :max_vocab
        ]

        # Ensure all required fallback bytes survive.
        required = {
            *SPECIAL_TOKENS,
            *[
                f"<byte:{i}>"
                for i in range(256)
            ],
        }

        existing = set(tokens)

        for token in required:
            if token not in existing:
                tokens.append(token)

        self.tokens = tokens
        self._rebuild_maps()
        self.save()

    def _byte_ids(
        self,
        text: str,
    ):
        result = []

        for value in text.encode(
            "utf-8",
            errors="replace",
        ):
            token = f"<byte:{value}>"

            result.append(
                self.token_to_id[token]
            )

        return result

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ):
        if not self.tokens:
            raise RuntimeError(
                "Tokenizer vocabulary has not been built."
            )

        ids = []

        if add_bos:
            ids.append(
                self.token_to_id["<bos>"]
            )

        # Prefer learned lexical pieces.
        pieces = re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE,
        )

        cursor = 0

        for piece in pieces:

            location = text.find(
                piece,
                cursor,
            )

            if location > cursor:
                whitespace = text[
                    cursor:location
                ]

                for char in whitespace:
                    if char.isspace():
                        ids.extend(
                            self._byte_ids(char)
                        )

            cursor = location + len(piece)

            token_id = self.token_to_id.get(
                piece
            )

            if token_id is not None:
                ids.append(token_id)
            else:
                ids.extend(
                    self._byte_ids(piece)
                )

        if cursor < len(text):
            ids.extend(
                self._byte_ids(
                    text[cursor:]
                )
            )

        if add_eos:
            ids.append(
                self.token_to_id["<eos>"]
            )

        if max_len is not None:
            ids = ids[-max_len:]

        return ids

    def decode(
        self,
        ids,
    ) -> str:

        output = bytearray()
        text_parts = []

        for index in ids:

            token = self.id_to_token.get(
                int(index),
                "",
            )

            if token.startswith(
                "<byte:"
            ):
                try:
                    value = int(
                        token[6:-1]
                    )
                    output.append(
                        value
                    )
                except Exception:
                    pass

            elif token in SPECIAL_TOKENS:
                continue

            else:
                if output:
                    text_parts.append(
                        output.decode(
                            "utf-8",
                            errors="ignore",
                        )
                    )
                    output.clear()

                text_parts.append(
                    token
                )

        if output:
            text_parts.append(
                output.decode(
                    "utf-8",
                    errors="ignore",
                )
            )

        result = "".join(
            text_parts
        )

        # Mild decoding normalization.
        result = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            result,
        )

        return result
