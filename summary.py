from __future__ import annotations


def summarize(
    messages,
):

    if not messages:
        return ""

    parts = []

    for message in messages[-12:]:

        text = " ".join(
            (
                message.get(
                    "content",
                    "",
                )
                or ""
            ).split()
        )

        text = text[:240]

        if not text:
            continue

        speaker = (
            "You"
            if message.get(
                "role"
            ) == "user"
            else "Verdant"
        )

        parts.append(
            f"{speaker}: {text}"
        )

    return " | ".join(
        parts
    )[-2000:]
