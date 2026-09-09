from __future__ import annotations


def summarize(messages):
    if not messages:
        return ""

    parts = []

    for message in messages[-12:]:
        content = " ".join(
            (
                message.get(
                    "content",
                    "",
                )
                or ""
            ).split()
        )[:240]

        if not content:
            continue

        speaker = (
            "You"
            if message.get("role")
            == "user"
            else "Verdant"
        )

        parts.append(
            f"{speaker}: {content}"
        )

    return " | ".join(
        parts
    )[-2000:]
