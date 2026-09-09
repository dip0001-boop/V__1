from __future__ import annotations

import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Verdant-1.0-learning-research/1.0"
)


def clean_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def wiki_search(
    query: str,
    limit: int = 12,
):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": limit,
        },
        headers={
            "User-Agent": USER_AGENT
        },
        timeout=20,
    )

    response.raise_for_status()

    results = []

    for item in (
        response.json()
        .get("query", {})
        .get("search", [])
    ):

        title = item["title"]

        results.append(
            {
                "title": title,
                "snippet": BeautifulSoup(
                    item.get(
                        "snippet",
                        "",
                    ),
                    "html.parser",
                ).get_text(" "),
                "url": (
                    "https://en.wikipedia.org/wiki/"
                    + quote(
                        title.replace(
                            " ",
                            "_",
                        )
                    )
                ),
            }
        )

    return results


def fetch_page(
    url: str,
    max_chars: int = 24000,
):
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT
        },
        timeout=25,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    return clean_text(
        soup.get_text(" ")
    )[:max_chars]


def relevance_score(
    goal: str,
    title: str,
    snippet: str,
):
    terms = set(
        re.findall(
            r"[a-z0-9]{3,}",
            goal.lower(),
        )
    )

    text = (
        title
        + " "
        + snippet
    ).lower()

    if not terms:
        return 0

    return sum(
        1
        for term in terms
        if term in text
    )


def research_goal(
    goal: str,
    limit: int = 12,
    seen_titles: set[str] | None = None,
):
    seen_titles = seen_titles or set()

    # Search several formulations so one poor result set
    # doesn't define the entire learning session.
    queries = [
        goal,
        f"{goal} fundamentals",
        f"{goal} tutorial concepts",
        f"{goal} examples",
    ]

    all_hits = {}

    for query in queries:

        try:
            hits = wiki_search(
                query,
                max(12, limit),
            )
        except Exception:
            continue

        for hit in hits:

            key = hit[
                "title"
            ].strip().lower()

            if key in seen_titles:
                continue

            score = relevance_score(
                goal,
                hit["title"],
                hit["snippet"],
            )

            previous = all_hits.get(
                key
            )

            if (
                previous is None
                or score > previous[0]
            ):
                all_hits[key] = (
                    score,
                    hit,
                )

    ranked = sorted(
        all_hits.values(),
        key=lambda item: (
            -item[0],
            item[1]["title"],
        ),
    )

    documents = []

    for score, hit in ranked:

        if score <= 0:
            continue

        try:
            text = fetch_page(
                hit["url"]
            )
        except Exception:
            continue

        if len(text) < 700:
            continue

        documents.append(
            {
                **hit,
                "text": text,
                "relevance": score,
            }
        )

        if len(documents) >= limit:
            break

    return documents
