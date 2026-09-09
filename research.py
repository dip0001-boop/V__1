from __future__ import annotations

import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Verdant-1.0/learning-client"
)


def clean_text(
    text: str,
) -> str:

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

                "snippet": (
                    BeautifulSoup(
                        item.get(
                            "snippet",
                            "",
                        ),
                        "html.parser",
                    ).get_text(" ")
                ),

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


def relevance(
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

    return sum(
        1
        for term in terms
        if term in text
    )


def research_goal(
    goal: str,
    limit: int = 12,
    seen_titles=None,
):

    seen_titles = (
        seen_titles
        if seen_titles is not None
        else set()
    )

    queries = [
        goal,
        f"{goal} fundamentals",
        f"{goal} concepts",
        f"{goal} examples",
        f"{goal} tutorial",
    ]

    candidates = {}

    for query in queries:

        try:
            hits = wiki_search(
                query,
                limit=15,
            )

        except Exception:
            continue

        for hit in hits:

            key = (
                hit["title"]
                .strip()
                .lower()
            )

            if key in seen_titles:
                continue

            score = relevance(
                goal,
                hit["title"],
                hit["snippet"],
            )

            previous = candidates.get(
                key
            )

            if (
                previous is None
                or score > previous[0]
            ):
                candidates[key] = (
                    score,
                    hit,
                )

    ranked = sorted(
        candidates.values(),
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
