from __future__ import annotations

import re
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Verdant-1.0-research/1.0"
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
    limit: int = 15,
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
            "User-Agent":
                USER_AGENT
        },
        timeout=20,
    )

    response.raise_for_status()

    output = []

    for item in (
        response.json()
        .get("query", {})
        .get("search", [])
    ):

        title = item[
            "title"
        ]

        output.append(
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

    return output


def fetch_page(
    url: str,
    limit: int = 30000,
):

    response = requests.get(
        url,
        headers={
            "User-Agent":
                USER_AGENT
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
    )[:limit]


def _terms(
    query: str,
):

    return set(
        re.findall(
            r"[a-z0-9]{3,}",
            query.lower(),
        )
    )


def relevance(
    goal: str,
    title: str,
    snippet: str,
):

    terms = _terms(
        goal
    )

    text = (
        title
        + " "
        + snippet
    ).lower()

    score = 0

    for term in terms:

        if term in text:
            score += (
                1
                + min(
                    3,
                    len(term) / 5
                )
            )

    return score


def research_goal(
    goal: str,
    limit: int = 12,
    seen_titles=None,
):

    seen_titles = set(
        seen_titles or []
    )

    queries = [
        goal,
        f"{goal} fundamentals",
        f"{goal} concepts",
        f"{goal} examples",
        f"{goal} explanation",
        f"{goal} common mistakes",
    ]

    candidates = {}

    for query in queries:

        try:
            hits = wiki_search(
                query
            )
        except Exception:
            continue

        for hit in hits:

            title_key = (
                hit["title"]
                .strip()
                .lower()
            )

            if title_key in seen_titles:
                continue

            score = relevance(
                goal,
                hit["title"],
                hit["snippet"],
            )

            old = candidates.get(
                title_key
            )

            if (
                old is None
                or score > old[0]
            ):
                candidates[
                    title_key
                ] = (
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
                "relevance":
                    score,
            }
        )

        if (
            len(documents)
            >= limit
        ):
            break

    return documents
