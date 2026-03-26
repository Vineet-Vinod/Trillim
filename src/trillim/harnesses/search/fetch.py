"""Search result fetching and bounded content assembly."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from trillim.harnesses.search.provider import (
    MAX_SEARCH_RESULTS,
    SEARCH_CONTENT_CHARS_PER_TOKEN,
    SearchError,
    SearchResult,
)

FETCH_TIMEOUT_SECONDS = 5.0
MAX_FETCH_BYTES = 200_000

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "about",
        "up",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "any",
        "if",
    }
)


def build_search_context(
    query: str,
    results: list[SearchResult],
    *,
    token_budget: int,
    fetcher=None,
) -> str:
    """Fetch result URLs, extract text, and format a bounded prompt payload."""
    if not results:
        raise SearchError("search returned no results")
    fetcher = _fetch_and_extract if fetcher is None else fetcher
    selected = list(results[:MAX_SEARCH_RESULTS])
    entries: list[str] = []
    per_result_budget = max(1, token_budget // max(1, len(selected)))
    for result in selected:
        if not is_safe_url(result.url):
            continue
        try:
            body = fetcher(
                result.url,
                timeout=FETCH_TIMEOUT_SECONDS,
                max_bytes=MAX_FETCH_BYTES,
            )
        except Exception:
            continue
        if not body:
            body = result.snippet
        if not body:
            continue
        title = result.title or result.url
        truncated = truncate_to_token_budget(body, query, token_budget=per_result_budget)
        if not truncated:
            continue
        entries.append(f"[{len(entries) + 1}] {title}\n{truncated}")
    if not entries:
        raise SearchError("search returned no relevant fetchable results")
    return _truncate_at_sentence(
        "\n\n".join(entries),
        token_budget * SEARCH_CONTENT_CHARS_PER_TOKEN,
    )


def is_safe_url(url: str) -> bool:
    """Apply minimal URL safety checks before fetching provider results."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = parsed.hostname
    if hostname is None:
        return False
    normalized = hostname.lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def resolves_to_safe_addresses(url: str) -> bool:
    """Resolve a URL hostname and reject any local or private target addresses."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return False
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not infos:
        return False
    for _family, _socktype, _proto, _canonname, sockaddr in infos:
        host = sockaddr[0]
        try:
            address = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return False
        if any(
            (
                address.is_private,
                address.is_loopback,
                address.is_link_local,
                address.is_multicast,
                address.is_reserved,
                address.is_unspecified,
            )
        ):
            return False
    return True


def truncate_to_token_budget(text: str, query: str, *, token_budget: int) -> str:
    """Select the most relevant paragraphs within a rough token budget."""
    char_budget = max(1, token_budget) * SEARCH_CONTENT_CHARS_PER_TOKEN
    paragraphs = [
        stripped
        for block in text.split("\n\n")
        for stripped in (line.strip() for line in block.split("\n"))
        if stripped
    ]
    if not paragraphs:
        return ""
    if len(paragraphs) == 1:
        return _truncate_at_sentence(paragraphs[0], char_budget)
    query_words = {
        word
        for word in re.findall(r"\w+", query.lower())
        if word not in _STOPWORDS
    }
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        paragraph_words = set(re.findall(r"\w+", paragraph.lower()))
        scored.append((-len(query_words & paragraph_words), index, paragraph))
    scored.sort()
    remaining = char_budget
    selected: list[tuple[int, str]] = []
    for _score, index, paragraph in scored:
        if len(paragraph) <= remaining:
            selected.append((index, paragraph))
            remaining -= len(paragraph) + 2
            continue
        if not selected:
            selected.append((index, _truncate_at_sentence(paragraph, char_budget)))
        break
    selected.sort()
    return "\n\n".join(paragraph for _index, paragraph in selected)


def _truncate_at_sentence(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    snippet = text[:budget]
    for index in range(len(snippet) - 1, -1, -1):
        if snippet[index] in ".!?":
            return snippet[: index + 1]
    return snippet


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects to unsafe URLs before the request is followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected_url = urllib.parse.urljoin(req.full_url, newurl)
        if not is_safe_url(redirected_url) or not resolves_to_safe_addresses(
            redirected_url
        ):
            raise urllib.error.URLError("unsafe redirect target")
        return super().redirect_request(req, fp, code, msg, headers, redirected_url)


def _open_request(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(_SafeRedirectHandler)
    return opener.open(request, timeout=timeout)


def _fetch_and_extract(url: str, *, timeout: float, max_bytes: int) -> str | None:
    """Fetch a result URL and extract its main text."""
    try:
        if not is_safe_url(url) or not resolves_to_safe_addresses(url):
            return None
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Trillim/1.0)"},
        )
        with _open_request(request, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
        html = payload[:max_bytes].decode("utf-8", errors="ignore")
        import trafilatura

        return trafilatura.extract(
            html,
            no_fallback=True,
            include_tables=False,
            include_comments=False,
        )
    except Exception:
        return None
