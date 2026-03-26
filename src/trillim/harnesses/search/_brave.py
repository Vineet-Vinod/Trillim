"""Brave search provider."""

from __future__ import annotations

import gzip
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from trillim.harnesses.search.provider import (
    SearchAuthenticationError,
    SearchError,
    SearchResult,
    coerce_search_result,
)


class BraveSearchProvider:
    """Brave provider via the official LLM context endpoint."""

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"

    def __init__(self, *, token_budget: int) -> None:
        self._token_budget = token_budget

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return normalized Brave results."""
        api_key = os.environ.get("SEARCH_API_KEY")
        if not api_key:
            raise SearchAuthenticationError(
                "Brave search requires SEARCH_API_KEY in the environment"
            )
        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": max_results,
                "maximum_number_of_urls": max_results,
                "maximum_number_of_tokens": max(1, self._token_budget),
                "context_threshold_mode": "balanced",
            }
        )
        request = urllib.request.Request(
            f"{self._ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "Trillim/1.0",
                "X-Subscription-Token": api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                payload = response.read()
                content_encoding = str(response.headers.get("Content-Encoding", "")).lower()
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise SearchAuthenticationError("Brave search failed: wrong SEARCH_API_KEY") from exc
            raise SearchError(f"Brave search failed with HTTP {exc.code}") from exc
        except Exception as exc:
            raise SearchError(f"Brave search failed: {exc}") from exc
        if "gzip" in content_encoding:
            try:
                payload = gzip.decompress(payload)
            except Exception as exc:
                raise SearchError(f"Brave search failed: {exc}") from exc
        try:
            decoded = json.loads(payload.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            raise SearchError("Brave search returned invalid JSON") from exc

        grounding = decoded.get("grounding", {})
        sources = decoded.get("sources", {})
        raw_results: list[dict] = []
        if isinstance(grounding, dict):
            generic = grounding.get("generic", [])
            if isinstance(generic, list):
                raw_results.extend(item for item in generic if isinstance(item, dict))
            poi = grounding.get("poi")
            if isinstance(poi, dict):
                raw_results.append(poi)
            mapped = grounding.get("map", [])
            if isinstance(mapped, list):
                raw_results.extend(item for item in mapped if isinstance(item, dict))
        results: list[SearchResult] = []
        for item in raw_results:
            url = str(item.get("url", "")).strip()
            source = sources.get(url, {}) if isinstance(sources, dict) else {}
            if not isinstance(source, dict):
                source = {}
            title = str(item.get("title", "") or source.get("title", "")).strip()
            snippets = item.get("snippets", [])
            snippet = ""
            if isinstance(snippets, list):
                snippet = "\n".join(str(value).strip() for value in snippets if value).strip()
            result = coerce_search_result(
                title=title or url,
                url=url,
                snippet=snippet,
            )
            if result is not None:
                results.append(result)
        if not results:
            raise SearchError("Brave search returned no usable results")
        return results
