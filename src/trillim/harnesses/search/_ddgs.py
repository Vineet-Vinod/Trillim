"""DuckDuckGo search provider."""

from __future__ import annotations

import os

from trillim.harnesses.search.provider import SearchError, SearchResult, coerce_search_result


class DDGSSearchProvider:
    """DuckDuckGo provider via ``ddgs``."""

    name = "ddgs"

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return normalized DDGS results."""
        from ddgs import DDGS

        try:
            stderr_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
            try:
                raw_results = DDGS().text(query, max_results=max_results)
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)
                os.close(devnull_fd)
            if not raw_results:
                raise SearchError("DDGS search returned no results")
            results: list[SearchResult] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                result = coerce_search_result(
                    title=str(item.get("title", "")),
                    url=str(item.get("href", "")),
                    snippet=str(item.get("body", "")),
                )
                if result is not None:
                    results.append(result)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(f"DDGS search failed: {exc}") from exc
        if not results:
            raise SearchError("DDGS search returned no usable results")
        return results
