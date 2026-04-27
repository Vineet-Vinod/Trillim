from __future__ import annotations

import asyncio
import gzip
import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from trillim.harnesses.search._brave import BraveSearchProvider
from trillim.harnesses.search._ddgs import DDGSSearchProvider
from trillim.harnesses.search.client import SearchClient
from trillim.harnesses.search.fetch import (
    build_search_context,
    is_safe_url,
    resolves_to_safe_addresses,
    truncate_to_token_budget,
)
from trillim.harnesses.search.metrics import SearchMetrics
from trillim.harnesses.search.provider import (
    MAX_MESSAGE_CHARS,
    SearchAuthenticationError,
    SearchError,
    SearchResult,
    coerce_search_result,
    extract_search_query,
    normalize_provider_name,
    resolve_search_token_budget,
    validate_harness_name,
    validate_search_query,
)


class SearchProviderTests(unittest.TestCase):
    def test_search_query_extraction_and_validation(self):
        self.assertEqual(
            extract_search_query("before <search> latest   docs </search> after"),
            "latest docs",
        )
        self.assertIsNone(extract_search_query("no tool call"))
        with self.assertRaisesRegex(SearchError, "must not be empty"):
            validate_search_query("   ")
        self.assertEqual(len(validate_search_query("x" * (MAX_MESSAGE_CHARS + 5))), MAX_MESSAGE_CHARS)

    def test_provider_and_harness_names_are_normalized(self):
        self.assertEqual(normalize_provider_name(" DuckDuckGo "), "ddgs")
        self.assertEqual(normalize_provider_name("brave_search"), "brave")
        self.assertEqual(validate_harness_name(" Search "), "search")
        self.assertEqual(resolve_search_token_budget(100, max_context_tokens=80), 20)

        with self.assertRaisesRegex(ValueError, "Unknown search provider"):
            normalize_provider_name("unknown")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resolve_search_token_budget(0, max_context_tokens=80)

    def test_coerce_search_result_drops_missing_url_and_normalizes_text(self):
        self.assertIsNone(coerce_search_result(title="Title", url=" "))
        result = coerce_search_result(
            title="  A   Title  ",
            url=" https://example.com ",
            snippet="  useful   text ",
        )

        self.assertEqual(
            result,
            SearchResult(
                title="A Title",
                url="https://example.com",
                snippet="useful text",
            ),
        )

    def test_brave_provider_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(SearchAuthenticationError, "SEARCH_API_KEY"):
                BraveSearchProvider(token_budget=10).search("query", max_results=1)

    def test_brave_provider_parses_gzipped_llm_context_response(self):
        payload = {
            "grounding": {
                "generic": [
                    {
                        "url": "https://example.com/a",
                        "title": " A result ",
                        "snippets": [" first ", " second "],
                    }
                ],
                "poi": {"url": "https://example.com/place", "title": "Place"},
            },
            "sources": {"https://example.com/a": {"title": "Fallback"}},
        }

        class Response:
            headers = {"Content-Encoding": "gzip"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return gzip.compress(json.dumps(payload).encode("utf-8"))

        with patch.dict("os.environ", {"SEARCH_API_KEY": "secret"}), patch(
            "urllib.request.urlopen",
            return_value=Response(),
        ):
            results = BraveSearchProvider(token_budget=12).search("query", max_results=2)

        self.assertEqual(results[0].title, "A result")
        self.assertEqual(results[0].snippet, "first second")
        self.assertEqual(results[1].url, "https://example.com/place")

    def test_brave_provider_maps_http_auth_and_invalid_json_errors(self):
        with patch.dict("os.environ", {"SEARCH_API_KEY": "secret"}), patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://api.example",
                403,
                "forbidden",
                {},
                None,
            ),
        ):
            with self.assertRaises(SearchAuthenticationError):
                BraveSearchProvider(token_budget=12).search("query", max_results=2)

        class BadResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"{"

        with patch.dict("os.environ", {"SEARCH_API_KEY": "secret"}), patch(
            "urllib.request.urlopen",
            return_value=BadResponse(),
        ):
            with self.assertRaisesRegex(SearchError, "invalid JSON"):
                BraveSearchProvider(token_budget=12).search("query", max_results=2)

    def test_ddgs_provider_normalizes_results_and_errors_when_empty(self):
        class DDGS:
            def text(self, query, *, max_results):
                return [
                    {"title": " One ", "href": "https://example.com", "body": " body "},
                    {"title": "No URL", "href": "", "body": "drop"},
                ]

        with patch("ddgs.DDGS", return_value=DDGS()):
            results = DDGSSearchProvider().search("query", max_results=3)

        self.assertEqual(
            results,
            [SearchResult(title="One", url="https://example.com", snippet="body")],
        )

        class EmptyDDGS:
            def text(self, query, *, max_results):
                return []

        with patch("ddgs.DDGS", return_value=EmptyDDGS()):
            with self.assertRaisesRegex(SearchError, "no results"):
                DDGSSearchProvider().search("query", max_results=3)


class SearchFetchTests(unittest.TestCase):
    def test_url_safety_rejects_local_and_non_http_targets(self):
        self.assertFalse(is_safe_url("file:///tmp/item"))
        self.assertFalse(is_safe_url("http://localhost/page"))
        self.assertFalse(is_safe_url("http://127.0.0.1/page"))
        self.assertTrue(is_safe_url("https://example.com/page"))

    def test_resolves_to_safe_addresses_rejects_private_resolution(self):
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
        with patch("socket.getaddrinfo", return_value=[private]):
            self.assertFalse(resolves_to_safe_addresses("https://example.com"))
        with patch("socket.getaddrinfo", return_value=[public]):
            self.assertTrue(resolves_to_safe_addresses("https://example.com"))
        with patch("socket.getaddrinfo", side_effect=OSError):
            self.assertFalse(resolves_to_safe_addresses("https://example.com"))

    def test_truncate_to_token_budget_prefers_query_relevant_paragraphs(self):
        text = "alpha only paragraph.\n\npython search result is relevant.\n\nomega only."

        truncated = truncate_to_token_budget(text, "python result", token_budget=10)

        self.assertIn("python search result", truncated)
        self.assertNotIn("alpha only", truncated)

    def test_build_search_context_formats_fetchable_safe_results(self):
        results = [
            SearchResult(title="Bad", url="http://127.0.0.1/private", snippet="hidden"),
            SearchResult(title="Good", url="https://example.com", snippet="fallback"),
        ]

        context = build_search_context(
            "python",
            results,
            token_budget=20,
            fetcher=lambda *args, **kwargs: "python content. extra sentence.",
        )

        self.assertIn("[1] Good", context)
        self.assertIn("python content", context)

    def test_build_search_context_raises_when_nothing_is_usable(self):
        with self.assertRaisesRegex(SearchError, "no results"):
            build_search_context("query", [], token_budget=10)
        with self.assertRaisesRegex(SearchError, "fetchable"):
            build_search_context(
                "query",
                [SearchResult(title="Local", url="http://localhost")],
                token_budget=10,
                fetcher=lambda *args, **kwargs: "ignored",
            )


class SearchClientTests(unittest.TestCase):
    def test_search_client_builds_real_selected_providers(self):
        self.assertIsInstance(
            SearchClient(provider_name="ddgs", token_budget=20)._provider,
            DDGSSearchProvider,
        )
        self.assertIsInstance(
            SearchClient(provider_name="brave", token_budget=20)._provider,
            BraveSearchProvider,
        )

    def test_search_client_propagates_real_provider_errors(self):
        client = SearchClient(provider_name="brave", token_budget=20)

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SearchAuthenticationError):
                asyncio.run(client.search("  python   testing "))

    def test_metrics_record_latest_generation(self):
        metrics = SearchMetrics()
        metrics.record_generation(prompt_tokens=3, completion_tokens=4)
        metrics.record_generation(prompt_tokens=1, completion_tokens=2)

        self.assertEqual(metrics.prompt_tokens, 1)
        self.assertEqual(metrics.completion_tokens, 2)
