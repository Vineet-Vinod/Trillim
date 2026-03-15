# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for tokenizer-aware search truncation helpers."""

import asyncio
import gzip
import json
from types import ModuleType
import unittest
import urllib.error
from unittest.mock import patch

from trillim.harnesses._search_utils import (
    BraveSearchProvider,
    DDGSSearchProvider,
    SearchClient,
    SearchError,
    SearchProvider,
    SearchResult,
    _token_budget_to_char_budget,
    _truncate_at_sentence,
    _truncate_to_exact_token_budget,
    extract_search_query,
    get_search_provider,
    truncate_to_token_budget,
)


def _word_token_count(text: str) -> int:
    return len(text.split())


class _ProviderStub(SearchProvider):
    def search(self, query: str) -> list[SearchResult]:
        del query
        return []


class _CapturingProvider:
    def __init__(self):
        self.received_count_tokens = None

    def search(self, query: str) -> list[SearchResult]:
        del query
        return [SearchResult(title="Result", href="https://example.com", body="body")]

    def format_for_prompt(
        self,
        query: str,
        results: list[SearchResult],
        *,
        token_budget: int,
        fetch_and_extract,
        count_tokens=None,
    ) -> str:
        del query, results, token_budget, fetch_and_extract
        self.received_count_tokens = count_tokens
        return "formatted"


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _UrlOpenResponse:
    def __init__(self, payload: bytes, *, headers: dict[str, str] | None = None):
        self._payload = payload
        self.headers = headers or {}

    def read(self, *args, **kwargs):
        del args, kwargs
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None


class SearchUtilsTests(unittest.TestCase):
    def test_extract_search_query_and_char_budget_helpers_cover_edge_cases(self):
        self.assertEqual(extract_search_query("hello <search>  cats and dogs  </search>"), "cats and dogs")
        self.assertIsNone(extract_search_query("hello"))
        self.assertEqual(_token_budget_to_char_budget(0), 4)

    def test_sentence_and_exact_truncation_helpers_cover_sentence_and_fallback_paths(self):
        self.assertEqual(_truncate_to_exact_token_budget("", 1, _word_token_count), "")
        self.assertEqual(_truncate_to_exact_token_budget("alpha beta", 2, _word_token_count), "alpha beta")
        self.assertEqual(_truncate_at_sentence("One. Two! Three", 7), "One.")
        self.assertEqual(_truncate_at_sentence("abcdef", 3), "abc")
        self.assertEqual(
            _truncate_to_exact_token_budget("alpha beta. gamma delta.", 2, _word_token_count),
            "alpha beta.",
        )
        self.assertEqual(
            _truncate_to_exact_token_budget("alphabet soup", 1, _word_token_count),
            "alphabet ",
        )

    def test_truncate_to_token_budget_falls_back_to_char_budget_without_counter(self):
        result = truncate_to_token_budget("abcdefghij", "query", token_budget=2)
        self.assertEqual(result, "abcdefgh")

    def test_truncate_to_token_budget_uses_exact_counter_for_single_paragraph(self):
        result = truncate_to_token_budget(
            "alpha beta gamma delta",
            "alpha",
            token_budget=2,
            count_tokens=_word_token_count,
        )
        self.assertEqual(result.strip(), "alpha beta")
        self.assertLessEqual(_word_token_count(result), 2)

    def test_truncate_to_token_budget_selects_relevant_paragraph_with_exact_counter(self):
        result = truncate_to_token_budget(
            "dogs bark loudly\n\ncats purr softly",
            "cats",
            token_budget=3,
            count_tokens=_word_token_count,
        )
        self.assertEqual(result, "cats purr softly")

    def test_truncate_to_token_budget_handles_empty_and_first_oversized_paragraphs(self):
        self.assertEqual(truncate_to_token_budget(" \n\n ", "query"), "")
        result = truncate_to_token_budget(
            "alpha beta gamma. delta epsilon zeta.\n\nsecondary paragraph",
            "alpha",
            token_budget=2,
            count_tokens=_word_token_count,
        )
        self.assertEqual(result, "alpha beta ")
        self.assertEqual(
            truncate_to_token_budget("abcdefghij\n\nshort", "query", token_budget=1),
            "abcd",
        )
        self.assertEqual(
            truncate_to_token_budget("cats purr\n\ndogs bark", "cats dogs", token_budget=5),
            "cats purr\n\ndogs bark",
        )

    def test_base_provider_format_for_prompt_hard_caps_exact_budget(self):
        provider = _ProviderStub(token_budget=6)
        output = provider.format_for_prompt(
            "alpha delta",
            [
                SearchResult(title="First", href="https://1", body="alpha beta gamma"),
                SearchResult(title="Second", href="https://2", body="delta epsilon zeta"),
            ],
            token_budget=6,
            fetch_and_extract=lambda url: None,
            count_tokens=_word_token_count,
        )
        self.assertIn("[1] First", output)
        self.assertLessEqual(_word_token_count(output), 6)

    def test_base_provider_format_for_prompt_sorts_by_year_and_errors_when_empty(self):
        provider = _ProviderStub(token_budget=6)

        with self.assertRaisesRegex(SearchError, "No search results found"):
            provider.format_for_prompt(
                "query",
                [],
                token_budget=6,
                fetch_and_extract=lambda url: None,
            )

        results = [
            SearchResult(title="Old article", href="https://1", body="plain body"),
            SearchResult(title="2026 article", href="https://2", body="recent body"),
        ]
        output = provider.format_for_prompt(
            "what happened in 2026",
            results,
            token_budget=100,
            fetch_and_extract=lambda url: "replacement body" if url.endswith("/2") else None,
        )
        self.assertTrue(output.startswith("[1] 2026 article"))

        with self.assertRaisesRegex(SearchError, "No relevant search results found"):
            provider.format_for_prompt(
                "query",
                [SearchResult(title="Missing", href="https://1", body="")],
                token_budget=10,
                fetch_and_extract=lambda url: None,
            )

        exact_output = provider.format_for_prompt(
            "query",
            [SearchResult(title="First", href="https://1", body="alpha beta gamma")],
            token_budget=1,
            fetch_and_extract=lambda url: None,
            count_tokens=_word_token_count,
        )
        self.assertLessEqual(_word_token_count(exact_output), 1)

        fallback_output = provider.format_for_prompt(
            "query",
            [SearchResult(title="First", href="https://1", body="abcdefghij")],
            token_budget=1,
            fetch_and_extract=lambda url: None,
        )
        self.assertLessEqual(len(fallback_output), 4)

    def test_brave_provider_applies_exact_local_cap_when_counter_present(self):
        provider = BraveSearchProvider(token_budget=5)
        output = provider.format_for_prompt(
            "query",
            [
                SearchResult(title="One", href="https://1", body="alpha beta gamma delta"),
                SearchResult(title="Two", href="https://2", body="epsilon zeta eta theta"),
            ],
            token_budget=5,
            fetch_and_extract=lambda url: None,
            count_tokens=_word_token_count,
        )
        self.assertLessEqual(_word_token_count(output), 5)

    def test_brave_provider_format_for_prompt_uses_href_and_raises_for_empty_inputs(self):
        provider = BraveSearchProvider(token_budget=5)
        output = provider.format_for_prompt(
            "query",
            [SearchResult(title="", href="https://example.com", body="")],
            token_budget=5,
            fetch_and_extract=lambda url: None,
        )
        self.assertEqual(output, "[1] https://example.com\nhttps://example.com")

        with self.assertRaisesRegex(SearchError, "No search results found"):
            provider.format_for_prompt(
                "query",
                [],
                token_budget=5,
                fetch_and_extract=lambda url: None,
            )

        with self.assertRaisesRegex(SearchError, "No relevant search results found"):
            provider.format_for_prompt(
                "query",
                [SearchResult(title="", href="", body="")],
                token_budget=5,
                fetch_and_extract=lambda url: None,
            )

    def test_ddgs_provider_emits_note_and_maps_results_and_errors(self):
        class _DDGS:
            def text(self, query, max_results):
                self.query = query
                self.max_results = max_results
                return [
                    {"title": "One", "href": "https://1", "body": "alpha"},
                    {"title": "No href", "href": "", "body": "skip"},
                ]

        with patch("sys.stderr") as stderr:
            provider = DDGSSearchProvider(max_results=2)
        stderr_output = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("impersonation warning", stderr_output)

        with patch.dict("sys.modules", {"ddgs": _module("ddgs", DDGS=lambda: _DDGS())}):
            results = provider.search("cats")
        self.assertEqual(results, [SearchResult(title="One", href="https://1", body="alpha")])

        with patch.dict(
            "sys.modules",
            {"ddgs": _module("ddgs", DDGS=lambda: _module("instance", text=lambda *a, **k: []))},
        ):
            self.assertEqual(provider.search("cats"), [])

        with patch.dict(
            "sys.modules",
            {"ddgs": _module("ddgs", DDGS=lambda: _module("instance", text=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))))},
        ):
            with self.assertRaisesRegex(SearchError, "Search unavailable: offline"):
                provider.search("cats")

    def test_brave_provider_search_handles_success_and_error_paths(self):
        payload = {
            "grounding": {
                "generic": [{"url": "https://1", "title": "One", "snippets": ["alpha", "beta"]}],
                "poi": {"url": "https://2", "snippets": ["poi snippet"]},
                "map": [{"url": "https://3"}],
            },
            "sources": {
                "https://2": {"title": "Two"},
                "https://3": {"title": "Three"},
            },
        }
        response = _UrlOpenResponse(
            gzip.compress(json.dumps(payload).encode("utf-8")),
            headers={"Content-Encoding": "gzip"},
        )
        provider = BraveSearchProvider(api_key="secret", max_results=2, token_budget=10)

        with patch("trillim.harnesses._search_utils.urllib.request.urlopen", return_value=response):
            results = provider.search("cats")

        self.assertEqual(
            results,
            [
                SearchResult(title="One", href="https://1", body="alpha\nbeta"),
                SearchResult(title="Two", href="https://2", body="poi snippet"),
                SearchResult(title="Three", href="https://3", body=""),
            ],
        )

        provider = BraveSearchProvider()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(SearchError, "requires SEARCH_API_KEY"):
                provider.search("cats")

        auth_error = urllib.error.HTTPError("https://example.com", 401, "unauthorized", hdrs=None, fp=None)
        with (
            patch.dict("os.environ", {"SEARCH_API_KEY": "token"}, clear=True),
            patch("trillim.harnesses._search_utils.urllib.request.urlopen", side_effect=auth_error),
        ):
            with self.assertRaisesRegex(SearchError, "authentication failed"):
                provider.search("cats")

        server_error = urllib.error.HTTPError(
            "https://example.com",
            500,
            "error",
            hdrs=None,
            fp=_UrlOpenResponse(b"detail"),
        )
        with (
            patch.dict("os.environ", {"SEARCH_API_KEY": "token"}, clear=True),
            patch("trillim.harnesses._search_utils.urllib.request.urlopen", side_effect=server_error),
        ):
            with self.assertRaisesRegex(SearchError, "Brave search API error \\(500\\). detail"):
                provider.search("cats")

        unreadable_error = urllib.error.HTTPError(
            "https://example.com",
            500,
            "error",
            hdrs=None,
            fp=_module(
                "broken-fp",
                read=lambda: (_ for _ in ()).throw(RuntimeError("no detail")),
                close=lambda: None,
            ),
        )
        with (
            patch.dict("os.environ", {"SEARCH_API_KEY": "token"}, clear=True),
            patch("trillim.harnesses._search_utils.urllib.request.urlopen", side_effect=unreadable_error),
        ):
            with self.assertRaisesRegex(SearchError, r"Brave search API error \(500\)\.$"):
                provider.search("cats")

        with (
            patch.dict("os.environ", {"SEARCH_API_KEY": "token"}, clear=True),
            patch("trillim.harnesses._search_utils.urllib.request.urlopen", side_effect=RuntimeError("offline")),
        ):
            with self.assertRaisesRegex(SearchError, "Search unavailable: offline"):
                provider.search("cats")

        payload = {
            "grounding": {
                "generic": [
                    {"url": "https://fallback"},
                    {"url": "", "title": "skip me"},
                ]
            },
            "sources": {},
        }
        with (
            patch.dict("os.environ", {"SEARCH_API_KEY": "token"}, clear=True),
            patch(
                "trillim.harnesses._search_utils.urllib.request.urlopen",
                return_value=_UrlOpenResponse(json.dumps(payload).encode("utf-8")),
            ),
        ):
            results = provider.search("cats")
        self.assertEqual(results, [SearchResult(title="https://fallback", href="https://fallback", body="")])

    def test_get_search_provider_validates_names(self):
        self.assertIsInstance(get_search_provider("ddgs"), DDGSSearchProvider)
        self.assertIsInstance(get_search_provider("brave"), BraveSearchProvider)
        with self.assertRaisesRegex(ValueError, "Unknown search provider"):
            get_search_provider("missing")

    def test_search_client_passes_token_counter_to_provider(self):
        client = SearchClient(provider_name="brave", count_tokens=_word_token_count)
        provider = _CapturingProvider()
        client.provider = provider

        output = client._search_sync("cats")

        self.assertEqual(output, "formatted")
        self.assertIs(provider.received_count_tokens, _word_token_count)

    def test_search_client_async_search_and_fetch_extract_paths(self):
        client = SearchClient(provider_name="brave", count_tokens=_word_token_count)
        provider = _CapturingProvider()
        client.provider = provider

        self.assertEqual(asyncio.run(client.search("cats")), "formatted")

        html_response = _UrlOpenResponse(b"<html>ok</html>")
        with (
            patch("trillim.harnesses._search_utils.urllib.request.urlopen", return_value=html_response),
            patch.dict("sys.modules", {"trafilatura": _module("trafilatura", extract=lambda *a, **k: "extracted text")}),
        ):
            self.assertEqual(client._fetch_and_extract("https://example.com"), "extracted text")

        with patch("trillim.harnesses._search_utils.urllib.request.urlopen", side_effect=RuntimeError("offline")):
            self.assertIsNone(client._fetch_and_extract("https://example.com"))
