"""Tests for tokenizer loading."""

import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.llm._tokenizer import load_tokenizer
from trillim.errors import ModelValidationError


class _TokenizerStub:
    def encode(self, text, add_special_tokens=True):
        return [1]

    def decode(self, token_ids, skip_special_tokens=True):
        return "x"


class TokenizerLoaderTests(unittest.TestCase):
    def test_load_tokenizer_uses_transformers_auto_tokenizer(self):
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_TokenizerStub(),
        ) as mock_loader:
            tokenizer = load_tokenizer(Path("/tmp/model"), trust_remote_code=False)

        self.assertIsInstance(tokenizer, _TokenizerStub)
        mock_loader.assert_called_once_with("/tmp/model", trust_remote_code=False)

    def test_load_tokenizer_rejects_invalid_tokenizer_objects(self):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
            with self.assertRaisesRegex(ModelValidationError, "encode/decode"):
                load_tokenizer(Path("/tmp/model"), trust_remote_code=False)
