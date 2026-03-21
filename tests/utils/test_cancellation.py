"""Tests for cancellation helpers."""

import asyncio
import unittest

from trillim.errors import OperationCancelledError
from trillim.utils.cancellation import CancellationSource


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_token_waits_and_raises(self):
        source = CancellationSource()

        async def _cancel_later():
            await asyncio.sleep(0)
            source.cancel()

        task = asyncio.create_task(_cancel_later())
        await source.token.wait()
        await task
        self.assertTrue(source.cancelled())
        with self.assertRaises(OperationCancelledError):
            source.token.raise_if_cancelled()

