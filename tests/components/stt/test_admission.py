"""Tests for STT admission control."""

from __future__ import annotations

import asyncio
import unittest

from trillim.components.stt._admission import TranscriptionAdmission
from trillim.errors import AdmissionRejectedError


class STTAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_allows_only_one_active_request(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        self.assertEqual(admission.active_count, 1)
        with self.assertRaisesRegex(AdmissionRejectedError, "STT is busy"):
            await admission.acquire()
        await lease.release()
        self.assertEqual(admission.active_count, 0)

    async def test_start_draining_rejects_new_work_until_finish_starting(self):
        admission = TranscriptionAdmission()
        await admission.start_draining()
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            await admission.acquire()
        await admission.finish_starting()
        lease = await admission.acquire()
        await lease.release()

    async def test_wait_for_idle_blocks_until_release(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        waiter = asyncio.create_task(admission.wait_for_idle())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        await lease.release()
        await waiter

    async def test_lease_release_is_idempotent(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        await lease.release()
        await lease.release()
        self.assertEqual(admission.active_count, 0)
