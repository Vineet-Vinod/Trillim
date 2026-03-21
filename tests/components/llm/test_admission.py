"""Tests for LLM admission control."""

import asyncio
import unittest

from trillim.components.llm._admission import GenerationAdmission
from trillim.errors import AdmissionRejectedError


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_admission_allows_one_active_generation(self):
        admission = GenerationAdmission()

        lease = await admission.acquire()
        self.assertEqual(admission.active_count, 1)
        with self.assertRaisesRegex(AdmissionRejectedError, "busy"):
            await admission.acquire()
        await lease.release()
        self.assertEqual(admission.active_count, 0)

    async def test_admission_drains_and_resumes(self):
        admission = GenerationAdmission()
        await admission.start_draining()
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            await admission.acquire()
        await admission.finish_swapping()
        lease = await admission.acquire()
        await lease.release()

    async def test_wait_for_idle_observes_active_releases(self):
        admission = GenerationAdmission()
        lease = await admission.acquire()

        waiter = asyncio.create_task(admission.wait_for_idle(timeout=1))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        await lease.release()
        await waiter
