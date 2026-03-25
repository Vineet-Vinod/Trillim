"""Tests for the managed TTS voice store."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trillim.components.tts._voices import (
    VoiceStore,
    VoiceStoreTamperedError,
    _storage_id_for_name,
    copy_source_audio,
    spool_request_voice_stream,
    spool_voice_bytes,
)
from trillim.errors import InvalidRequestError


class _UnsafeState:
    pass


def _valid_state_bytes() -> bytes:
    buffer = io.BytesIO()
    torch.save({"layer": {"cache": torch.tensor([1.0])}}, buffer)
    return buffer.getvalue()


class _Chunks:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class TTSVoiceStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"
        self.store = VoiceStore(self.root, built_in_voice_names=("alba", "marius"))

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_register_list_resolve_and_delete_custom_voice(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(audio_path: Path) -> bytes:
            self.assertEqual(audio_path.read_bytes(), b"voice")
            return _valid_state_bytes()

        try:
            self.assertEqual(
                await self.store.register_owned_upload(
                    name="custom",
                    upload=upload,
                    build_voice_state=fake_builder,
                ),
                "custom",
            )
        finally:
            upload.path.unlink(missing_ok=True)

        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])
        resolved = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        self.assertEqual(resolved.kind, "state_file")
        self.assertTrue(Path(resolved.reference).exists())
        Path(resolved.reference).unlink(missing_ok=True)

        self.assertEqual(await self.store.delete("custom"), "custom")
        self.assertEqual(await self.store.list_names(), ["alba", "marius"])

    async def test_duplicate_or_builtin_name_is_rejected(self):
        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await self.store.ensure_name_available("alba")

    async def test_delete_rejects_default_or_missing_voice(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)
        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()
        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)
        with self.assertRaisesRegex(InvalidRequestError, "default_voice"):
            await self.store.delete("custom", protected_name="custom")
        with self.assertRaises(KeyError):
            await self.store.delete("missing")

    async def test_delete_failure_keeps_manifest_and_state_consistent(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)

        with patch(
            "trillim.components.tts._voices.unlink_if_exists",
            side_effect=PermissionError("read only"),
        ):
            with self.assertRaisesRegex(PermissionError, "read only"):
                await self.store.delete("custom")
        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])
        resolved = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        Path(resolved.reference).unlink(missing_ok=True)

    async def test_spool_request_stream_and_copy_source_audio(self):
        owned = await spool_request_voice_stream(_Chunks([b"a", b"b"]), spool_dir=self.spool_dir)
        self.assertEqual(owned.size_bytes, 2)
        self.assertEqual(owned.path.read_bytes(), b"ab")
        owned.path.unlink(missing_ok=True)

        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"hello")
        copied = await copy_source_audio(source, spool_dir=self.spool_dir)
        self.assertEqual(copied.path.read_bytes(), b"hello")
        copied.path.unlink(missing_ok=True)

    async def test_tampered_manifest_disables_custom_voice_functionality(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": "../escape",
                            "size_bytes": 4,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_non_directory_voice_store_root_fails_closed(self):
        self.root.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_missing_manifest_with_leftover_files_fails_closed(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{_storage_id_for_name('custom')}.state").write_bytes(_valid_state_bytes())
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("custom")
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_tampered_state_file_disables_custom_voice_resolution(self):
        upload = await spool_voice_bytes(b"voice", spool_dir=self.spool_dir)

        async def fake_builder(_path: Path) -> bytes:
            return _valid_state_bytes()

        try:
            await self.store.register_owned_upload(
                name="custom",
                upload=upload,
                build_voice_state=fake_builder,
            )
        finally:
            upload.path.unlink(missing_ok=True)

        initial = await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        Path(initial.reference).unlink(missing_ok=True)

        buffer = io.BytesIO()
        torch.save({"bad": _UnsafeState()}, buffer)
        state_path = self.root / f"{_storage_id_for_name('custom')}.state"
        state_path.write_bytes(buffer.getvalue())

        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.resolve_for_session("custom", spool_dir=self.spool_dir)
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        resolved = await self.store.resolve_for_session("alba", spool_dir=self.spool_dir)
        self.assertEqual(resolved.reference, "alba")

    async def test_manifest_is_revalidated_after_initial_successful_read(self):
        storage_id = _storage_id_for_name("custom")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{storage_id}.state").write_bytes(_valid_state_bytes())
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": storage_id,
                            "size_bytes": len(_valid_state_bytes()),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(await self.store.list_names(), ["alba", "marius", "custom"])

        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": "../escape",
                            "size_bytes": len(_valid_state_bytes()),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.list_names()
        with self.assertRaisesRegex(VoiceStoreTamperedError, "tampered"):
            await self.store.ensure_name_available("fresh")

    async def test_untracked_state_files_fail_closed_with_cleanup_guidance(self):
        storage_id = _storage_id_for_name("custom")
        self.root.mkdir(parents=True, exist_ok=True)
        state_bytes = _valid_state_bytes()
        (self.root / f"{storage_id}.state").write_bytes(state_bytes)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": storage_id,
                            "size_bytes": len(state_bytes),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (self.root / f"{_storage_id_for_name('orphan')}.state").write_bytes(state_bytes)

        with self.assertRaisesRegex(
            VoiceStoreTamperedError,
            r"Delete stale \.state files",
        ):
            await self.store.list_names()
        with self.assertRaisesRegex(
            VoiceStoreTamperedError,
            r"Delete stale \.state files",
        ):
            await self.store.ensure_name_available("fresh")
