from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.tts._limits import MAX_CUSTOM_VOICES, MAX_VOICE_UPLOAD_BYTES, VOICE_MANIFEST_NAME
from trillim.components.tts._validation import PayloadTooLargeError, save_voice_state_safetensors
from trillim.components.tts._voices import (
    VOICE_STATE_SUFFIX,
    VoiceStoreTamperedError,
    _copy_source_audio_sync,
    _ensure_store_root,
    _has_non_legacy_children,
    _load_manifest,
    _load_manifest_entry,
    _load_optional_state,
    _raise_if_symlink_for_write,
    _warn_for_inventory_mismatch,
    _warn_for_legacy_files,
    _warn_if_symlink,
    copy_source_audio,
    delete_custom_voice,
    load_custom_voice_states,
    publish_custom_voice,
    spool_voice_bytes,
)
from trillim.errors import InvalidRequestError

from tests.components.tts.support import sample_voice_state


class VoicePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name) / "voices"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_load_custom_voice_states_loads_only_valid_safetensors(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(sample_voice_state(), state_path)
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        states = await load_custom_voice_states(
            self.root,
            built_in_voice_names=("alba",),
        )

        self.assertEqual(list(states), ["custom"])
        self.assertEqual(states["custom"]["module"]["cache"].tolist(), [1.0])

    async def test_load_custom_voice_states_skips_legacy_and_invalid_files_with_warning(self):
        self.root.mkdir(parents=True)
        (self.root / "legacy.state").write_bytes(b"legacy")
        bad_storage_id = _storage_id_for_name("bad")
        bad_path = self.root / f"{bad_storage_id}{VOICE_STATE_SUFFIX}"
        bad_path.write_bytes(b"not safetensors")
        manifest = {
            "voices": [
                {
                    "name": "bad",
                    "storage_id": bad_storage_id,
                    "size_bytes": bad_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("legacy", "\n".join(logs.output))
        self.assertIn("valid safetensors", "\n".join(logs.output))

    async def test_publish_and_delete_custom_voice_update_disk_manifest(self):
        name, state = await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=sample_voice_state(),
            existing_names={"alba"},
        )

        self.assertEqual(name, "custom")
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])
        manifest_path = self.root / VOICE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["voices"][0]["name"], "custom")
        state_path = self.root / f"{manifest['voices'][0]['storage_id']}{VOICE_STATE_SUFFIX}"
        self.assertTrue(state_path.exists())

        deleted = await delete_custom_voice(self.root, name="custom")
        self.assertEqual(deleted, "custom")
        self.assertFalse(state_path.exists())
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"voices": []},
        )

    async def test_publish_rejects_duplicate_names(self):
        await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=sample_voice_state(),
            existing_names={"alba"},
        )

        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await publish_custom_voice(
                self.root,
                name="custom",
                voice_state=sample_voice_state(),
                existing_names={"alba", "custom"},
            )
        manifest_path = self.root / VOICE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state_path = self.root / f"{manifest['voices'][0]['storage_id']}{VOICE_STATE_SUFFIX}"
        state_path.unlink()
        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await publish_custom_voice(
                self.root,
                name="custom",
                voice_state=sample_voice_state(),
                existing_names={"alba"},
            )

        with self.assertRaisesRegex(InvalidRequestError, "already contains"):
            await publish_custom_voice(
                self.root,
                name="another",
                voice_state=sample_voice_state(),
                existing_names={f"voice{i}" for i in range(MAX_CUSTOM_VOICES)},
            )
        with self.assertRaisesRegex(InvalidRequestError, "malformed"):
            await publish_custom_voice(
                self.root,
                name="badstate",
                voice_state={},
                existing_names={"alba"},
            )
        def save_empty_state(_state, path):
            Path(path).write_bytes(b"")

        with patch("trillim.components.tts._voices.save_voice_state_safetensors", save_empty_state):
            with patch("trillim.components.tts._voices.load_safe_voice_state_safetensors") as load:
                load.return_value = sample_voice_state()
                with self.assertRaisesRegex(InvalidRequestError, "voice state exceeds"):
                    await publish_custom_voice(
                        self.root,
                        name="emptystate",
                        voice_state=sample_voice_state(),
                        existing_names={"alba"},
                    )

        full_manifest = {
            "voices": [
                {
                    "name": f"stored{i}",
                    "storage_id": _storage_id_for_name(f"stored{i}"),
                    "size_bytes": MAX_VOICE_UPLOAD_BYTES,
                }
                for i in range(10)
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(full_manifest), encoding="utf-8")
        with self.assertRaisesRegex(InvalidRequestError, "custom voice storage exceeds"):
            await publish_custom_voice(
                self.root,
                name="huge",
                voice_state=sample_voice_state(),
                existing_names={"alba"},
            )

    async def test_publish_rolls_back_state_file_when_manifest_write_fails(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"

        with patch(
            "trillim.components.tts._voices.atomic_write_bytes",
            side_effect=OSError("manifest boom"),
        ):
            with self.assertRaisesRegex(OSError, "manifest boom"):
                await publish_custom_voice(
                    self.root,
                    name="custom",
                    voice_state=sample_voice_state(),
                    existing_names={"alba"},
                )

        self.assertFalse(state_path.exists())

    async def test_delete_rejects_symlinked_state_file_for_write_safety(self):
        await publish_custom_voice(
            self.root,
            name="custom",
            voice_state=sample_voice_state(),
            existing_names={"alba"},
        )
        state_path = self.root / f"{_storage_id_for_name('custom')}{VOICE_STATE_SUFFIX}"
        target_path = self.root / "target.safetensors"
        state_path.rename(target_path)
        state_path.symlink_to(target_path)

        with self.assertLogs("trillim.components.tts._voices", level="WARNING"):
            with self.assertRaisesRegex(VoiceStoreTamperedError, "symlinks"):
                await delete_custom_voice(self.root, name="custom")

    async def test_delete_missing_voice_returns_normalized_name(self):
        self.assertEqual(await delete_custom_voice(self.root, name="missing"), "missing")

    async def test_malformed_manifest_is_skipped_with_warning(self):
        self.root.mkdir(parents=True)
        (self.root / VOICE_MANIFEST_NAME).write_text("{", encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("manifest is malformed", "\n".join(logs.output))

    async def test_malformed_manifest_payload_shapes_are_skipped_with_warning(self):
        self.root.mkdir(parents=True)
        for payload in ([], {"voices": "bad"}):
            with self.subTest(payload=payload):
                (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
                with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
                    self.assertEqual(
                        await load_custom_voice_states(self.root, built_in_voice_names=("alba",)),
                        {},
                    )
                self.assertIn("manifest is malformed", "\n".join(logs.output))

    async def test_manifest_root_symlink_and_file_root_are_skipped(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        target = self.root.parent / "target"
        target.mkdir()
        self.root.symlink_to(target)
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertEqual(_load_manifest(self.root, built_ins=set()), {})
        self.assertIn("symlink", "\n".join(logs.output))

        self.root.unlink()
        self.root.write_text("not a directory", encoding="utf-8")
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertEqual(_load_manifest(self.root, built_ins=set()), {})
        self.assertIn("root is malformed", "\n".join(logs.output))

    async def test_malformed_manifest_entries_are_skipped_with_warning(self):
        self.root.mkdir(parents=True)
        manifest = {
            "voices": [
                "not an entry",
                {"name": "bad-name", "storage_id": "x", "size_bytes": 1},
                {"name": "alba", "storage_id": _storage_id_for_name("alba"), "size_bytes": 1},
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("malformed custom TTS voice manifest entry", "\n".join(logs.output))

    async def test_manifest_entry_shape_validation_rejects_each_bad_field(self):
        cases = (
            {},
            {"name": "", "storage_id": "x", "size_bytes": 1},
            {"name": "custom", "storage_id": 1, "size_bytes": 1},
            {"name": "custom", "storage_id": _storage_id_for_name("custom"), "size_bytes": "1"},
            {"name": "custom", "storage_id": _storage_id_for_name("custom"), "size_bytes": 0},
            {"name": "custom", "storage_id": "wrong", "size_bytes": 1},
        )
        for item in cases:
            with self.subTest(item=item):
                with self.assertLogs("trillim.components.tts._voices", level="WARNING"):
                    self.assertIsNone(_load_manifest_entry(item))

    async def test_missing_manifest_with_legacy_and_unexpected_files_warns_and_skips(self):
        self.root.mkdir(parents=True)
        (self.root / "legacy.state").write_bytes(b"legacy")
        (self.root / "unexpected.safetensors").write_bytes(b"not tracked")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        output = "\n".join(logs.output)
        self.assertEqual(states, {})
        self.assertIn("legacy", output)
        self.assertIn("manifest is missing", output)

    async def test_missing_manifest_without_inventory_noise_returns_empty(self):
        self.root.mkdir(parents=True)
        self.assertEqual(
            await load_custom_voice_states(self.root, built_in_voice_names=("alba",)),
            {},
        )

    async def test_inventory_mismatch_warns_and_keeps_valid_manifest_voices(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(sample_voice_state(), state_path)
        (self.root / "orphan.safetensors").write_bytes(b"orphan")
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(list(states), ["custom"])
        self.assertIn("unexpected TTS voice store files", "\n".join(logs.output))

    async def test_inventory_and_legacy_warning_os_errors_are_handled(self):
        with patch.object(Path, "iterdir", side_effect=OSError("boom")):
            with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
                _warn_for_inventory_mismatch(self.root, {})
            self.assertIn("inventory check", "\n".join(logs.output))

            with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
                _warn_for_legacy_files(self.root)
            self.assertIn("voice store root is malformed", "\n".join(logs.output))

            with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
                self.assertFalse(_has_non_legacy_children(self.root))
            self.assertIn("voice store root is malformed", "\n".join(logs.output))

        self.root.mkdir(parents=True)
        (self.root / "regular.txt").write_text("regular", encoding="utf-8")
        _warn_for_legacy_files(self.root)
        target = self.root / "target.txt"
        target.write_text("target", encoding="utf-8")
        (self.root / "linked.state").symlink_to(target)
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            _warn_for_legacy_files(self.root)
        self.assertIn("symlink", "\n".join(logs.output))

    async def test_manifest_size_mismatch_skips_voice(self):
        storage_id = _storage_id_for_name("custom")
        state_path = self.root / f"{storage_id}{VOICE_STATE_SUFFIX}"
        self.root.mkdir(parents=True)
        save_voice_state_safetensors(sample_voice_state(), state_path)
        manifest = {
            "voices": [
                {
                    "name": "custom",
                    "storage_id": storage_id,
                    "size_bytes": state_path.stat().st_size + 1,
                }
            ]
        }
        (self.root / VOICE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            states = await load_custom_voice_states(
                self.root,
                built_in_voice_names=("alba",),
            )

        self.assertEqual(states, {})
        self.assertIn("size does not match manifest", "\n".join(logs.output))

    async def test_manifest_and_state_symlink_guards(self):
        self.root.mkdir(parents=True)
        manifest_target = self.root / "target.json"
        manifest_target.write_text(json.dumps({"voices": []}), encoding="utf-8")
        (self.root / VOICE_MANIFEST_NAME).symlink_to(manifest_target)
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertEqual(_load_manifest(self.root, built_ins=set()), {})
        self.assertIn("symlink", "\n".join(logs.output))

        state_target = self.root / "target.safetensors"
        save_voice_state_safetensors(sample_voice_state(), state_target)
        state_link = self.root / f"{_storage_id_for_name('custom')}{VOICE_STATE_SUFFIX}"
        state_link.symlink_to(state_target)
        entry = _load_manifest_entry(
            {
                "name": "custom",
                "storage_id": _storage_id_for_name("custom"),
                "size_bytes": state_target.stat().st_size,
            }
        )
        assert entry is not None
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertIsNone(_load_optional_state(entry, state_link))
        self.assertIn("symlink", "\n".join(logs.output))

    async def test_optional_state_missing_unreadable_and_non_file_paths_are_skipped(self):
        entry = _load_manifest_entry(
            {
                "name": "custom",
                "storage_id": _storage_id_for_name("custom"),
                "size_bytes": 1,
            }
        )
        assert entry is not None
        state_path = self.root / f"{entry.storage_id}{VOICE_STATE_SUFFIX}"
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertIsNone(_load_optional_state(entry, state_path))
        self.assertIn("missing", "\n".join(logs.output))

        self.root.mkdir(parents=True, exist_ok=True)
        state_path.mkdir()
        with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
            self.assertIsNone(_load_optional_state(entry, state_path))
        self.assertIn("not a regular file", "\n".join(logs.output))

        with patch("trillim.components.tts._voices._warn_if_symlink", return_value=False):
            with patch.object(Path, "stat", side_effect=OSError("stat")):
                with self.assertLogs("trillim.components.tts._voices", level="WARNING") as logs:
                    self.assertIsNone(_load_optional_state(entry, state_path))
                self.assertIn("could not be read", "\n".join(logs.output))

    async def test_store_root_write_guards(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(VoiceStoreTamperedError, "root is malformed"):
            _ensure_store_root(self.root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target)
            self.assertTrue(_warn_if_symlink(link, "reason"))
            with self.assertRaisesRegex(VoiceStoreTamperedError, "symlinks"):
                _raise_if_symlink_for_write(link)

    async def test_spool_and_copy_voice_uploads(self):
        spool_dir = self.root / "spool"
        upload = await spool_voice_bytes(b"voice", spool_dir=spool_dir)
        self.assertEqual(upload.path.read_bytes(), b"voice")
        self.assertEqual(upload.size_bytes, 5)

        source = self.root / "source.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"audio")
        copied = await copy_source_audio(source, spool_dir=spool_dir)
        self.assertEqual(copied.path.read_bytes(), b"audio")
        self.assertEqual(copied.size_bytes, 5)

    async def test_spool_and_copy_cleanup_on_errors(self):
        spool_dir = self.root / "spool"
        with patch("trillim.components.tts._voices.os.fdopen", side_effect=OSError("write")):
            with self.assertRaisesRegex(OSError, "write"):
                await spool_voice_bytes(b"voice", spool_dir=spool_dir)

        source = self.root / "large.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x" * (MAX_VOICE_UPLOAD_BYTES + 1))
        fd = os.open(source, os.O_RDONLY)
        try:
            with self.assertRaisesRegex(PayloadTooLargeError, "voice upload exceeds"):
                _copy_source_audio_sync(fd, spool_dir)
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)

        fd = os.open(source, os.O_RDONLY)
        try:
            with patch("trillim.components.tts._voices._create_owned_temp_file", side_effect=OSError("temp")):
                with self.assertRaisesRegex(OSError, "temp"):
                    _copy_source_audio_sync(fd, spool_dir)
                fd = -1
        finally:
            if fd >= 0:
                os.close(fd)

        fd = os.open(source, os.O_RDONLY)
        temp_fd, temp_name = tempfile.mkstemp(dir=self.root, prefix="copy-", suffix=".audio")
        temp_path = Path(temp_name)
        os.close(temp_fd)
        try:
            with patch(
                "trillim.components.tts._voices._create_owned_temp_file",
                return_value=(os.open(temp_path, os.O_WRONLY), temp_path),
            ):
                with patch("trillim.components.tts._voices.os.fdopen", side_effect=OSError("fdopen")):
                    with self.assertRaisesRegex(OSError, "fdopen"):
                        _copy_source_audio_sync(fd, spool_dir)
                fd = -1
        finally:
            if fd >= 0:
                os.close(fd)


def _storage_id_for_name(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
