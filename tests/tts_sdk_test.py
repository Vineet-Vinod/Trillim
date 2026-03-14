# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the public TTS SDK helpers."""

import asyncio
import threading
import unittest

from trillim.server._models import SpeechRequest
from trillim.server._tts import TTSEngine, _StreamingPCMStretcher
import trillim
from trillim import SentenceChunker
from trillim.server import TTS


class _FakeTTSEngine:
    DEFAULT_VOICE = "alba"

    def __init__(self):
        self.default_voice = "alba"
        self.sample_rate = 24000
        self.speed = 1.0
        self.calls: list[tuple] = []
        self._voices = [
            {"voice_id": "alba", "name": "alba", "type": "predefined"},
            {"voice_id": "jean", "name": "jean", "type": "predefined"},
            {"voice_id": "custom", "name": "custom", "type": "custom"},
        ]

    def list_voices(self) -> list[dict]:
        self.calls.append(("list_voices",))
        return list(self._voices)

    async def register_voice(self, voice_id: str, audio_bytes: bytes) -> None:
        self.calls.append(("register_voice", voice_id, audio_bytes))
        self._voices.append({"voice_id": voice_id, "name": voice_id, "type": "custom"})

    async def delete_voice(self, voice_id: str) -> None:
        self.calls.append(("delete_voice", voice_id))
        self._voices = [voice for voice in self._voices if voice["voice_id"] != voice_id]
        if self.default_voice == voice_id:
            self.default_voice = self.DEFAULT_VOICE

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ):
        self.calls.append(("synthesize_stream", text, voice, speed))
        yield b"pcm-a"
        yield b"pcm-b"


class _FakeTensor:
    def __init__(self, samples):
        self._samples = samples

    def numpy(self):
        import numpy as np

        return np.array(self._samples, dtype=np.float32)


class _FakeStreamingModel:
    def __init__(self, chunks):
        self._chunks = chunks

    def generate_audio_stream(self, **_):
        for chunk in self._chunks:
            yield _FakeTensor(chunk)

    def get_state_for_audio_prompt(self, prompt, truncate=False):
        return {"prompt": prompt, "truncate": truncate}


class _GatedStreamingModel(_FakeStreamingModel):
    def __init__(self, chunks, gates):
        super().__init__(chunks)
        self._gates = gates

    def generate_audio_stream(self, **_):
        for index, chunk in enumerate(self._chunks):
            gate = self._gates.get(index)
            if gate is not None:
                gate.wait(timeout=1.0)
            yield _FakeTensor(chunk)


class TTSSdkTests(unittest.IsolatedAsyncioTestCase):
    def _make_tts(self) -> tuple[TTS, _FakeTTSEngine]:
        tts = TTS()
        engine = _FakeTTSEngine()
        tts._engine = engine
        tts._default_voice = engine.default_voice
        return tts, engine

    async def test_public_exports_include_sentence_chunker(self):
        self.assertIs(trillim.SentenceChunker, SentenceChunker)

    async def test_default_voice_getter_and_setter_use_component_api(self):
        tts, engine = self._make_tts()

        self.assertEqual(tts.default_voice, "alba")
        tts.default_voice = "jean"

        self.assertEqual(tts.default_voice, "jean")
        self.assertEqual(engine.default_voice, "jean")

    async def test_default_voice_can_be_set_before_start(self):
        tts = TTS()

        tts.default_voice = "jean"

        self.assertEqual(tts.default_voice, "jean")

    async def test_default_voice_rejects_empty_value(self):
        tts = TTS()

        with self.assertRaisesRegex(ValueError, "default_voice must not be empty"):
            tts.default_voice = ""

    async def test_default_voice_rejects_unknown_voice_when_started(self):
        tts, _ = self._make_tts()

        with self.assertRaisesRegex(ValueError, "Unknown voice"):
            tts.default_voice = "missing"

    async def test_sample_rate_and_list_voices_are_public(self):
        tts, engine = self._make_tts()

        self.assertEqual(tts.sample_rate, 24000)
        self.assertEqual(tts.list_voices(), engine.list_voices())

    async def test_speed_getter_and_setter_use_component_api(self):
        tts, engine = self._make_tts()

        self.assertEqual(tts.speed, 1.0)
        tts.speed = 1.5

        self.assertEqual(tts.speed, 1.5)
        self.assertEqual(engine.speed, 1.5)

    async def test_speed_can_be_set_before_start(self):
        tts = TTS(speed=1.25)

        self.assertEqual(tts.speed, 1.25)

    async def test_speed_rejects_out_of_range_values(self):
        tts = TTS()

        with self.assertRaisesRegex(ValueError, "speed must be between 0.25 and 4.0"):
            tts.speed = 0.2

        with self.assertRaisesRegex(ValueError, "speed must be between 0.25 and 4.0"):
            TTS(speed=4.5)

    async def test_register_and_delete_voice_use_public_wrappers(self):
        tts, engine = self._make_tts()
        tts.default_voice = "custom"

        await tts.register_voice("newvoice", b"wav-bytes")
        await tts.delete_voice("custom")

        self.assertIn(("register_voice", "newvoice", b"wav-bytes"), engine.calls)
        self.assertIn(("delete_voice", "custom"), engine.calls)
        self.assertEqual(tts.default_voice, "alba")

    async def test_synthesize_stream_and_wav_use_public_wrappers(self):
        tts, engine = self._make_tts()

        chunks = [
            chunk
            async for chunk in tts.synthesize_stream(
                "hello",
                voice="jean",
                speed=1.5,
            )
        ]
        wav_bytes = await tts.synthesize_wav("hello", voice="jean", speed=1.5)

        self.assertEqual(chunks, [b"pcm-a", b"pcm-b"])
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertEqual(wav_bytes[44:], b"pcm-apcm-b")
        self.assertIn(("synthesize_stream", "hello", "jean", 1.5), engine.calls)
        self.assertEqual(
            engine.calls.count(("synthesize_stream", "hello", "jean", 1.5)),
            2,
        )

    async def test_synthesis_rejects_empty_input(self):
        tts, _ = self._make_tts()

        with self.assertRaisesRegex(ValueError, "input text is empty"):
            await tts.synthesize_wav("   ")

        with self.assertRaisesRegex(ValueError, "input text is empty"):
            [chunk async for chunk in tts.synthesize_stream("   ")]

    async def test_synthesis_requires_started_component(self):
        tts = TTS()

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            await tts.synthesize_wav("hello")

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            [chunk async for chunk in tts.synthesize_stream("hello")]

    async def test_tts_public_api_requires_started_component(self):
        tts = TTS()

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            _ = tts.sample_rate

        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            tts.list_voices()

    async def test_sentence_chunker_is_top_level_and_usable(self):
        chunker = SentenceChunker()

        parts = chunker.feed("Hello world. Another sentence")
        remainder = chunker.flush()

        self.assertEqual(parts, ["Hello world."])
        self.assertEqual(remainder, "Another sentence")

    async def test_speech_request_accepts_speed(self):
        request = SpeechRequest(input="Hello", speed=1.5)

        self.assertEqual(request.speed, 1.5)

    async def test_pitch_preserving_speed_changes_duration(self):
        sample_rate = 24000
        pcm = _sine_pcm_bytes(sample_rate=sample_rate, frequency_hz=440.0, duration_s=1.0)

        slower_stretcher = _StreamingPCMStretcher(0.5)
        slower = b"".join([slower_stretcher.push(pcm), slower_stretcher.finish()])
        faster_stretcher = _StreamingPCMStretcher(2.0)
        faster = b"".join([faster_stretcher.push(pcm), faster_stretcher.finish()])

        self.assertGreater(len(slower), len(pcm))
        self.assertLess(len(faster), len(pcm))
        self.assertAlmostEqual(
            len(slower) / len(pcm),
            2.0,
            delta=0.2,
        )
        self.assertAlmostEqual(
            len(faster) / len(pcm),
            0.5,
            delta=0.15,
        )
        self.assertAlmostEqual(
            _dominant_frequency_hz(slower, sample_rate),
            440.0,
            delta=25.0,
        )
        self.assertAlmostEqual(
            _dominant_frequency_hz(faster, sample_rate),
            440.0,
            delta=25.0,
        )

    async def test_pcm_stretcher_handles_empty_pcm(self):
        stretcher = _StreamingPCMStretcher(1.5)
        self.assertEqual(b"".join([stretcher.push(b""), stretcher.finish()]), b"")

    async def test_pcm_stretcher_handles_tiny_pcm(self):
        pcm = _sine_pcm_bytes(sample_rate=24000, frequency_hz=440.0, duration_s=0.0008)

        stretcher = _StreamingPCMStretcher(2.0)
        stretched = b"".join([stretcher.push(pcm), stretcher.finish()])

        self.assertTrue(stretched)
        self.assertLessEqual(len(stretched), len(pcm))

    async def test_pcm_stretcher_handles_short_pcm(self):
        pcm = _sine_pcm_bytes(sample_rate=24000, frequency_hz=440.0, duration_s=0.03)
        stretcher = _StreamingPCMStretcher(2.0)
        stretched = b"".join([stretcher.push(pcm), stretcher.finish()])

        self.assertTrue(stretched)
        self.assertLess(len(stretched), len(pcm))

    async def test_streaming_pcm_stretcher_handles_odd_byte_boundaries(self):
        pcm = _sine_pcm_bytes(sample_rate=24000, frequency_hz=440.0, duration_s=0.15)
        stretcher = _StreamingPCMStretcher(1.5)

        pieces = [
            stretcher.push(pcm[:101]),
            stretcher.push(pcm[101:733]),
            stretcher.push(pcm[733:]),
            stretcher.finish(),
        ]
        stretched = b"".join(pieces)

        self.assertTrue(stretched)
        self.assertAlmostEqual(
            _dominant_frequency_hz(stretched, 24000),
            440.0,
            delta=35.0,
        )

    async def test_engine_synthesize_stream_applies_speed_progressively(self):
        pcm = _sine_pcm_bytes(sample_rate=24000, frequency_hz=440.0, duration_s=0.4)
        release_second = threading.Event()
        chunk_a = _pcm_bytes_to_float_samples(pcm[:4096])
        chunk_b = _pcm_bytes_to_float_samples(pcm[4096:])
        normal_engine = TTSEngine(speed=1.0)
        normal_engine._model = _FakeStreamingModel([chunk_a, chunk_b])
        normal_engine._voice_states["alba"] = {"prompt": "alba"}
        fast_engine = TTSEngine(speed=1.0)
        fast_engine._model = _GatedStreamingModel([chunk_a, chunk_b], {1: release_second})
        fast_engine._voice_states["alba"] = {"prompt": "alba"}

        normal = b"".join(
            [chunk async for chunk in normal_engine.synthesize_stream("hello", speed=1.0)]
        )
        iterator = fast_engine.synthesize_stream("hello", speed=2.0).__aiter__()
        first_chunk = await asyncio.wait_for(iterator.__anext__(), timeout=0.5)
        release_second.set()
        remaining = [chunk async for chunk in iterator]
        faster = b"".join([first_chunk, *remaining])

        self.assertTrue(first_chunk)
        self.assertGreater(len(normal), len(faster))
        self.assertAlmostEqual(
            _dominant_frequency_hz(faster, 24000),
            440.0,
            delta=35.0,
        )


def _sine_pcm_bytes(
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float,
) -> bytes:
    import math
    import struct

    sample_count = int(sample_rate * duration_s)
    samples = []
    for index in range(sample_count):
        value = math.sin((2.0 * math.pi * frequency_hz * index) / sample_rate)
        samples.append(struct.pack("<h", int(value * 32767)))
    return b"".join(samples)


def _dominant_frequency_hz(pcm_bytes: bytes, sample_rate: int) -> float:
    import numpy as np

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    window = np.hanning(samples.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * window))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum[1:]) + 1)])


def _pcm_bytes_to_float_samples(pcm_bytes: bytes) -> list[float]:
    import struct

    sample_count = len(pcm_bytes) // 2
    return [
        struct.unpack_from("<h", pcm_bytes, offset * 2)[0] / 32767.0
        for offset in range(sample_count)
    ]


if __name__ == "__main__":
    unittest.main()
