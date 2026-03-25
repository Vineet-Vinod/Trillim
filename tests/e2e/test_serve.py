"""Process-level end-to-end tests for ``trillim serve``."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import http.client
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_ROOT = REPO_ROOT / "tests" / "e2e" / "support"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ServeProcess:
    def __init__(self, *, voice: bool = False) -> None:
        self.voice = voice
        self.port = _free_port()
        self._stack = ExitStack()
        self.process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._log_handle = None

    def __enter__(self) -> _ServeProcess:
        temp_dir = self._stack.enter_context(tempfile.TemporaryDirectory())
        home = Path(temp_dir) / "home"
        model_root = home / ".trillim" / "models" / "Trillim" / "fake"
        model_root.mkdir(parents=True, exist_ok=True)
        self._log_path = Path(temp_dir) / "server.log"
        self._log_handle = self._log_path.open("wb")

        env = os.environ.copy()
        pythonpath_entries = [str(SUPPORT_ROOT), str(REPO_ROOT)]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env.update(
            {
                "HOME": str(home),
                "PYTHONPATH": os.pathsep.join(pythonpath_entries),
                "TRILLIM_E2E_CHILD": "1",
                "TRILLIM_E2E_PORT": str(self.port),
            }
        )
        if os.environ.get("COVERAGE_PROCESS_START"):
            env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_PROCESS_START"]

        command = ["uv", "run"]
        if os.environ.get("COVERAGE_PROCESS_START"):
            command.extend(["--extra", "dev"])
        command.extend(["trillim", "serve", "Trillim/fake"])
        if self.voice:
            command.append("--voice")
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait_until_ready()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
        self._stack.close()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            process = self.process
            if process is None:
                raise AssertionError("Server process was not started")
            if process.poll() is not None:
                raise AssertionError(
                    "Server exited before becoming ready:\n" + self.read_log()
                )
            try:
                status, body, _headers = self.request("/healthz")
            except OSError:
                time.sleep(0.1)
                continue
            if status == 200 and json.loads(body) == {"status": "ok"}:
                return
            time.sleep(0.1)
        raise AssertionError("Timed out waiting for server readiness:\n" + self.read_log())

    def read_log(self) -> str:
        if self._log_path is None or not self._log_path.exists():
            return ""
        if self._log_handle is not None:
            self._log_handle.flush()
        return self._log_path.read_text(encoding="utf-8", errors="replace")

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> tuple[int, bytes, dict[str, str]]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            headers=headers or {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def json_request(
        self,
        path: str,
        payload: object,
        *,
        method: str = "POST",
        timeout: float = 5.0,
    ) -> tuple[int, dict[str, object], dict[str, str]]:
        status, body, headers = self.request(
            path,
            method=method,
            body=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            timeout=timeout,
        )
        return status, json.loads(body), headers

    def raw_request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> tuple[int, bytes, dict[str, str]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()


class ServeE2ETests(unittest.TestCase):
    def test_serve_reports_health_models_and_chat(self):
        with _ServeProcess() as server:
            status, body, _headers = server.request("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"status": "ok"})

            status, body, _headers = server.request("/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["data"][0]["id"], "fake")

            status, body, _headers = server.json_request(
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Say hi"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "hello")

    def test_serve_streams_chat_and_disables_swap_route(self):
        with _ServeProcess() as server:
            status, body, headers = server.request(
                "/v1/chat/completions",
                method="POST",
                body=json.dumps(
                    {
                        "messages": [{"role": "user", "content": "stream please"}],
                        "stream": True,
                    }
                ).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
            self.assertEqual(status, 200)
            text = body.decode("utf-8")
            self.assertIn('"delta": {"role": "assistant"}', text)
            self.assertIn('"content": "s"', text)
            self.assertIn('"content": "y"', text)
            self.assertIn("data: [DONE]", text)

            status, body, _headers = server.json_request(
                "/v1/models/swap",
                {"model_dir": "Trillim/next"},
            )
            self.assertEqual(status, 404)
            self.assertIn("Not Found", body["detail"])

    def test_serve_rejects_malformed_and_oversized_json_without_crashing(self):
        with _ServeProcess() as server:
            status, body, _headers = server.request(
                "/v1/chat/completions",
                method="POST",
                body=b"{",
                headers={"content-type": "application/json"},
            )
            self.assertEqual(status, 400)
            self.assertIn("invalid JSON body", json.loads(body)["detail"])

            status, body, _headers = server.raw_request(
                "/v1/chat/completions",
                method="POST",
                body=b"{}",
                headers={
                    "content-type": "application/json",
                    "content-length": str((2 * 1024 * 1024) + 32),
                },
            )
            self.assertEqual(status, 413)
            self.assertIn("request body too large", json.loads(body)["detail"])

            status, body, _headers = server.json_request(
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Still there?"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "hello")

    def test_serve_rejects_concurrent_generation_and_recovers(self):
        with _ServeProcess() as server, ThreadPoolExecutor(max_workers=1) as executor:
            slow_future = executor.submit(
                server.json_request,
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "slow request"}]},
            )

            rejected = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if slow_future.done():
                    break
                status, body, _headers = server.json_request(
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "second request"}]},
                )
                if status == 429:
                    rejected = body
                    break
                time.sleep(0.05)

            self.assertIsNotNone(rejected)
            self.assertIn("busy", rejected["detail"])

            slow_status, slow_body, _headers = slow_future.result(timeout=5.0)
            self.assertEqual(slow_status, 200)
            self.assertEqual(slow_body["choices"][0]["message"]["content"], "hello")

            status, body, _headers = server.request("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"status": "ok"})

    def test_serve_rejects_concurrent_streaming_generation_before_sse(self):
        with _ServeProcess() as server, ThreadPoolExecutor(max_workers=1) as executor:
            slow_future = executor.submit(
                server.json_request,
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "slow request"}]},
            )

            stream_body = json.dumps(
                {
                    "messages": [{"role": "user", "content": "stream please"}],
                    "stream": True,
                }
            ).encode("utf-8")
            busy_confirmed = False
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if slow_future.done():
                    break
                status, _body, _headers = server.json_request(
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "probe request"}]},
                )
                if status == 429:
                    busy_confirmed = True
                    break
                time.sleep(0.05)

            self.assertTrue(busy_confirmed)
            status, body, _headers = server.request(
                "/v1/chat/completions",
                method="POST",
                body=stream_body,
                headers={"content-type": "application/json"},
            )
            self.assertEqual(status, 429)
            rejected = json.loads(body)
            self.assertIn("busy", rejected["detail"])

            slow_status, slow_body, _headers = slow_future.result(timeout=5.0)
            self.assertEqual(slow_status, 200)
            self.assertEqual(slow_body["choices"][0]["message"]["content"], "hello")


class VoiceServeE2ETests(unittest.TestCase):
    def test_serve_voice_handles_transcription_voice_store_and_speech(self):
        with _ServeProcess(voice=True) as server:
            status, body, _headers = server.request(
                "/v1/audio/transcriptions?language=en",
                method="POST",
                body=b"abc",
                headers={"content-type": "audio/wav"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"text": "en:abc"})

            status, body, _headers = server.request(
                "/v1/voices",
                method="POST",
                body=b"voice",
                headers={"name": "custom"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"name": "custom", "status": "created"})

            status, body, _headers = server.request("/v1/voices")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"voices": ["alba", "marius", "custom"]})

            status, body, _headers = server.request(
                "/v1/audio/speech",
                method="POST",
                body=b"hello world",
                headers={"voice": "alba", "speed": "1.0"},
            )
            self.assertEqual(status, 200)
            text = body.decode("utf-8")
            self.assertIn("event: audio", text)
            self.assertIn("event: done", text)

            status, body, _headers = server.request(
                "/v1/voices/custom",
                method="DELETE",
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"name": "custom", "status": "deleted"})

    def test_serve_voice_rejects_invalid_utf8(self):
        with _ServeProcess(voice=True) as server:
            status, body, _headers = server.request(
                "/v1/audio/speech",
                method="POST",
                body=b"\xff",
                headers={"voice": "alba"},
            )
            self.assertEqual(status, 400)
            self.assertIn("valid UTF-8", json.loads(body)["detail"])
