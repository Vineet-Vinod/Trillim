"""Unified CLI entry point for Trillim."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import prompt as better_input
from prompt_toolkit.key_binding import KeyBindings

from trillim import LLM, STT, TTS, Runtime, Server, _model_store
from trillim._bundle_metadata import CURRENT_FORMAT_VERSION
from trillim.components.llm._events import ChatDoneEvent, ChatTokenEvent
from trillim.components.llm._model_dir import validate_lora_dir, validate_model_dir
from trillim.errors import ModelValidationError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SUPPORTED_FORMAT_VERSION = CURRENT_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class _LocalBundle:
    model_id: str
    entry_type: str
    size_bytes: int
    size_human: str


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes:.0f} {unit}"
            return f"{size_bytes:.1f} {unit}".rstrip("0").rstrip(".")
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _validate_pull_id(model_id: str) -> str:
    namespace, name = _model_store.parse_store_id(model_id, error_type=RuntimeError)
    if namespace != "Trillim":
        raise RuntimeError(
            "trillim pull only supports Hugging Face IDs of the form Trillim/<name>"
        )
    return f"{namespace}/{name}"


def _normalize_platform_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"arm64", "aarch64"}:
        return "aarch64"
    if normalized in {"amd64", "x86_64"}:
        return "x86_64"
    return normalized


def _warn_on_trillim_config(path: Path) -> None:
    config_path = path / "trillim_config.json"
    if not config_path.is_file():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Could not read trillim_config.json: {exc}")
        return
    if not isinstance(payload, dict):
        print("Warning: Could not interpret trillim_config.json metadata")
        return
    format_version = payload.get("format_version")
    if isinstance(format_version, int) and format_version > SUPPORTED_FORMAT_VERSION:
        print(
            "Warning: Model format version "
            f"{format_version} is newer than supported version {SUPPORTED_FORMAT_VERSION}. "
            "Consider upgrading trillim."
        )
    platforms = payload.get("platforms")
    current_platform = platform.machine()
    normalized_current_platform = _normalize_platform_name(current_platform)
    normalized_platforms = (
        {_normalize_platform_name(str(entry)) for entry in platforms}
        if isinstance(platforms, list)
        else set()
    )
    if normalized_platforms and normalized_current_platform not in normalized_platforms:
        print(
            f"Warning: This model lists platforms {platforms} but your system is {current_platform}."
        )


def _require_remote_code_opt_in(store_id: str, *, label: str, trust_remote_code: bool) -> None:
    if trust_remote_code:
        return
    bundle_path = _model_store.resolve_existing_store_id(store_id, error_type=RuntimeError)
    config_path = bundle_path / "trillim_config.json"
    if not config_path.is_file():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or not bool(payload.get("remote_code", False)):
        return
    raise RuntimeError(
        f"{label} '{store_id}' requires trust_remote_code. Re-run with --trust-remote-code."
    )


def _pull_model(model_id: str, *, revision: str | None, force: bool) -> Path:
    normalized = _validate_pull_id(model_id)
    local_dir = _model_store.store_path_for_id(normalized, error_type=RuntimeError)
    if local_dir.is_dir() and not force:
        print(f"Model '{normalized}' already exists at {local_dir}")
        print("Use --force to re-download.")
        return local_dir
    if force and local_dir.exists():
        if local_dir.is_symlink() or local_dir.is_file():
            local_dir.unlink()
        else:
            shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pulling {normalized} ...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            normalized,
            local_dir=str(local_dir),
            revision=revision,
            force_download=force,
        )
    except Exception as exc:  # pragma: no cover - exercised via class-name branches
        class_name = type(exc).__name__
        if class_name == "RepositoryNotFoundError":
            raise RuntimeError(
                f"Repository '{normalized}' not found on Hugging Face"
            ) from exc
        if class_name == "GatedRepoError":
            raise RuntimeError(
                f"'{normalized}' is a gated repository. Authenticate with: hf auth login"
            ) from exc
        raise
    print(f"Downloaded to {local_dir}")
    _warn_on_trillim_config(local_dir)
    return local_dir


def _iter_local_bundles(namespace: str) -> list[_LocalBundle]:
    root = _model_store.store_namespace_root(namespace)
    bundles: list[_LocalBundle] = []
    if not root.is_dir():
        return bundles
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            validate_model_dir(entry)
        except ModelValidationError:
            pass
        else:
            size_bytes = (entry / "qmodel.tensors").stat().st_size
            bundles.append(
                _LocalBundle(
                    model_id=f"{namespace}/{entry.name}",
                    entry_type="model",
                    size_bytes=size_bytes,
                    size_human=_human_size(size_bytes),
                )
            )
            continue
        try:
            validate_lora_dir(entry)
        except ModelValidationError:
            continue
        size_bytes = (entry / "qmodel.lora").stat().st_size
        bundles.append(
            _LocalBundle(
                model_id=f"{namespace}/{entry.name}",
                entry_type="adapter",
                size_bytes=size_bytes,
                size_human=_human_size(size_bytes),
            )
        )
    return bundles


def _print_local_table(title: str, bundles: list[_LocalBundle]) -> None:
    print(title)
    if not bundles:
        print("(none)")
        return
    model_width = max(len(bundle.model_id) for bundle in bundles)
    model_width = max(model_width, len("MODEL ID"))
    type_width = max(len(bundle.entry_type) for bundle in bundles)
    type_width = max(type_width, len("TYPE"))
    size_width = max(len(bundle.size_human) for bundle in bundles)
    size_width = max(size_width, len("SIZE"))
    print(
        f"{'MODEL ID':<{model_width}}  {'TYPE':<{type_width}}  {'SIZE':>{size_width}}"
    )
    print(f"{'-' * model_width}  {'-' * type_width}  {'-' * size_width}")
    for bundle in bundles:
        print(
            f"{bundle.model_id:<{model_width}}  "
            f"{bundle.entry_type:<{type_width}}  "
            f"{bundle.size_human:>{size_width}}"
        )


def _local_downloaded_ids() -> set[str]:
    if not _model_store.DOWNLOADED_ROOT.is_dir():
        return set()
    return {
        f"Trillim/{entry.name}"
        for entry in _model_store.DOWNLOADED_ROOT.iterdir()
        if entry.is_dir()
    }


def _list_remote_models() -> list[dict[str, object]]:
    try:
        from huggingface_hub import list_models as hf_list_models
    except ImportError as exc:  # pragma: no cover - dependency is installed in project
        raise RuntimeError(
            "huggingface_hub is required. Install it with: uv add huggingface_hub"
        ) from exc
    local_ids = _local_downloaded_ids()
    results: list[dict[str, object]] = []
    try:
        for repo in hf_list_models(author="Trillim", full=True):
            sibling_names = [sibling.rfilename for sibling in (repo.siblings or ())]
            entry_type = "adapter" if "qmodel.lora" in sibling_names else "model"
            base_model = ""
            for tag in repo.tags or ():
                if tag.startswith("base_model:") and not tag.startswith(
                    ("base_model:quantized:", "base_model:adapter:")
                ):
                    base_model = tag.split(":", 1)[1]
                    break
            last_modified = ""
            if repo.last_modified is not None:
                last_modified = repo.last_modified.strftime("%Y-%m-%d")
            results.append(
                {
                    "model_id": repo.id,
                    "type": entry_type,
                    "downloads": repo.downloads or 0,
                    "last_modified": last_modified,
                    "base_model": base_model,
                    "local": repo.id in local_ids,
                }
            )
    except Exception as exc:  # pragma: no cover - exercised via class-name branch
        class_name = type(exc).__name__
        if class_name in {"ConnectionError", "HfHubHTTPError"}:
            raise RuntimeError(
                f"Failed to fetch models from Hugging Face: {exc}"
            ) from exc
        raise
    return results


def _print_available_table(title: str, entries: list[dict[str, object]]) -> None:
    print(title)
    if not entries:
        print("(none)")
        return
    model_width = max(len(str(entry["model_id"])) for entry in entries)
    model_width = max(model_width, len("MODEL ID"))
    base_width = max(len(str(entry.get("base_model", ""))) for entry in entries)
    base_width = max(base_width, len("BASE MODEL"))
    pulls_width = len("PULLS")
    modified_width = len("MODIFIED")
    status_width = len("STATUS")
    print(
        f"{'MODEL ID':<{model_width}}  {'BASE MODEL':<{base_width}}  "
        f"{'PULLS':>{pulls_width}}  {'MODIFIED':<{modified_width}}  {'STATUS':<{status_width}}"
    )
    print(
        f"{'-' * model_width}  {'-' * base_width}  "
        f"{'-' * pulls_width}  {'-' * modified_width}  {'-' * status_width}"
    )
    for entry in entries:
        status = "local" if entry.get("local") else ""
        print(
            f"{str(entry['model_id']):<{model_width}}  "
            f"{str(entry.get('base_model', '')):<{base_width}}  "
            f"{int(entry.get('downloads', 0)):>{pulls_width}}  "
            f"{str(entry.get('last_modified', '')):<{modified_width}}  "
            f"{status:<{status_width}}"
        )


def _preflight_voice_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("faster_whisper", "numpy", "soundfile", "pocket_tts"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Voice dependencies are not installed "
            f"(missing: {joined}). Install the optional voice extras, for example: uv sync --extra voice."
        )


def _stream_assistant_turn(runtime: Runtime, session, messages_snapshot) -> object:
    stream = session.stream_chat()
    saw_token = False
    try:
        for event in stream:
            if isinstance(event, ChatTokenEvent):
                if event.text:
                    print(event.text, end="", flush=True)
                    saw_token = True
            elif isinstance(event, ChatDoneEvent) and event.text and not saw_token:
                print(event.text, end="", flush=True)
        print()
        return session
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
        try:
            stream.close()
        except Exception:
            pass
        finally:
            try:
                session.close()
            except Exception:
                pass
        raise
    except Exception:
        try:
            session.close()
        except Exception:
            pass
        raise
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _make_chat_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-g")
    def _open_editor(event) -> None:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
        buffer = event.app.current_buffer
        with tempfile.NamedTemporaryFile(
            suffix=".txt",
            mode="w+",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(buffer.text)
            temp_path = handle.name
        try:
            subprocess.call([editor, temp_path])
            with open(temp_path, encoding="utf-8") as handle:
                text = handle.read()
            buffer.text = text
            buffer.cursor_position = len(text)
        finally:
            os.unlink(temp_path)

    return kb


def _run_chat(
    model_id: str,
    adapter_id: str | None,
    *,
    trust_remote_code: bool = False,
) -> int:
    _require_remote_code_opt_in(
        model_id,
        label="Model",
        trust_remote_code=trust_remote_code,
    )
    if adapter_id is not None:
        _require_remote_code_opt_in(
            adapter_id,
            label="Adapter",
            trust_remote_code=trust_remote_code,
        )
    runtime = Runtime(
        LLM(
            model_id,
            lora_dir=adapter_id,
            trust_remote_code=trust_remote_code,
        )
    )
    with runtime:
        session = runtime.llm.open_session()
        key_bindings = _make_chat_key_bindings()
        print(f"Model: {model_id}")
        if adapter_id is not None:
            print(f"Adapter: {adapter_id}")
        print("Commands: /new to reset, q to quit, Ctrl+G for editor")
        try:
            while True:
                try:
                    prompt = better_input("user: ", key_bindings=key_bindings)
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    raise
                stripped = prompt.strip()
                if not stripped:
                    continue
                if stripped == "q":
                    break
                if stripped == "/new":
                    session.close()
                    session = runtime.llm.open_session()
                    print("Conversation reset.")
                    continue
                session.add_user(prompt)
                snapshot = session.messages
                print("assistant: ", end="", flush=True)
                session = _stream_assistant_turn(runtime, session, snapshot)
        finally:
            try:
                session.close()
            except Exception:
                pass
    return 0


def _run_serve(
    model_id: str,
    *,
    voice: bool,
    trust_remote_code: bool = False,
) -> int:
    _require_remote_code_opt_in(
        model_id,
        label="Model",
        trust_remote_code=trust_remote_code,
    )
    if voice:
        _preflight_voice_dependencies()
    llm = LLM(model_id, trust_remote_code=trust_remote_code)
    components = [llm]
    if voice:
        components.extend([STT(), TTS()])
    Server(*components, allow_hot_swap=False).run(host=DEFAULT_HOST, port=DEFAULT_PORT)
    return 0


def _run_quantize_command(args: argparse.Namespace) -> int:
    from trillim.quantize import quantize

    quantize(args.model_dir, args.adapter_dir)
    return 0


def _run_pull_command(args: argparse.Namespace) -> int:
    _pull_model(args.model_id, revision=args.revision, force=args.force)
    return 0


def _run_list_command() -> int:
    _print_local_table("Downloaded", _iter_local_bundles("Trillim"))
    print()
    _print_local_table("Local", _iter_local_bundles("Local"))
    return 0


def _run_models_command() -> int:
    entries = _list_remote_models()
    models = [entry for entry in entries if entry["type"] == "model"]
    adapters = [entry for entry in entries if entry["type"] == "adapter"]
    _print_available_table("Models", models)
    print()
    _print_available_table("Adapters", adapters)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trillim",
        description="Trillim - The fastest inference framework to run AI on CPUs",
    )
    subparsers = parser.add_subparsers(dest="command")

    pull_parser = subparsers.add_parser(
        "pull",
        help="Download a pre-quantized model from the Trillim Hugging Face org",
    )
    pull_parser.add_argument("model_id", help="Hugging Face model ID (Trillim/<name>)")
    pull_parser.add_argument("--revision", help="Branch, tag, or commit hash")
    pull_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if the model already exists locally",
    )

    subparsers.add_parser(
        "list", help="List locally available downloaded and local bundles"
    )
    subparsers.add_parser(
        "models", help="List available models in the Trillim Hugging Face org"
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Open a quick multi-turn chat shell"
    )
    chat_parser.add_argument(
        "model_dir", help="Store-qualified model ID (Trillim/<name> or Local/<name>)"
    )
    chat_parser.add_argument(
        "adapter_dir",
        nargs="?",
        help="Optional store-qualified adapter ID (Trillim/<name> or Local/<name>)",
    )
    chat_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom tokenizer/config code referenced by the bundle",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the demo API server")
    serve_parser.add_argument(
        "model_dir", help="Store-qualified model ID (Trillim/<name> or Local/<name>)"
    )
    serve_parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable STT and TTS components",
    )
    serve_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom tokenizer/config code referenced by the bundle",
    )

    quantize_parser = subparsers.add_parser(
        "quantize",
        help="Quantize one local model directory or adapter directory into Local/",
    )
    quantize_parser.add_argument("model_dir", help="Local filesystem path to the source model directory")
    quantize_parser.add_argument(
        "adapter_dir",
        nargs="?",
        help="Optional local filesystem path to the source adapter directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    handlers = {
        "pull": lambda: _run_pull_command(args),
        "list": _run_list_command,
        "models": _run_models_command,
        "chat": lambda: _run_chat(
            args.model_dir,
            args.adapter_dir,
            trust_remote_code=args.trust_remote_code,
        ),
        "serve": lambda: _run_serve(
            args.model_dir,
            voice=args.voice,
            trust_remote_code=args.trust_remote_code,
        ),
        "quantize": lambda: _run_quantize_command(args),
    }
    try:
        return handlers[args.command]()
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
