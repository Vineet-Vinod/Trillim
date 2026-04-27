# What Is Trillim?

Trillim is a local AI stack built to make CPU-first AI practical and pleasant to use.

The goal is straightforward:

- run useful local models without requiring a GPU stack
- give developers one path from terminal experimentation to embedded SDK use to local HTTP serving
- keep the public surface small, predictable, and easy to build on

Trillim ships three main entry points:

- a CLI for pulling, quantizing, chatting with, and serving bundles
- a Python SDK for embedding LLM, STT, and TTS directly
- a FastAPI server with OpenAI-compatible chat routes and optional voice routes, including PCM/WAV STT and streamed TTS

For LLM bundles, Trillim supports both BitNet-style ternary models and PrismML's Bonsai (1-bit and ternary) models through the same managed bundle workflow.

## Who It Is For

Trillim is a good fit when you want:

- local AI on laptops, desktops, or CPU-only servers
- a small SDK you can embed directly in Python
- a local server for OpenAI-style chat clients
- a path from raw checkpoints to managed local bundles

## Start Here

- use the [install guides](install-mac.md) if you are getting set up
- use the [CLI docs](cli.md) if you want to try Trillim quickly
- use [Python SDK](components.md) and [API Server](server.md) if you are integrating it into an app
- use [Advanced SDK and Server Notes](advanced.md) if you want the deeper operational details

## License

The Python SDK source code is MIT-licensed. The bundled inference binaries are proprietary and are licensed for use as part of Trillim. See [LICENSE](../LICENSE) for the full terms.
