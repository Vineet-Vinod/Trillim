# What is Trillim?

Trillim is a high-performance CPU inference engine for BitNet models. It runs ternary-quantized language models — models whose weights are constrained to {-1, 0, 1} — using platform-specific SIMD optimizations (AVX2 on x86, NEON on ARM).

## Why Trillim?

Most LLM inference requires expensive GPUs. BitNet models use ternary weights, which means matrix multiplications reduce to simple additions and subtractions. This makes fast CPU inference possible without specialized hardware.

Trillim takes advantage of this by:

- **Running entirely on CPU** — no GPU, no CUDA, no ROCm. Any modern x86 or ARM machine works.
- **Using SIMD instructions** — AVX2 on Intel/AMD and NEON on Apple Silicon/ARM processors for maximum throughput.
- **Quantizing to a compact binary format** — ternary weights are packed tightly, reducing memory usage and improving cache performance.
- **Supporting LoRA adapters** — fine-tune a BitNet model with PEFT, then run the adapter on top of the ternary base model.
- **Providing an OpenAI-compatible API** — drop Trillim into existing applications that use the OpenAI client library. Chat completions, text completions, and streaming all work out of the box.
- **Including a voice pipeline** — speech-to-text (via Whisper) and text-to-speech are built-in server components. Enable them with a single flag.

## Who is Trillim for?

- **Developers** who want to run a local LLM without a GPU.
- **Hobbyists** who want to experiment with language models on consumer hardware.
- **Teams** building applications that need a self-hosted, low-cost inference backend.
- **Researchers** working with BitNet architectures who need a fast way to iterate.

## Supported Architectures

| Architecture | Description |
|---|---|
| `BitNetForCausalLM` | BitNet with ternary weights and ReLU² activation |
| `LlamaForCausalLM` | Llama-style with SiLU activation |

## Platform Support

| Platform | Status |
|---|---|
| x86_64 (AVX2) | Supported |
| ARM64 (NEON) | Supported |

## License

The Trillim Python SDK source code is MIT-licensed. The C++ inference engine binaries (`inference`, `trillim-quantize`) bundled in the pip package are proprietary — you may use them as part of Trillim but may not reverse-engineer or redistribute them separately.
