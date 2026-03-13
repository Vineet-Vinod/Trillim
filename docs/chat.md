# Interactive Chat

Use `trillim chat` when you want a local multi-turn terminal conversation with a model.

If you are embedding Trillim in Python instead of using the CLI, use `llm.session(...)` for multi-turn conversations. `llm.chat(...)` is the one-turn helper.

If you installed with `uv`, prefix each command on this page with `uv run`.

## Before You Start

- Pull a model first with `trillim pull Trillim/BitNet-TRNQ`
- If you plan to use Brave search, set `SEARCH_API_KEY`

## Start a Session

```bash
trillim chat <model_dir>
```

`model_dir` can be a local path or a HuggingFace model ID. A typical first run looks like:

```bash
trillim pull Trillim/BitNet-TRNQ
trillim chat Trillim/BitNet-TRNQ
```

## Session Behavior

Once the model loads, the prompt looks like:

```
Talk to BitNet-TRNQ (Ctrl+D or 'q' to quit, '/new' for new conversation)
>
```

Responses stream token by token:

```
> What is the capital of France?
The capital of France is Paris.
> Tell me more about it.
Paris is the largest city in France and serves as ...
```

Trillim keeps the full conversation history by default. When the conversation approaches the context limit, it automatically resets to the most recent message and continues.

Prompt caching is enabled for normal multi-turn usage, so follow-up turns are faster than the first turn.

## Session Controls

| Command | Description |
|---|---|
| `/new` | Clear the conversation history and start fresh |
| `q` | Quit the chat |
| `Ctrl+D` | Quit the chat |
| `Ctrl+C` | Quit the chat |

## Use a LoRA Adapter

Quantize the adapter first, then pass it to `--lora`:

```bash
trillim quantize <path-to-base-model> --adapter <path-to-adapter>
trillim chat Trillim/BitNet-TRNQ --lora <adapter-dir>
```

If the adapter tokenizer differs from the base model tokenizer, Trillim automatically uses the adapter tokenizer.

## Use the Search Harness

The `search` harness is intended for models that emit `<search>...</search>` tags during generation.

Enable it with:

```bash
trillim chat <model_dir> --harness search
```

If the model is not search-tuned, stay on the default harness:

```bash
trillim chat <model_dir> --harness default
```

Providers for `--search-provider`:

- `ddgs` for DuckDuckGo via `ddgs`
- `brave` for Brave Search, which requires `SEARCH_API_KEY`

```bash
export SEARCH_API_KEY=<your_api_key>
trillim chat <model_dir> --harness search --search-provider brave
```

Typical status markers during search:

- `[Spin-Jump-Spinning...]`
- `[Searching: <query>]`
- `[Synthesizing...]`
- `[Search unavailable]`

The harness allows up to 2 search rounds before the final streamed answer.

To inspect intermediate generations and fetched search context, set `SearchHarness.DEBUG = True` in `src/trillim/harnesses/_search.py`.

## Sampling and Performance

`trillim chat` uses the model's default sampling parameters. The relevant controls are:

| Parameter | Description |
|---|---|
| `temperature` | Lower values make output more deterministic |
| `top_k` | Limits sampling to the top K tokens |
| `top_p` | Nucleus sampling threshold |
| `repetition_penalty` | Reduces repeated tokens |

If you need to override those values per request, start `trillim serve` and send them to `POST /v1/chat/completions`.

Performance tips:

- Thread count defaults to `num_cores - 2`; override it with `--threads N`
- `--lora-quant` and `--unembed-quant` trade quality for lower memory use
- Use `/new` when long conversations start to slow down
