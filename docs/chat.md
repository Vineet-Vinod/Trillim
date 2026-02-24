# Interactive Chat

The `trillim chat` command starts an interactive terminal session where you can have a multi-turn conversation with a model.

## Starting a Chat

```bash
trillim chat <model_dir>
```

`model_dir` can be a local path or a HuggingFace model ID. If you've previously pulled a model with `trillim pull`, you can use the same ID:

```bash
trillim pull Trillim/BitNet-TRNQ
trillim chat Trillim/BitNet-TRNQ
```

## The Chat Interface

Once the model loads, you'll see a prompt:

```
Talk to BitNet-TRNQ (Ctrl+D or 'q' to quit, '/new' for new conversation)
>
```

Type your message and press Enter. The model's response streams token-by-token:

```
> What is the capital of France?
Model Response: The capital of France is Paris.
> Tell me more about it.
Model Response: Paris is the largest city in France and serves as ...
```

## Commands

| Command | Description |
|---|---|
| `/new` | Clear the conversation history and start fresh |
| `q` | Quit the chat |
| `Ctrl+D` | Quit the chat |
| `Ctrl+C` | Quit the chat |

## Multi-Turn Conversations

Trillim maintains full conversation history across turns. Each message you send includes all prior messages so the model has full context.

When the conversation approaches the model's context window limit, Trillim automatically resets to just the last message and continues from there.

### Prompt Caching

Trillim uses incremental prompt caching to speed up multi-turn conversations. On each turn, only the new portion of the conversation is tokenized and sent to the engine — the KV cache from prior turns is reused. This means follow-up messages are faster than the first message.

## Using LoRA Adapters

Load a fine-tuned LoRA adapter on top of the base model:

```bash
trillim chat Trillim/BitNet-TRNQ --lora Trillim/MyAdapter-TRNQ
```

The adapter must have been quantized with `trillim quantize` first. If the adapter's tokenizer differs from the base model's, Trillim automatically uses the adapter's tokenizer.

## Sampling Parameters

The chat uses default sampling parameters from the model's configuration. These typically include:

| Parameter | Description |
|---|---|
| `temperature` | Controls randomness. Lower values produce more deterministic output |
| `top_k` | Limits sampling to the top K most likely tokens |
| `top_p` | Nucleus sampling — limits to tokens whose cumulative probability exceeds P |
| `repetition_penalty` | Penalizes repeated tokens to reduce repetitive output |

To customize sampling parameters, use the API server instead (see [Server](server.md)), which exposes these as request fields.

## Performance Tips

- **Thread count**: By default, Trillim uses `num_cores - 2` threads. Override with `--threads N` if needed.
- **Quantization options**: Use `--lora-quant` and `--unembed-quant` to control how LoRA and unembed layers are quantized. Lower quantization (e.g. `q4_0`) uses less memory at a small quality cost.
- **Context resets**: If responses slow down as the conversation gets long, use `/new` to start fresh.
