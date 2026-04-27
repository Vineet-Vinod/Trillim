# Benchmarks

This page has two goals:

- preserve the existing directional benchmark material used in the older Trillim docs
- show you how to benchmark the current SDK and server surfaces yourself without collecting misleading numbers

## Existing Directional Benchmark Data

These charts compare DarkNet, the inference engine behind Trillim, against bitnet.cpp on consumer CPUs.

Treat them as directional, not absolute. CPU inference numbers are highly sensitive to thermals, boost behavior, memory bandwidth, OS scheduling, and prompt shape.

### How Those Runs Were Collected

The published runs used the same process:

- fresh system restart before each benchmark session
- 5 warmup runs for both engines before recording data
- interleaved execution between engines to reduce time-drift bias
- cool-down between runs until CPU temperature returned to `45C`

These controls reduce noise, but they do not remove it.

### Decode Throughput

Run A:

![Decode throughput run A](imgs/DecodeA.png)

Run B:

![Decode throughput run B](imgs/DecodeB.png)

Takeaways:

- decode throughput is broadly comparable to bitnet.cpp
- DarkNet reaches higher peaks
- the gap is most visible once `num_threads >= 4`

### Runtime Quantization

#### Q4_0

Run A:

![Q4_0 run A](imgs/Q4_0A.png)

Run B:

![Q4_0 run B](imgs/Q4_0B.png)

#### Q5_0

Run A:

![Q5_0 run A](imgs/Q5_0A.png)

Run B:

![Q5_0 run B](imgs/Q5_0B.png)

#### Q6_K

Run A:

![Q6_K run A](imgs/Q6_KA.png)

Run B:

![Q6_K run B](imgs/Q6_KB.png)

#### Q8_0

Run A:

![Q8_0 run A](imgs/Q8_0A.png)

Run B:

![Q8_0 run B](imgs/Q8_0B.png)

### Main Takeaways

- DarkNet tends to pull ahead once the thread count reaches 4 or more.
- Average decode rates are close to bitnet.cpp, but DarkNet shows higher peak throughput.
- Runtime quantization behavior depends heavily on the CPU, thermal budget, and memory subsystem.

### Limits of That Data

- consumer CPUs vary in boost behavior, thermal limits, and power settings
- background processes and OS scheduling can materially affect short runs
- memory bandwidth and cache behavior can dominate results as thread count increases
- SMT and Hyper-Threading behavior differs by CPU generation and workload shape
- compiler flags, kernel versions, and microcode updates can shift the results
- prompt mix, context length, and warm-up policy also affect measured decode rates

## What to Measure in Trillim Today

For current Trillim SDK and server benchmarks, the useful split is:

- startup time: process start plus model load
- first-token latency: request start to first emitted text
- decode throughput: tokens per second after generation starts
- end-to-end latency: full request wall time
- memory footprint: resident memory while the model is loaded and while a request is active

Those numbers move independently. A change that helps decode throughput may not help startup or first-token latency.

## Benchmarking Rules That Matter

If you want comparisons that are actually useful, keep these fixed across runs:

- same CPU and memory configuration
- same OS power mode
- same model bundle
- same thread count
- same prompt and same `max_tokens`
- same warm-up policy
- same harness configuration
- same adapter configuration

Also avoid mixing unlike cases:

- do not compare base-model runs with adapter runs unless that is the point of the test
- do not compare `default` harness requests with `search` harness requests
- do not compare warmed cache hits with cold first runs without labeling them clearly

## Recommended Methodology

For a useful local benchmark pass:

1. Reboot or at least stop unrelated heavy processes.
2. Pull or prepare the exact bundle you want to test.
3. Warm up the model with a few requests before recording anything.
4. Record at least 5 measured runs for each configuration.
5. Keep prompts stable across runs.
6. Record first-token latency and full latency separately.
7. If you compare thread counts, test the same prompt at each setting.

At minimum, report:

- CPU model
- OS
- Python version
- Trillim version
- model ID
- adapter ID, if any
- thread count
- harness name
- measured prompt
- `max_tokens`

## Benchmark the SDK

This example measures end-to-end wall time for repeated one-turn calls through the sync `Runtime` facade. It does not run extra warmups, so consumer laptops are less likely to throttle before the measured runs start.

```python
import statistics
import time

from trillim import LLM, Runtime

PROMPT = "Explain CPU inference in one sentence."


def run_trials(model_id: str, *, trials: int = 5) -> None:
    durations = []
    with Runtime(LLM(model_id)) as runtime:
        for _ in range(trials):
            start = time.perf_counter()
            with runtime.llm.open_session() as session:
                session.collect(PROMPT)
            durations.append(time.perf_counter() - start)

    print("runs:", [round(value, 3) for value in durations])
    print("mean_seconds:", round(statistics.mean(durations), 3))
    print("median_seconds:", round(statistics.median(durations), 3))


run_trials("Trillim/BitNet-TRNQ")
```

If you want first-token timing instead of full-response timing, benchmark `ChatSession.generate(...)` and stop the timer on the first `ChatTokenEvent`.

## Benchmark Streaming First-Token Latency

```python
import time

from trillim import LLM, Runtime
from trillim.components.llm import ChatTokenEvent

PROMPT = "Give me a short answer about CPUs."

with Runtime(LLM("Trillim/BitNet-TRNQ")) as runtime:
    with runtime.llm.open_session() as session:
        start = time.perf_counter()
        for event in session.generate(PROMPT):
            if isinstance(event, ChatTokenEvent) and event.text:
                print("first_token_seconds:", round(time.perf_counter() - start, 3))
                break
```

## Benchmark the Server

If you want to measure the HTTP path instead of the SDK path, start the server first:

```bash
trillim serve Trillim/BitNet-TRNQ
```

Then drive one stable request shape repeatedly from a client script. This example measures full-request latency with the OpenAI-compatible route:

```python
import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
BODY = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "Explain CPU inference in one sentence."}
        ]
    }
).encode("utf-8")
HEADERS = {"content-type": "application/json"}

durations = []
for _ in range(5):
    request = urllib.request.Request(URL, data=BODY, headers=HEADERS, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
    durations.append(time.perf_counter() - start)

print("runs:", [round(value, 3) for value in durations])
print("mean_seconds:", round(statistics.mean(durations), 3))
```

## Search and Voice Benchmarks Need Separate Buckets

Do not roll these into the same charts as plain LLM inference:

- search-harness runs include network and fetch time
- STT includes upload, normalization, and transcription time
- TTS includes segmentation, synthesis, and streaming time

Benchmark them separately and label them clearly.

## What to Compare

Useful comparisons in practice:

- one model across several thread counts
- base model vs base model plus adapter
- SDK path vs HTTP path
- warm runs vs cold runs
- one CPU against another CPU on the same prompt set

Less useful comparisons:

- different prompts with different output lengths
- different harnesses in the same chart
- a cached follow-up turn against a cold first turn without calling that out

## How to Read the Results

A few practical expectations help interpret the numbers:

- short prompts emphasize startup and first-token latency more than decode rate
- long outputs emphasize decode throughput
- multi-turn sessions can benefit from prompt-state reuse in ways that one-turn cold tests do not
- search-harness runs are dominated by more than just inference

If you publish or share benchmark numbers, include the methodology beside them. Without that, the numbers are not portable enough to be useful.
