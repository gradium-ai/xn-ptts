# Phonon serving optimization: voice cache

This fork starts with a bounded LRU cache and a reproducible HTTP benchmark. Upstream
already caches voice conditioning. At revision
`31e4e9f80275e432b570e17f5ccbefa895eff109`, inserting a seventeenth retained voice
prefix clears the whole cache. This change evicts only the least recently used
prefixes needed to fit the configured entry and KV-byte budgets.

The intended gain is lower time to first audio when a mostly hot voice set sees
occasional additional voices. It does not accelerate the transformer or Mimi kernels.
Single-voice traffic should be treated as a regression control, not a source of gains.

## What is implemented

- LRU retention, default 16 entries and 64 MiB of flow-LM KV payloads.
- Limits through `SynthBuilder::voice_cache`, server flags, and environment variables.
- Coherent occupancy and lifetime hit, miss, insertion, eviction, invalidation, and
  bypass counters at `GET /v1/audio/cache`.
- A standard-library Python HTTP harness that records every warm-up and measured
  request, including HTTP 429 and transport failures, and hashes returned PCM.
- Unit tests and opt-in real-checkpoint checks for cache eviction, disabled retention,
  voice/condition invalidation, and isolation between concurrent sessions.

The KV budget excludes weights, shared conditions, allocation overhead and active
request buffers. Evicted tensors can remain alive while a session references them.
It is not an RSS or VRAM ceiling. The queue uses a linear lookup for the small default
capacity; profile before configuring thousands of retained voices.

## Reproduce on an Apple silicon CPU

From the repository root, prepare assets **outside** the checkout:

```sh
uv run --no-project --with huggingface_hub scripts/prepare-voice-bench.py ../models
cargo build --locked --release -p ptts-openai-server --features accelerate,kai
RAYON_NUM_THREADS=4 XN_KAI=1 target/release/ptts-openai-server \
  --model ../models/phonon-31m-en --quant q8 --device cpu --lang en \
  --voice-catalog ../models/voices/voices.jsonl \
  --addr 127.0.0.1:18880 --max-concurrent-requests 1 --temperature 0.3 --seed 42 \
  --voice-cache-entries 16 --voice-cache-bytes 67108864
```

In a second terminal:

```sh
python3 scripts/bench-serving.py \
  --base-url http://127.0.0.1:18880 \
  --workload benchmarks/voice-cache/workloads/voice-pressure.jsonl \
  --output ../results/voice-pressure-fork-1 --label fork \
  --server-revision "$(git rev-parse HEAD)" \
  --server-binary target/release/ptts-openai-server --concurrency 1
```

For the baseline, build upstream revision `31e4e9f80275e432b570e17f5ccbefa895eff109`
in a separate checkout with the **same** Rust version, lockfile, build features,
native CPU flags and machine. Keep its binary separately. Launch with the same
command, omitting the two new `--voice-cache-*` flags. Use that exact revision and
binary in the harness metadata. A missing cache endpoint is recorded and does not
prevent baseline benchmarking. The prepared assets are pinned to:

- `gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f`
- `gradium/phonon-voices@da6c9ec0d47be8df5c87c730a3aa97187f3229c3`

Run only one server at a time. Restart between every workload/variant/trial; the
server seed advances on requests, so identical request order and warm-up are needed
for waveform comparisons. Alternate baseline/fork order across repeated trials.
Record hardware, OS, compiler, feature flags, thread count, all server arguments,
asset hashes and binary hashes. Do not build or run another inference job during
measurement. The server revision argument is caller-declared; binary hashes provide
an additional identity check.

Other CPUs can build without `accelerate,kai`. CUDA and Metal require their own
feature/backend choices and separate measurements. These CPU results are not GPU
performance estimates.

## Workloads and metrics

All three checked-in workloads use the same short public sentence. The pressure
tests use 17 **distinct** voice embeddings, not aliases of one embedding.

| Workload | Warm-up / measured | Purpose |
|---|---:|---|
| `warm-one.jsonl` | 1 / 96 | One cached voice; measure any overhead introduced by LRU and accounting. |
| `voice-pressure.jsonl` | 16 / 96 | Controlled boundary diagnostic: 15 frequently used voices and two alternating occasional voices. Returning hot voices are visited in reverse order. This deliberately exposes upstream's clear-all behavior. |
| `round-robin-17.jsonl` | 17 / 102 | Adverse control: cycle through 17 voices with a 16-entry cache. LRU also thrashes here; expect little or no improvement. |

Do not present the pressure workload as a representative production distribution.
For production decisions, replay anonymized real voice frequencies, text lengths,
arrival rates and cancellation patterns. Include long text, new voices, mixed
languages/checkpoints and guidance as separate scenarios where supported.

- **Client TTFA:** dispatch to receipt of the first complete 16-bit PCM sample;
  response headers do not stop the timer. This is not time to first audible phoneme.
- **Aggregate throughput:** successful audio seconds divided by scenario wall time,
  including dispatch and client bookkeeping. Never divide by the sum of overlapping
  request latencies. RTF is also reported per successful request (lower is better).
- **Failures:** all outcomes remain in raw records and summary counts. Failed partial
  audio contributes zero to successful throughput. Latency percentiles explicitly
  cover completed requests; success counts and SLO counts must accompany them.
- **Streaming:** chunk arrival times and sizes are retained. Playback underruns and
  audio-device latency are not yet computed.
- **Load:** bounded, closed-loop concurrency. This does not establish sustainable
  open-loop QPS, queueing behavior or fleet capacity. Server admission above its
  configured limit returns 429, which is recorded rather than retried.
- **Quality:** a completed response means valid transport, PCM content type and
  nonempty whole samples. It does not prove correct speech or semantic completion.
  Waveform hashes allow sequential deterministic comparisons; WER/listening quality
  are not evaluated by the harness. Concurrent scheduling changes seed assignment,
  so matching request IDs alone is insufficient for concurrent hash comparisons.
- **Statistics:** nearest-rank percentiles. Around 100 requests are useful for a local
  diagnostic, not a reliable production p99 estimate. Use repeated trials and report
  all trials; do not claim significance from a small median difference.

Each run writes `requests.jsonl` and `report.json`. Outputs are never overwritten.
Warm-up failure stops the run with its raw record retained. `report.json` includes
workload/harness/binary hashes, cache snapshots, concurrency, timeouts and summaries.

## Validation

```sh
cargo fmt --all -- --check
cargo test --locked -p ptts -p ptts-openai-server --features ptts/hf
python3 -m unittest discover -s scripts/tests -v
PTTS_TEST_MODEL="$PWD/../models/phonon-31m-en" RAYON_NUM_THREADS=1 XN_KAI=0 \
  cargo test --locked --release -p ptts --features hf,accelerate,kai \
  --test voice_cache_checkpoint -- --ignored --test-threads=1
```

The checkpoint test disables KleidiAI for stable numerical comparison and checks
waveform error below `1e-5`, equal length, and finite samples. It downloads nothing.
Run the cache tests on each GPU backend before claiming validation there.

## Next optimization layers

1. **Measure request stages.** Extend the timing approach used in the existing vLLM/
   SGLang work: admission, normalization, voice prime, session setup, flow generation,
   Mimi decode, encoding and first PCM. Add cancellation, failure and memory telemetry
   with bounded labels. The current endpoint provides aggregate cache counters only;
   full OpenTelemetry tracing is not implemented.
2. **Reuse session capacity and workers.** Profile allocation, KV growth/copy and thread
   startup. Prototype a bounded session pool with exclusive ownership and complete
   per-request RNG, voice, condition, decoder and cancellation reset. Prove isolation
   before enabling reuse; the cache added here stores immutable prefixes only.
3. **Integrate upstream GPU work selectively.** Evaluate upstream flow/Mimi CUDA graphs,
   conditioning and stream changes on a fixed GPU and revision. Benchmark cold/warm
   TTFA, streaming gaps and concurrency before accepting a kernel speedup.
4. **Add scheduling only after profiling.** Continuous batching, admission queues and
   separate flow/Mimi scheduling need TTS-aware state and playback deadlines. Borrow
   vLLM/SGLang scheduling ideas; this Rust flow-matching/codec stack is not a drop-in
   Hugging Face causal-language-model backend for either server.

Keep changes that improve the target workload without audio regressions. CPU HTTP
results here cannot establish a win over upstream's published native M5 numbers,
which use different hardware and measurement boundaries.
