# Tuning guide

Everything here is measured on an **Apple M5 Pro, 20-core GPU, 64 GB**, macOS 27.0,
with `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit`.
Your numbers will differ; re-run `./bench/bench.sh` to get your own.

---

## 1. Why this stack

### The model is not a normal 27B

`Qwen3.8-27B` is architecture `qwen3_5`. The important part for performance:

| | |
|---|---|
| Layers | 64 |
| Full-attention layers | **16** (every 4th: 3, 7, 11, … 63) |
| Linear-attention (Gated DeltaNet) layers | **48** |
| KV heads × head dim | 4 × 256 |
| Built-in MTP head | 1 layer, ships with the model |

Only 16 of 64 layers keep a growing KV cache. The other 48 are recurrent: they
carry a **fixed** ~150 MiB state regardless of context length.

That is the single most important fact in this document. It makes 200k+ context
cheap on a 64 GB Mac, where a dense 27B would have run out of memory long before.

### MTP is what makes it fast

The model ships a **multi-token-prediction head**: a small trained layer that
guesses the next *several* tokens, which the main model then verifies in one
batched pass. MTPLX implements this with exact rejection sampling (Leviathan &
Chen), so the output distribution is unchanged — it is not a greedy shortcut and
it stays correct at `temperature=1.0`.

Measured on this machine:

| Mode | tok/s | vs AR | draft acceptance |
|---|---:|---:|---|
| Autoregressive (MTP off) | 15.20 | 1.00× | — |
| MTP depth 1 | 42.98 | 2.83× | 98.2% |
| **MTP depth 2** | **51.34** | **3.38×** | 100.0%, 97.4% |
| MTP depth 3 | 39.94 | 2.63× | 98.4%, 90.6%, 81.0% |

Depth 2 wins here. **Depth is hardware-specific** — run
`./bench/bench.sh --tune` on your own Mac. Higher depth is not better: each extra
draft token adds a full verify pass, and acceptance decays.

### Why MLX and not GGUF/llama.cpp

llama.cpp now supports this model's MTP head through `--spec-type draft-mtp`,
and the GGUF you may already have embeds the head. On Apple Silicon, though, the
Metal kernels do not amortise a 3–5 row verify batch the way CUDA does, and the
published result for this exact fine-tune is:

> "the same fine-tune as Q6 GGUF with the FastMTP sidecar in llama.cpp reached
> **~12.5 tok/s** on the same hardware" — where the MLX build reached 30.9 tok/s.

Our own llama.cpp smoke test on this Mac reached 20.7 tok/s at 512 context with
MTP enabled. MLX reaches 53. Measured, not theoretical: **MLX wins by ~2.5×.**

Keep GGUF for a different reason if you want it: `mmproj` vision support is more
mature there, and llama.cpp is easier to embed. For raw agent throughput, MLX.

---

## 1a. Applying a change

`env.conf` is read when the server starts and passed to the runtime as
command-line arguments, so **an edit does nothing until you restart**:

```bash
./restart.sh            # stops, waits for the port and GPU to settle, starts
./restart.sh --print    # show what would change, do nothing
```

`restart.sh` prints a paired diff of the *effective* settings, so you can see
exactly what you are switching between without diffing the file yourself.

## 2. The knobs, in order of how much they matter

### MTP depth — biggest single lever (3.4× here)

```conf
MTP_DEPTH="auto"     # uses run/tuning.json from ./install.sh
MTP_DEPTH="2"        # pin it
MTP_DEPTH="0"        # disable: imports --no-mtp, ~3x slower
```

### Context window — costs KV, not much else

Because only 16 layers cache KV, one token costs
`16 × 2 × 4 heads × 256 dim = 32,768 elements`:

| KV type | bytes/element | per token | 131,072 | 204,800 | 262,144 |
|---|---|---:|---:|---:|---:|
| f16 | 2.0 | 64 KB | 8.0 GB | 12.5 GB | 16.0 GB |
| q8 | ~1.06 | 34 KB | 4.3 GB | 6.7 GB | 8.5 GB |
| q4 | ~0.56 | 18 KB | 2.3 GB | 3.5 GB | 4.5 GB |

On 64 GB with the 4-bit model (15 GB): **262,144 context with q8 KV = ~24 GB
total.** 200k is not a stretch goal on this machine; it is comfortable.

```conf
CONTEXT_WINDOW=131072    # 128K, safe default
CONTEXT_WINDOW=204800    # 200K
CONTEXT_WINDOW=262144    # the model's maximum
```

> **Important:** `CONTEXT_WINDOW` is a *capacity*, not a reservation for a
> single request. MTPLX sizes paged KV per request, so a short chat does not
> pay for 128K of cache. Setting it high costs nothing until you use it.

### KV quantization — the memory/speed tradeoff

```conf
KV_QUANT="off"   # fastest decode, biggest KV
KV_QUANT="q8"    # default. ~2x smaller KV, near-free decode
KV_QUANT="q4"    # smallest, but decode re-dequantizes every step
```

`q8` is the right default. It is worth knowing what it costs: with a quantized
paged KV cache, MTPLX's **compiled-verify prewarm is skipped** — you will see
`"skipped": ["quantized_paged_kv"]` at startup. The q8 path decodes through a
fused inline-dequant kernel instead, which is close but not identical.

If you are chasing the last few percent and have RAM for f16 KV, set
`KV_QUANT="off"` and measure with `./bench/bench.sh`.

### Profile

```conf
PROFILE="turbo"      # default. NAX verify kernels + context-routed compiled verify
PROFILE="sustained"  # long-context path without the NAX kernels
```

`turbo` is a superset of `sustained` for this flagship model. Compiled verify is
exact up to 32,768 tokens and then falls back to eager automatically, so it is
safe at any context. Switch to `sustained` only if turbo misbehaves.

### Batching preset — matters when several agents share the box

```conf
BATCHING_PRESET="agent"    # default: coding-agent traffic, long shared prefixes
BATCHING_PRESET="solo"     # one client, lowest latency
BATCHING_PRESET="throughput"  # many concurrent clients
MAX_CONCURRENT=2           # in-flight requests
```

Single-client latency is best at `MAX_CONCURRENT=1`. Multiple agents doing
independent work want 2–4, at some cost each.

### Thinking effort

```conf
THINKING="low"      # default
THINKING="off"      # tool-calling agents: fastest, no thinking block
THINKING="medium"
THINKING="high"     # maps to the model's 'xhigh' template setting
```

This changes **how many tokens the model emits**, not the per-token speed — but
it is usually the largest wall-clock lever of all. Thinking tokens are generated
at the same rate as answer tokens and are pure overhead for a tool-calling loop.
The model's own chat template supports `low | medium | xhigh`; `high` is mapped
to `xhigh`.

### Thinking — five levels, and a live toggle

```conf
THINKING="minimal"     # off | minimal | low | medium | high
```

The model's chat template only distinguishes `low / medium / xhigh`, and the
`low` instruction is a *soft nudge* — the model routinely ignores it. Verified
by reading the template: only `xhigh` and `low` carry any instruction text at
all, and `medium` sends an empty string.

What actually bounds thinking is MTPLX's **thinking guard**, a token budget that
closes the thinking block and forbids re-entry. UpinelAIOS turns it on; MTPLX
ships it off by default. Measured, with tools present:

```json
{"enabled": true, "budget_tokens": 128, "think_tokens": 140,
 "engaged": "budget", "forced_emitted": 13,
 "reentry_banned_positions": 54, "novelty_close": true}
```

`engaged: "budget"` is the guard doing the work — the model did not stop on its
own, it was stopped. `reentry_banned_positions: 54` is it being kept out of a
second thinking block afterwards.

| level | budget | what it means |
|---|---:|---|
| `off` | — | no thinking at all |
| `minimal` | ~128 tok | answer first, think only if forced. **Default.** |
| `low` | ~512 tok | brief thinking, hard-capped |
| `medium` | ~2048 tok | more room, still capped |
| `high` | none | the model's own `xhigh` — it decides |

Only `off` removes thinking entirely; the rest are *bounded*, which is the
difference between asking the model to be brief and making it be brief.

Two caveats worth knowing:

- **The guard applies to requests that carry tools**, which is the agent case
  and the reason it is useful here. A plain chat request with no tools is not
  guarded.
- **`medium` is not a middle setting** in the template sense — it differs from
  `low` only in budget, not in what the model is asked to do.

Override the per-level default with `THINKING_BUDGET_TOKENS`, or turn off the
early stop with `THINKING_NOVELTY_CLOSE=0`.

#### Changing it without a restart

```bash
./status.sh --thinking            # show the current setting
./status.sh --thinking off        # switch live
./status.sh --thinking minimal
```

This uses MTPLX's live settings endpoint. The model stays loaded — only the
decode policy changes — so it is instant and does not disturb a call in flight.
It is **not** persisted: set `THINKING` in `env.conf` to survive a restart.
A client can also override per request:

```json
{"chat_template_kwargs": {"enable_thinking": false,
                          "reasoning_effort": "low"}}
```

### Reasoning history — the knob nobody expects to matter### Reasoning history — the knob nobody expects to matter

```conf
PRESERVE_THINKING="scoped"   # default
```

MTPLX defaults to `auto`, which for this model resolves to **preserving all
thinking forever**. Every turn then re-sends the model's complete reasoning
transcript. Measured on a real agent session on this machine:

| prompt tokens | reasoning history in prompt | decode |
|---:|---:|---:|
| 934 | 0 chars | 22.4 t/s |
| 33,434 | 31,897 chars | 12.7 t/s |
| 34,500 | 34,781 chars | **11.3 t/s** |

Roughly 35,000 characters — about 10,000 tokens — of *finished* reasoning were
being carried on every request, halving decode speed. That is pure waste: the
model already thought those thoughts and committed the conclusions.

`scoped` keeps reasoning only inside the active agent round, which is the
contract the model was trained on, so answer quality is preserved while the
history stops growing without bound. Set it to `on` if you explicitly want the
full transcript preserved (some workflows do), or `off` to strip it entirely.

This one setting is worth more than most decode-side tuning, and it costs
nothing to enable.

### Prompt caching — the sleeper win for agents

```conf
SSD_SESSION_CACHE="on"    # default
```

MTPLX keeps committed session prefixes (`warm-prefix session bank`) and spills
them to SSD. An agent that sends a 100k-token conversation, then sends it again
with one more turn, normally re-prefills all 100k. With the session bank it
reuses the cached blocks instead. On this machine a cold 131k prefill takes
minutes; a warm one takes seconds.

This is why `SSD_SESSION_CACHE="on"` is worth more to an agent workload than
almost any decode-side tweak.

### Thermals

```conf
FAN_MODE="default"   # let macOS manage. Recommended.
FAN_MODE="smart"     # boost only while generating (one-time sudo unlock)
FAN_MODE="max"       # pinned high. Loud. Best sustained clocks.
```

A long agent run will thermally throttle a MacBook. `smart` or `max` recover
some of that. `--tune` results in this repo were measured with **fans on auto**,
so a pinned-fan run may score higher.

---

## 3. Memory

MTPLX defaults to a hard allocator cap of **75% of physical RAM** and a wired
cap of **60%**. On 64 GB that is 48 GB / 38.4 GB, which is the right shape. The
bundle pins the allocator cap explicitly from `MEMORY_LIMIT_GB` so the behaviour
is identical on a 32 GB or a 128 GB Mac.

Two things worth knowing before you raise limits:

1. **Wired memory is not free to take back.** A process holding a large wired
   MLX allocation that is `SIGKILL`ed can leak those pages at the kernel level
   until the machine reboots. This is why `./stop.sh` sends `SIGTERM` and waits
   30 seconds before escalating. Always stop gracefully.
2. **Setting `iogpu.wired_limit_mb` near physical RAM makes jetsam kills more
   likely, not less.** If macOS cannot reclaim, it kills the process instead of
   swapping. `WIRED_LIMIT_GB=0` (leave macOS alone) is the correct default.

A safe plan on 64 GB: 15 GB weights + 4–8 GB KV + ~6 GB activations ≈ **25–29 GB**,
against a 48 GB cap. Comfortable.

---

## 4. Reproducing the measurements

```bash
./bench/bench.sh --quick                 # ~30 s sanity check
./bench/bench.sh                         # 512 / 8k / 32k / your context
./bench/bench.sh --tune                  # re-measure AR vs D1/D2/D3
./bench/bench.sh --contexts 512,262144 --max-tokens 256 --repeats 3
```

The harness times the client side of a streaming request, so what it reports is
what a client actually experiences:

```
TTFT   = time to first streamed token          (dominated by prefill)
decode = (completion_tokens - 1) / (last_chunk - first_chunk)
```

A run that generates fewer than 32 tokens is flagged as noise — MTP has a
startup cost per request, and a 5-token sample measures that, not throughput.

---

## 5. Measured results on the reference machine

**Apple M5 Pro, 20-core GPU, 64 GB, macOS 27.0.**
Model: `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` (15 GB).
MTP depth 2, KV `q8`, `THINKING="low"`, `max_tokens=192`, one request per row,
all measured client-side with `bench/bench.py`.

| prompt tokens | prefill | decode | profile |
|---:|---:|---:|---|
| 391 | 254 t/s | **52.3 t/s** | turbo |
| 8,035 | 402 t/s | **38.6 t/s** | sustained |
| 32,137 | 437 t/s | **22.5 t/s** | sustained |
| 93,406 | 217 t/s | **14.3 t/s** | turbo (q8) |

**Run-to-run variance is real and large.** Three consecutive identical
short-context runs measured 34.3 / 41.4 / 30.8 tok/s, while the tuning harness on
a quiet machine measures 51.3. Draft acceptance depends on what the model is
writing, and a warm chassis throttles. Treat any single number — including the
ones below — as a sample, not a constant. Use `--repeats 3` for a median.

**Decode falls off with context.** This is the honest headline, and it is not a
tuning failure — it is what the architecture does:

| context | decode | as a share of short-context speed |
|---:|---:|---:|
| ~0.4k | 52.3 t/s | 100% |
| ~8k | 38.6 t/s | 74% |
| ~32k | 22.5 t/s | 43% |
| ~93k | 14.3 t/s | 27% |

At 93k the MTP speedup has largely evaporated: the autoregressive baseline is
15.2 t/s, so speculative decoding is buying almost nothing there. The cause is
that each verify pass has to attend over the whole context, and the draft
acceptance rate does not improve to compensate.

**Prefill is the other half of the story.** It holds ~400 t/s to about 32k, then
halves. A cold 131k prefill takes several minutes. This is why prompt caching is
the highest-value feature in the stack for agents: the *second* turn with the
same prefix is nearly free.

### Practical guidance from these numbers

- **Keep agent contexts under ~32k where you can.** That is where the speed is.
  RAG and targeted file reads beat dumping a whole repo into the window.
- **A cold 128k turn costs minutes.** Structure an agent so the expensive
  prefix is built once and then reused, rather than rebuilt per turn.
- **`THINKING="off"`** matters more at long context, not less: thinking tokens
  are generated at the depressed long-context decode rate.

### Memory behaviour at long context — read this before setting CONTEXT_WINDOW=262144

Long-context prefill on a 64 GB Mac that is *also* running a desktop session is
the one configuration that genuinely struggles. Observed during a cold 131k
prefill with a browser and editor open:

```
Pages wired down:  ~50 GB       (macOS baseline is ~6 GB)
Pages free:        ~150 MB
swap used:         ~12 GB
prefill:           stalled for >12 minutes with 0 tokens emitted
```

The inference process had ballooned to roughly 50 GB of wired memory. macOS
cannot reclaim wired pages, so it swapped, and prefill throughput collapsed.

Three levers, in the order worth trying:

1. **`SESSION_BANK_GB`** — defaults to `8` here. MTPLX otherwise auto-sizes the
   prefix cache to *half the post-model RAM surplus*, which was **16.6 GB** on
   this Mac. Capping it at 8 GB returned ~8 GB to the system.
2. **`PREFILL_CHUNK_TOKENS`** — set a value like `8192` to prefill in chunks
   instead of one giant batch, which lowers peak transient memory.
3. **`MEMORY_LIMIT_GB`** — lower it to 40 or 36 if you keep a heavy desktop
   session running. The weights plus a 128k KV cache need about 23 GB, so 36 GB
   still leaves room while giving macOS more to work with.

If you want 200k+ context to be genuinely comfortable rather than merely
possible, run the endpoint on a Mac that is not also your desktop, or move to
96 GB. The 4-bit weights plus 262k of q8 KV is ~24 GB; the headroom problem is
entirely about everything *else* on the machine.
