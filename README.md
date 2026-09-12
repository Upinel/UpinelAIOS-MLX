# M5 Pro Qwen Agent

A portable, one-command **Qwen3.8-27B uncensored agent endpoint** for Apple
Silicon Macs, tuned for maximum tokens/sec and long-context agent work.

Built and measured on an **M5 Pro / 20-core GPU / 64 GB**, serving
`itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` through
[MTPLX](https://github.com/youssofal/MTPLX):

```
~40-53 tok/s short-context decode   (2.6-3.4x over autoregressive)
128K context default, 262K capable
OpenAI-compatible API on your LAN
```

Clone it, edit `env.conf`, run `./start.sh`. Nothing else.

---

## Quick start

```bash
git clone <this repo> && cd M5Pro-QwenAgent

./install.sh          # scans your Mac, suggests settings, installs everything
./start.sh            # serves http://<your-lan-ip>:8000/v1
./status.sh           # live dashboard: CPU, GPU, memory, live decode rate
./stop.sh
```

`install.sh` starts by scanning your hardware — chip, GPU cores, unified memory,
free disk — and printing a table of suggested settings for **your** Mac next to
the current ones:

```
  SETTING                  CURRENT (env.conf)                 SUGGESTED
  ------------------------ ---------------------------------- ----------------------------------
  MODEL                    itrejomx/...Aggressive-MTPLX-4bit  itrejomx/...Aggressive-MTPLX-4bit
  CONTEXT_WINDOW           262144                             131072
  MEMORY_LIMIT_GB          8                                  48
  SESSION_BANK_GB          32                                 8

 ok Your current env.conf fits this machine (25 GB plan, 48 GB cap).
  Apply the suggested settings to env.conf? [Y/n]
```

It explains *why* each value is suggested, only rewrites the keys that need
changing (your comments survive), and tells you if the download will not fit on
disk. Use `--yes` to accept non-interactively, or `--scan-only` to just look.
If your current settings already fit, it says so and leaves them alone.

After that it installs MTPLX, downloads the model with resume, and finishes by
measuring your machine's optimal MTP depth — which is genuinely hardware-specific,
so do not skip that step (`--no-tune` skips it and falls back to depth 3, which
is safe but usually not optimal).

Total time is 15–30 minutes, most of it the download.

Point any OpenAI-compatible client at the URL `status.sh` prints. See
[docs/CLIENTS.md](docs/CLIENTS.md) for Open WebUI, Claude Code, Cline, Aider, and
the OpenAI Python SDK.

---

## Watching it work

`./status.sh` is a live dashboard, refreshed once a second:

```
  Qwen3.8-27B Agent                                            ● serving  up 02:14:33
  ────────────────────────────────────────────────────────────────────────────────
  itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit
  sustained · MTP d2 · think low · hist scoped · KV q8 · ctx 131072

    SYSTEM                               ACTIVITY
    CPU   ███░░░░░░░░░░░░░░░   15.9%     in-flight      1
    GPU   ██████████████████    100%     sessions       6
    ANE   ░░░░░░░░░░░░░░░░░░   n/a       decode live    22.4 t/s
    RAM   ████████████████░░   40.0G     last prompt    34500 tok
    SWAP  ████████████████░░   11.5G     last ttft      9.42s

    MEMORY  64 GB unified                RECENT REQUESTS   ctx / decode / tok / stop
    wired        28.9G  █████░░░░░░░       34500    11.3 t/s    153  tool_call
    app          9.40G  ██░░░░░░░░░░       33434    12.7 t/s    796  tool_call
    cached       21.4G  ████░░░░░░░░         547    22.4 t/s    192  length
    free         2.80G  █░░░░░░░░░░░         934    22.4 t/s     70  tool_call

    cpu ▁▂▃▅▇█▇▅▃▂▁        gpu ▁▂▅█▇▅▃▂▁        t/s ▃▅█▇▅▃
```

| Setting | What it shows | Source |
|---|---|---|
| CPU | total, plus user/sys split | `host_statistics` tick deltas — exact, no sudo |
| GPU | the GPU's own busy counter | IOKit `IOAccelerator`, no sudo |
| ANE | `n/a` unless `--power` | see the note below |
| RAM | wired / app / cached / compressed / free | `vm_stat`, conventional macOS breakdown |
| ACTIVITY | in-flight requests, sessions, live decode rate, last TTFT | MTPLX `/health`, `/admin/sessions`, `/metrics` |

`--once` prints a plain summary for logs and scripts, `--json` emits a
machine-readable snapshot, and `--interval N` slows the refresh.

**On the ANE.** Apple exposes the Neural Engine only through `powermetrics`,
which needs root. More importantly, **MLX does not use the ANE** — this
workload is GPU-only, so the ANE really is idle. Rather than fabricate a
number, the dashboard shows `n/a` and offers `--power` to read the real
figures via `sudo powermetrics` when you have passwordless sudo configured.

**On the GPU number.** `IOAccelerator` reports the *whole* GPU, including
WindowServer and browser compositing, which is why it can read high even when
the model is idle. It is a good signal for "is the machine busy", not a
per-process attribution. The `renderer` and `tiler` counters under the gauges
are usually more informative.

---

## Why these choices

Three decisions dominate everything else, and all three are backed by
measurement on this exact hardware rather than by convention.

### 1. MLX, not GGUF — because llama.cpp's MTP loses on Metal

llama.cpp added `--spec-type draft-mtp` and the GGUF you may already have embeds
the head. On Apple Silicon it does not pay off: Metal does not amortise a 3–5 row
verify batch the way CUDA does, so the draft passes cost more than they save.
The MLX conversion of this same fine-tune reports:

> "the same fine-tune as Q6 GGUF with the FastMTP sidecar in llama.cpp reached
> **~12.5 tok/s** on the same hardware" — against 30.9 tok/s for the MLX build.

We reproduced the shape of that on this Mac: llama.cpp + `--spec-type draft-mtp`
reached **20.7 tok/s**; the MLX stack reaches **53.4**. That is the whole
argument. Details in [docs/TUNING.md §1](docs/TUNING.md).

### 2. MTP depth 2 — a 3.4× win, and the optimum is not the maximum

The model ships a trained multi-token-prediction head. MTPLX uses it with exact
rejection sampling, so output stays distributionally identical at
`temperature=1.0` — it is real speculative decoding, not a greedy shortcut.

| Mode | tok/s | vs AR | acceptance |
|---|---:|---:|---|
| AR (MTP off) | 15.20 | 1.00× | — |
| Depth 1 | 42.98 | 2.83× | 98.2% |
| **Depth 2** | **51.34** | **3.38×** | 100%, 97.4% |
| Depth 3 | 39.94 | 2.63× | 98.4%, 90.6%, 81.0% |

> **On run-to-run variance:** three back-to-back identical short-context runs on
> this machine gave 34.3, 41.4, and 30.8 tok/s. The 51.3 above is the tuning
> harness's measurement on a quiet machine, and it is reproducible *there*, but
> a warm Mac with a browser open will sit lower. Every figure in this README is
> a measurement, not a spec sheet — re-run `./bench/bench.sh` for yours.

Depth 3 is *slower* than depth 2 — each extra draft token adds a full verify
pass while acceptance decays. This is why `install.sh` measures rather than
hardcoding: on your Mac the winner may be 1, 2, or 3.

### 3. 128K context, and it could be 262K

This is not a normal 27B. Only **16 of its 64 layers** keep a growing KV cache;
the other 48 are linear-attention (Gated DeltaNet) and carry a **fixed** ~150 MiB
state no matter how long the conversation gets.

So one token costs `16 layers × 2 (K,V) × 4 heads × 256 dim = 32,768 elements`:

| Context | f16 KV | q8 KV | q4 KV |
|---:|---:|---:|---:|
| 32,768 | 2.0 GB | 1.1 GB | 0.6 GB |
| 131,072 | 8.0 GB | 4.3 GB | 2.3 GB |
| 204,800 | 12.5 GB | 6.7 GB | 3.5 GB |
| 262,144 | 16.0 GB | 8.5 GB | 4.5 GB |

On 64 GB with 15 GB of weights, **262,144 context with q8 KV is ~24 GB total**.
The default is a conservative 131,072; 200K+ is a one-line change. And because
MTPLX sizes paged KV per request, a high `CONTEXT_WINDOW` costs nothing until
you actually use it.

### What long context actually costs you

Capacity and speed are different things. Measured here:

| prompt tokens | prefill | decode |
|---:|---:|---:|
| ~400 | 254 t/s | **52.3 t/s** |
| 8,000 | 402 t/s | **38.6 t/s** |
| 32,000 | 437 t/s | **22.5 t/s** |
| 93,000 | 217 t/s | **14.3 t/s** |

Decode falls off as context grows — each verify pass attends over everything —
and by 90k the MTP speedup has largely evaporated (the autoregressive baseline
is 15.2 t/s). Prefill is the other half: a **cold** 128k prompt takes minutes.

The practical consequence for agents is that **prompt caching matters more than
any decode tweak**. A conversation that grows turn by turn reuses its cached
prefix and is cheap. An agent that rebuilds its context every request pays full
prefill cost every time. `SSD_SESSION_CACHE="on"` and a stable prefix are worth
more than everything else in this file combined.

The session bank is not theory. Back-to-back requests during final verification
where the second prompt extended the first:

| request | prefill rate | TTFT |
|---|---:|---:|
| 547 tokens, cold | 138 t/s | 3.95 s |
| 8,035 tokens, sharing a prefix with it | **5,857 t/s** | **1.37 s** |
| 32,137 tokens, no shared prefix | 444 t/s | 72.4 s |

Same server, same context window: the shared-prefix request was ~40× faster to
first token than a cold prefill of comparable size. That is the difference
between an agent that is usable and one that is not.

> **One hard limit, measured:** a single ~131k-token prefill on this 64 GB Mac
> (with a normal desktop session running) fails with a Metal command-buffer
> allocation error, `kIOGPUCommandBufferCallbackErrorOutOfMemory`. This is a
> Metal allocator ceiling, not a RAM shortage — freeing memory will not fix it.
> Prompts up to ~93k prefill fine, and incremental context growth works.
> `PREFILL_CHUNK_TOKENS=2048` is the first thing to try; see
> [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## The one file you edit

`env.conf` is fully commented, and `./install.sh` offers to set it for your
machine. `MODEL` takes any MTPLX-format Hugging Face repo id, or one of five
tested aliases:

| alias | size | repo | notes |
|---|---:|---|---|
| `4bit` | 15 GB | `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` | **default**, fastest, uncensored |
| `6bit` | 23 GB | `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-6bit` | same fine-tune, higher fidelity |
| `4bit-opus` | 17 GB | `barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit` | different fine-tune, agent-focused |
| `6bit-opus` | 24 GB | `barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-6bit` | as above, higher fidelity |
| `official` | 20 GB | `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` | **aligned, not uncensored**; only one with vision |

The settings you are most likely to touch:

```conf
MODEL="itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit"
CONTEXT_WINDOW=131072     # 204800 or 262144 if you left yourself headroom
THINKING="low"            # off | low | medium | high
MTP_DEPTH="auto"          # auto | 1 | 2 | 3 | 0 (off)
KV_QUANT="q8"             # off | q8 | q4
PROFILE="sustained"       # sustained (long ctx) | turbo (short ctx only)
PREFILL_CHUNK_TOKENS=2048 # lower if a long prompt fails with a Metal OOM
MEMORY_LIMIT_GB=48        # raise on a bigger Mac, lower on 32 GB
SESSION_BANK_GB=8         # prefix cache; 0 = let MTPLX auto-size (it over-allocates)
HOST="0.0.0.0"            # LAN. Use 127.0.0.1 for this Mac only
PORT=8000
```

`install.sh` and `start.sh` print a memory budget line before loading, so a
configuration that cannot fit fails immediately with an explanation instead of
after a two-minute load.

---

## Making it faster

In rough order of payoff for agent work:

1. **`THINKING="off"`** for tool-calling loops. Thinking tokens are generated at
   the same speed as answer tokens and are pure latency for an agent that just
   needs to call a tool. This is usually a 2–5× wall-clock win, and it is not a
   decode-speed change at all — which is why it is easy to overlook.
2. **Keep prefixes stable.** The server caches committed session prefixes and
   spills them to SSD. An agent that injects a timestamp or reorders its system
   prompt every turn invalidates that cache and re-prefills the whole context.
3. **Re-tune MTP depth** (`./bench/bench.sh --tune`) — worth a few percent.
4. **`KV_QUANT="off"`** if you have RAM to spare and want the compiled-verify
   fast path back.
5. **`FAN_MODE="smart"`** for long runs; a laptop will thermally throttle.

---

## Benchmarking

```bash
./bench/bench.sh --quick      # ~30 s sanity check
./bench/bench.sh              # 512 / 8k / 32k / your configured context
./bench/bench.sh --tune       # re-measure AR vs depth 1/2/3
```

The harness times a streaming request client-side and reports TTFT, prefill
rate, and steady-state decode, which is what a client actually feels. Samples
under 32 generated tokens are flagged as noise.

---

## Portability notes

- **Apple Silicon only.** MTPLX is MLX-native; there is no x86 path.
- **32 GB Macs:** set `MEMORY_LIMIT_GB=24` and lower `CONTEXT_WINDOW` to 65536.
  `install.sh` warns when your plan does not fit.
- **96 / 128 GB Macs:** raise `MEMORY_LIMIT_GB` and `CONTEXT_WINDOW`; consider
  `MODEL="6bit"` for better fidelity, which still fits comfortably.
- **`WIRED_LIMIT_GB`** defaults to `0`, leaving macOS to manage the GPU wired
  ceiling (~75% of RAM). Do not raise it near physical RAM: it makes jetsam kills
  more likely, and a hard-killed process can leak wired pages until reboot.
- **Always stop with `./stop.sh`.** It sends `SIGTERM` and waits; the
  `--force` path sends `SIGKILL` and is a last resort, for exactly that reason.

---

## Repo layout

```
env.conf              the only file you edit
install.sh            deps + model download + MTP depth tune
start.sh / stop.sh    server lifecycle
status.sh             live dashboard (--once / --json for scripts)
lib/dashboard.py      metrics collection and TUI rendering
lib/common.sh         config loading, memory math, health helpers
lib/fetch-model.sh    resumable HF downloader, no pip dependency
bench/bench.sh        sweep / tune entrypoint
bench/bench.py        streaming benchmark harness
bench/verify-tools.sh tool-calling diagnostic — run this if an agent misbehaves
lib/preflight.sh      hardware scan and per-machine configuration suggestions
docs/TUNING.md        the full performance story and every knob
docs/TROUBLESHOOTING.md  what to do when something does not work
docs/CLIENTS.md       connecting Open WebUI, Claude Code, Cline, Aider, SDKs
docs/RESEARCH.md      upstream research: MTP on Metal, memory math, sources
```

Model weights live in `models/` and are **not** committed — `install.sh` fetches
them. Set `MODEL_SOURCE_DIR=/path/to/existing/copy` to hardlink a copy you
already have instead of downloading.

---

## Credits and licences

This bundle is glue: the hard parts belong to other people.

- **[MTPLX](https://github.com/youssofal/MTPLX)** by Youssof Altoukhi — Apache-2.0,
  with a **mandatory in-product attribution requirement**. If you ship something
  built on this, it must display *"Powered by MTPLX — https://github.com/youssofal/MTPLX"*
  somewhere a user can see it.
- **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — Apache-2.0.
- **HauhauCS** — the Aggressive uncensored fine-tune.
- **itrejomx** — the MLX 4-bit MTP conversion used by default.

Model weights carry their own upstream licences. Uncensored fine-tunes are
uncensored: you are responsible for how you use the endpoint, and for the fact
that you just exposed it to your LAN.
