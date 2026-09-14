<!--
  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  dev@upinel.com  ·  upinel.com
  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.

  Upinel Personal Free License: free for personal use, and free for creators
  (YouTubers, KOLs) to make content with - just email dev@upinel.com to say so.
  Other commercial use needs written permission. Derivatives must credit the
  author. Covers this project's own code only. See LICENSE.

  "Make it work, make it right, make it fast - then measure it, because
   the third one is only a claim until the numbers agree."
-->
<h1 align="center">UpinelAIOS-MLX</h1>

<p align="center">
  <b>Up to 79 t/s decode — uncensored, 100% local, on your own Mac.</b><br>
  One-click AI agent server OS for Apple Silicon · MLX / MTPLX<br>
  <sub>Extreme performance optimisation for AI agent workflows.</sub><br>
  <sub>One command to install. One command to serve. Your data never leaves the LAN.</sub>
</p>

<p align="center">
  <sub>Built by <b>Nova Upinel Chow</b>, MSc, LLM, BBA, MENSA &nbsp;·&nbsp;
  <a href="mailto:upinel@me.com">upinel@me.com</a> &nbsp;·&nbsp;
  <a href="mailto:dev@upinel.com">dev@upinel.com</a> &nbsp;·&nbsp;
  <a href="https://upinel.com">upinel.com</a></sub><br>
  <sub>Free for personal use, and free for creators — YouTubers, KOLs, streamers
  and bloggers, and <b>you keep the money you make</b>: just email dev@upinel.com
  to say so, no reply needed. Other commercial use by written permission.
  Derivatives must credit the author. See <a href="LICENSE">LICENSE</a>.</sub>
</p>

---

**Just want it running?** On an Apple Silicon Mac with 32 GB or more:

```bash
git clone https://github.com/upinel/UpinelAIOS-MLX && cd UpinelAIOS-MLX
./install.sh      # scans your Mac, suggests settings, installs everything
./start.sh        # serves http://<your-lan-ip>:8000/v1
./chat.sh         # talk to it, right here in the terminal
```

That is the whole quick start. Everything below is detail you can come back to —
[Requirements](#requirements) if your Mac is small, [Configure it](#configure-it)
to change model or context, [Benchmarks](#benchmarks) for the measured numbers.

---

> **Sister project: [UpinelAIOS-GGUF](https://github.com/Upinel/UpinelAIOS-GGUF)** —
> the same one-click agent server built on **llama.cpp** rather than MLX.
> That one serves both **Gemma 4** and **Qwen 3.8** from GGUF, and is the only
> one of the two that can run an uncensored Gemma 4 at all (MTPLX needs a
> target/assistant pair, and the only pair in existence is built from Google's
> aligned models). This one is the faster choice for Qwen specifically, because
> MTPLX's MTP speculative decoding beats llama.cpp's MTP on Metal.
>
> **Rule of thumb:** serving Qwen → this project. Serving Gemma 4, or anything
> else in GGUF → UpinelAIOS-GGUF.

A portable, one-command **uncensored Qwen agent endpoint** for Apple Silicon
Macs, tuned for maximum tokens/sec and long-context agent work.

The default is a **mixture-of-experts** model — 35B total, only ~3B active per
token — because that is what makes agents fast, and agent work is what this
serves. The dense 27B is one command away if you would rather have it.

Built and measured on an **M5 Pro / 20-core GPU / 64 GB**, serving
`hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16` through
[MTPLX](https://github.com/youssofal/MTPLX):

```
74-88 tok/s decode                  (MTP speculative decoding on, measured)
128K context default, 262K capable
OpenAI-compatible API on your LAN
live dashboard for CPU, GPU, memory and token throughput
```

Swapping to the dense 27B, or any other model in the table below:

```bash
./model_download.sh                              # what is available
./model_download.sh --switch 4bit                # fetch it, switch, restart
```

Clone it, run `./install.sh`, run `./start.sh`. Nothing else.

**What "Server OS" means here, precisely:** UpinelAIOS-MLX is not an operating
system. It is a self-contained serving stack — runtime, model, configuration,
lifecycle, telemetry and diagnostics — that turns a Mac into an agent server
appliance with a single command. It installs nothing into your system beyond
Homebrew's `mtplx` and the model weights, and `./stop.sh` leaves the machine
as it found it.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
  - [Chatting from the terminal](#chatting-from-the-terminal)
- [Configure it](#configure-it)
  - [The one file you edit](#the-one-file-you-edit)
    - [Models: uncensored only](#models-uncensored-only)
  - [Changing settings](#changing-settings)
- [Going deeper](#going-deeper)
  - [Why these choices](#why-these-choices)
    - [1. MLX, not GGUF: because llama.cpp's MTP loses on Metal](#1-mlx-not-gguf-because-llamacpps-mtp-loses-on-metal)
    - [2. MTP depth 2: a 3.4× win, and the optimum is not the maximum](#2-mtp-depth-2-a-34-win-and-the-optimum-is-not-the-maximum)
    - [3. 128K context, and it could be 262K](#3-128k-context-and-it-could-be-262k)
    - [What long context actually costs you](#what-long-context-actually-costs-you)
  - [Portability notes](#portability-notes)
- [Benchmarks](#benchmarks)
  - [Measured throughput](#measured-throughput)
    - [The same models on both runtimes](#the-same-models-on-both-runtimes)
    - [Every run varies, so quote the range](#every-run-varies-so-quote-the-range)
    - [Why the MoE is the default](#why-the-moe-is-the-default)
  - [Making it faster](#making-it-faster)
  - [Benchmarking](#benchmarking)
- [Running it day to day](#running-it-day-to-day)
  - [Updating the code](#updating-the-code)
  - [Watching it work](#watching-it-work)
    - [Keyboard control](#keyboard-control)
    - [Two limits, stated rather than hidden](#two-limits-stated-rather-than-hidden)
- [Repo layout](#repo-layout)
- [Credits and licences](#credits-and-licences)

## Requirements

|   |   |
|---|---|
| **Minimum** | Apple Silicon Mac, **32 GB** unified memory, macOS 14+, ~25 GB free disk |
| **Recommended** | **64 GB** or more, for 128K context with headroom for a normal desktop |
| **Chip** | Any M-series. Speed scales with memory bandwidth, not core count. |

32 GB is genuinely usable — a 4-bit 27B is 15 GB, and the model only caches KV
on 16 of its 64 layers, so context is cheap. What you give up is context length
and headroom: set `MEMORY_LIMIT_GB=24` and `CONTEXT_WINDOW=65536` there, and
expect the machine to be busy while it serves. `install.sh` scans your hardware
and proposes exactly these numbers for you.

64 GB is the comfortable target: 15 GB of weights, 4–8 GB of KV at 128K, and
plenty left for macOS, your editor and a browser. That is what this was built
and measured on.

Disk: 5–22 GB per model, depending which you pick. `install.sh` downloads only
the model you have selected, and lets you pick a different one by number if the
suggested model is not what you want.

## Quick start

```bash
git clone https://github.com/upinel/UpinelAIOS-MLX && cd UpinelAIOS-MLX

./install.sh          # scans your Mac, suggests settings, installs everything
./start.sh            # serves http://<your-lan-ip>:8000/v1
./chat.sh             # talk to it right here in the terminal
./status.sh           # live dashboard: CPU, GPU, memory, live decode rate
./restart.sh          # apply an env.conf change
./model_download.sh   # download another model
./stop.sh
```

### Chatting from the terminal

`./chat.sh` is a streaming chat client for the server you already have running.
Nothing new is served and nothing is reconfigured — it just connects back to it.

```
you ▸ Explain what a mixture-of-experts model is, briefly.
ai  ▸ A mixture-of-experts model splits its feed-forward layers into many
      expert subnetworks and routes each token to only a few of them...
        48.2 t/s   prefill 610 t/s   1.1s
```

Replies stream as they are generated, and every turn ends with its own line of
telemetry so you can see what the server is doing.

```bash
./chat.sh                      # start chatting
./chat.sh --system "You are a terse assistant."
./chat.sh --temp 0.2           # tighter sampling
./chat.sh --no-stream          # wait for whole replies
```

In-session commands: `/help`, `/reset`, `/system <text>`, `/thinking`,
`/temp <0..2>`, `/stats`, `/save <file>`, `/clear`, `/exit`. `Ctrl-C` stops a
reply without quitting; `Ctrl-D` leaves.

**Pasting works, and a paste is one message.** Paste a stack trace, a diff, or a
whole file and it is sent as a single turn rather than one turn per line, with
indentation intact. Lines inside a paste are never treated as commands, so
pasting a script that contains `/exit` sends it to the model instead of quitting
on you. A paste that is a single line still works as a command — `/help` on its
own runs, `/help` inside a pasted block is text. The client says which it did:

```
you ▸ <12 lines pasted here>
  pasted 12 lines, 431 chars - sending as one message
ai  ▸ ...
```

`chat.sh` is the same client as the sister GGUF project's, and `/thinking`
behaves slightly differently in each: llama.cpp takes thinking as a per-request
`chat_template_kwargs`, so there `/thinking on` takes effect on the next turn,
while MTPLX takes it as a live server setting, so this build points you at
`./status.sh --thinking` instead of pretending to change it.

`install.sh` starts by scanning your hardware — chip, GPU cores, unified memory,
free disk — and printing a table of suggested settings for **your** Mac next to
the current ones:

```
  SETTING                  CURRENT (env.conf)                 SUGGESTED
  ------------------------ ---------------------------------- ----------------------------------
  MODEL                    hawhyhb/...Heretic-MTPLX-4bit-FP16  hawhyhb/...Heretic-MTPLX-4bit-FP16
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

**If you have more than one model downloaded, `start.sh` and `restart.sh` ask
which one to serve.** Press a number, or Enter for the default; with no answer
within five seconds the default from `env.conf` is used, so nothing ever waits
on you:

```
  Models on disk   2 downloaded - pick one to serve now

   1  moe           22 GB  <- default
   2  4bit          15 GB

  Number [1-2], or Enter for the default. Auto-selects in 5s:
```

It is a one-run choice and is not written back — `MODEL` in `env.conf` is still
the default. Pass `--model 4bit` to skip the question entirely, and note that it
is skipped automatically whenever there is no terminal to ask on (a pipe, CI,
`nohup`, launchd) or only one model is downloaded.

Only complete downloads are offered. MLX packs name every shard in
`model.safetensors.index.json`, and a directory missing any of them is not
listed, so the picker cannot hand you a half-downloaded model.

---

---

## Configure it

### The one file you edit

`env.conf` is fully commented, and `./install.sh` offers to set it for your
machine.

#### Models: uncensored only

**Every model UpinelAIOS-MLX ships or suggests is an uncensored fine-tune.** That is
a product rule, not a coincidence: this endpoint exists so you can run a model
that will not refuse you, and there is no point being fast at something that
won't answer.

You do not have to choose from this table by hand. `install.sh` scans your Mac
and proposes a model and settings for it; if you would rather not have that one,
answer `n` at the prompt and it prints a numbered list of everything below, each
row marked with a verdict for *your* memory:

```
    #  ALIAS      SIZE   VERDICT              NOTE
    1  4bit       15 GB  fits comfortably     dense 27B, ~34.7 tok/s - the best quality
    ...
    6  moe        22 GB  RECOMMENDED          35B MoE, ~3B active - the fastest here (~79.4 tok/s)
  Model number:
```

`RECOMMENDED` / `fits comfortably` / `tight - expect paging` / `will not fit` are
computed from your unified memory, the model's published size, and that model's
own KV cost per token — the 27B is a hybrid that caches KV on only 16 of its 64
layers, while the dense 9B caches on all 32, so the 9B costs *twice* as much per
token despite being a quarter of the size. Verdicts are per-model, not per-size.
Enter a number to use it, or press Enter to keep the config you already have.
Choosing something that will not fit is allowed; it warns, then does what you
asked.

| alias | size | repo |
|---|---:|---|
| **`moe`** | 22 GB | `hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16` — **DEFAULT**. 35B MoE, ~3B active, 79.4 t/s. Fastest and the best agent balance. |
| `4bit` | 15 GB | `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` — dense 27B at 34.7 t/s. The quality pick. |
| `6bit` | 22 GB | `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-6bit` |
| `27b-3bit` | 13 GB | `barozp/Qwen3.8-27B-Uncensored-MTPLX-3bit` — friendliest to 32 GB |
| `27b-4bit` | 16 GB | `barozp/Qwen3.8-27B-Uncensored-MTPLX-4bit` |
| `9b` | 5 GB | `Foresee/Qwen3.8-9B-heretic-uncensored-4bit-MTPLX` — 65.1 t/s. Only when memory is tight. |

Each is an MLX pack with a verified MTP head, which is what makes the
speculative decoding work. The 27B entries are HauhauCS Aggressive
fine-tunes; the MoE is a Heretic abliteration.

**Why the MoE leads.** A dense model reads every weight on every token, so a
dense 27B reads ~15 GB per token no matter how little of it the answer needs.
The MoE activates ~3B of its 35B, so it reads roughly a fifth as much and goes
proportionally faster — which is precisely the trade agents want, since an
agent emits many small tool calls rather than long essays.

Both are uncensored, both are one command apart. Measured here on an M5 Pro —
the summary is:

| model | decode | pick it when |
|---|---:|---|
| **`moe`** &nbsp;Qwen 3.6 35B-A3B | **79.4 t/s** | Almost always. Fastest and, for agent work, the best balance. |
| `4bit` &nbsp;Qwen 3.8 27B dense | 34.7 t/s | You want maximum quality and will pay ~2.3× the time for it. |

The MoE activates ~3B of its 35B, so it reads far fewer weights per token and
goes proportionally faster — exactly the trade agents want, since an agent emits
many small tool calls rather than long essays. Full numbers, including the same
models on llama.cpp for comparison, are in [Benchmarks](#benchmarks).
MTP depth 1 is optimal for the MoE and depth 2 for the dense model; both are
recorded per model, so switching does not make you re-tune by hand.

> **A naming trap worth knowing.** Packs literally called
> `...-MTPLX-Optimized-Speed` also exist in uncensored variants, and they do
> **not** run on MTPLX. They are built for a different runtime (`lightning-mlx`)
> with a different MTP tensor layout, and MTPLX refuses them:
>
> ```
> tier              architecture-compatible-but-unverified
> can_run           False
> runtime_compat    needs-grafting
> message           ... local MTP artifact inspection did not pass;
>                   refusing to run without repair.
> ```
>
> The name means "packaged for a runtime", not "runs on MTPLX". The `moe` entry
> above is verified-native with a complete contract (`can_run: True`), which is
> what actually matters. If a model is not in this table, check it before
> downloading 20 GB:
>
> ```bash
> mtplx inspect <model-dir> --json    # look for "can_run": true
> ```

**On Gemma 4 — verified, and the answer is no.** Gemma 4 is not in the list and
cannot be, for a structural reason rather than an oversight.

Gemma 4 does not use an MTP head. MTPLX drives it through a **target/assistant
pair**: a `target/` folder holding the verifier model and an `assistant/` folder
holding a separately-trained drafter, bound together by `mtplx_pair.json`. The
only such pair that exists is built from `google/gemma-4-31B-it` plus
`google/gemma-4-31B-it-assistant` — both Google's aligned models.

Without that bundle MTPLX refuses outright. Pointed at a plain Gemma 4 MLX
checkpoint it answers:

```
tier              no-MTP
support_level     gemma4-pair-bundle-required
can_run           False
runtime_compat    incomplete-assistant-pair
message           Gemma 4 target folder detected, but MTPLX Gemma requires the
                  assistant-pair bundle root containing mtplx_pair.json,
                  target/, and assistant/.
```

And no uncensored Gemma 4 has one. HauhauCS's excellent uncensored Gemma 4 line
— 12B, 26B-A4B, 31B, E4B, E2B — is **GGUF only**, shipped with a separate MTP
file for llama.cpp's own speculative decoding. The uncensored Gemma 4 MLX
conversions that exist (heretic, abliterated) are plain single-model
conversions: no assistant, no pair manifest, no runtime contract.

Building a pair for an uncensored target would mean converting Google's
assistant into MTPLX's format — which needs custom classes, since stock
`mlx_lm.convert` does not support `model_type=gemma4_assistant` — and then
running it against a target it was never trained for. That is a Forge project,
not a setting, and the acceptance rate would suffer for exactly the reason the
pair exists.

**So: uncensored Gemma 4 belongs to llama.cpp in this bundle's world.** It is a
different stack, not a different line in `env.conf`.

```bash
./model_download.sh                # what is available, what you have
./model_download.sh 9b             # download one
./model_download.sh --switch 9b    # download if needed, switch, restart
./start.sh --model 9b              # serve a different model for one run
```

`--switch` rewrites `MODEL` in `env.conf` and restarts; `--model` is a one-off.
Any `owner/name` MTPLX repo id also works if you have one of your own. Downloads
resume, so an interrupted fetch is cheap to restart.

The settings you are most likely to touch:

```conf
MODEL="hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16"
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

### Changing settings

`env.conf` settings are passed to the runtime as command-line arguments at
launch, so **editing the file has no effect until you restart**. That is what
`restart.sh` is for:

```bash
$ ./restart.sh
UpinelAIOS-MLX restart

  env.conf changed since the server last started:
    CONTEXT_WINDOW         131072  ->  204800
    KV_QUANT               q8  ->  off
    THINKING               low  ->  high

  left of the arrow is what is running now; right is what will start.

  Starting with:
    model      hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16
    served as  Upinel-AIOS-MLX
    context    204800   KV off   MTP depth 2
    profile    sustained   thinking high   history scoped

==> Stopping pid 78517 (SIGTERM)...
 ok Port 8000 is free.
==> Waiting for the port and GPU memory to settle...
==> Loading the model - first token takes ~30-90s.
 ok Server is up.
```

It diffs the *effective* settings, not the file, so editing a comment or
reordering lines does not show up as a change. `--print` shows what would
happen without doing it.

The stop is graceful and the script waits for the port to free plus a few
seconds before starting, because macOS needs a moment to release the wired GPU
allocation — starting into a port that is still closing is the usual cause of a
failed restart. If the graceful path hangs, `--force` sends `SIGKILL`.

---

---

## Going deeper

### Why these choices

Three decisions dominate everything else, and all three are backed by
measurement on this exact hardware rather than by convention.

#### 1. MLX, not GGUF: because llama.cpp's MTP loses on Metal

llama.cpp added `--spec-type draft-mtp` and the GGUF you may already have embeds
the head. On Apple Silicon it does not pay off: Metal does not amortise a 3–5 row
verify batch the way CUDA does, so the draft passes cost more than they save.
The MLX conversion of this same fine-tune reports:

> "the same fine-tune as Q6 GGUF with the FastMTP sidecar in llama.cpp reached
> **~12.5 tok/s** on the same hardware" — against 30.9 tok/s for the MLX build.

We reproduced the shape of that on this Mac: llama.cpp + `--spec-type draft-mtp`
reached **20.7 tok/s**; the MLX stack reaches **53.4**. That is the whole
argument. Details in [docs/TUNING.md §1](docs/TUNING.md).

#### 2. MTP depth 2: a 3.4× win, and the optimum is not the maximum

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

#### 3. 128K context, and it could be 262K

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

#### What long context actually costs you

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

### Portability notes

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

---

## Benchmarks

### Measured throughput

**Measured on an M5 Pro (20-core GPU, 64 GB).** Decode is the rate once
generating; prefill is the rate ingesting the prompt; TTFT is time to first
token. Every model below is uncensored.

| model | decode | prefill | TTFT | when to use it |
|---|---:|---:|---:|---|
| **`moe`** &nbsp;Qwen 3.6 35B-A3B | **79.4 t/s** | 32 t/s | 3.7 s | **The default.** MoE, ~3B active per token, and the fastest thing here by a wide margin. |
| `4bit` &nbsp;Qwen 3.8 27B dense | 34.7 t/s | 112 t/s | 1.5 s | When you want the best quality this project can serve and will trade speed for it. |
| `9b` &nbsp;Qwen 3.8 9B | 65.1 t/s | 55 t/s | 4.3 s | Only when memory is tight. The `moe` is faster *and* more capable, so treat this as a fallback rather than a preference. |

> Prefill and TTFT were measured on **short prompts**, where per-request
> overhead dominates the figure. They indicate responsiveness, not long-context
> prefill.

#### The same models on both runtimes

This is the clearest reason UpinelAIOS-MLX exists alongside
[UpinelAIOS-GGUF](https://github.com/Upinel/UpinelAIOS-GGUF). The two projects
serve the same Qwen models through different runtimes, and MTPLX's MTP
implementation runs on Metal where llama.cpp's does not:

| model | GGUF / llama.cpp | **MLX / MTPLX** | MLX advantage |
|---|---:|---:|---:|
| Qwen 3.6 35B-A3B | 70.5 t/s | **79.4 t/s** | ~13% |
| Qwen 3.8 9B | 44.3 t/s | **65.1 t/s** | **~47%** |
| Qwen 3.8 27B | 13.5 t/s | **34.7 t/s** | **~2.6×** |

The denser the model, the bigger the MLX win — at 27B it is most of a 3×
speedup. **If you are going to run Qwen, run it here.**

#### Every run varies, so quote the range

These figures are medians, and the spread around them is real rather than an
artefact of the tooling: the same model, prompt and context produced 80–110 t/s
across repeat runs of the `moe`. MTP commits a variable number of tokens per
forward pass depending on how predictable the continuation is, so throughput
genuinely moves with the text being generated.

**Estimated** for other Macs, scaled by memory bandwidth and cross-checked
against published MTPLX figures. Treat these as order-of-magnitude:

| Mac | short-context decode | at ~10k context |
|---|---:|---:|
| M1 / M2 (any) | 15–25 t/s | 6–10 t/s |
| M3 / M4 base | 25–35 t/s | 10–15 t/s |
| M4 Pro / M5 base | 40–50 t/s | 15–20 t/s |
| **M5 Pro**, Qwen 3.6 35B-A3B (measured) | **79 t/s** | **~74 t/s** |
| **M5 Pro**, Qwen 3.8 27B (measured) | **35 t/s** | **~19 t/s** |
| M4 Max / M5 Max | 55–65 t/s | 20–25 t/s |
| M3 Ultra | 60–75 t/s | 22–28 t/s |

The jump from M5 Pro to M5 Max is much smaller than the bandwidth ratio
suggests. Beyond a point these models stop being purely memory-bandwidth-bound,
so a 2× wider chip does not give 2×.

#### Why the MoE is the default

The `4bit` dense 27B is the quality pick, but it is not the speed pick, and the
ceiling on it is arithmetic rather than tuning:

- It is dense, so every token reads all ~15 GB of 4-bit weights. That is what
  this memory bandwidth supports.
- MTP is what beats that limit — it verifies several drafted tokens per weight
  read — and it takes the 27B to 34.7 t/s. Going much beyond that would need ~5
  tokens per pass, and draft acceptance decays sharply with depth (98% → 91% →
  81% at positions 1/2/3), so a deeper draft is not viable.

A mixture-of-experts checkpoint sidesteps the limit entirely by computing only
its active parameters. `Qwen3.6-35B-A3B` has 35B total but **3B active**, so each
token reads roughly a tenth as many weights — which is why it reaches
**79.4 t/s**, well over twice the dense model's rate. That is the whole reason
it is the default.

Prefill improves too, from ~440 to ~1,200 t/s, so long prompts arrive sooner as
well.

> **An earlier, faster MoE was dropped, and the numbers are worth keeping.**
> The `moe` alias used to point at `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`,
> a deliberately speed-optimised build of the same architecture, recorded at
> **78 / 208 / 125 t/s** across the three contexts above.
>
> **208 t/s at 8k is not credible as a decode figure, and re-measuring explains
> why**: it is reproducible if the prompt is a single sentence repeated hundreds
> of times. On the current MoE build, an identical setup gives **117 t/s** at 10k
> with a repetitive filler prompt and **74 t/s** with varied prose — same model,
> same context, same tool. MTP commits more tokens per pass when the
> continuation is predictable, so a filler prompt inflates long-context decode by
> more than 50%. The 208 reading is almost certainly that artefact, not the
> machine warming up, which is what an earlier version of this note guessed.
>
> The 8k and 32k figures are the ones to trust, and only for a filler prompt.
>
> It is not here because it is **Qwen's own aligned model**, and this project
> ships uncensored only. `moe` now points at a Heretic abliteration of the same
> architecture, which is why long-context throughput is lower: same shape,
> different fine-tune, not the same tuning.
>
> If throughput matters to you more than refusals, that build is one line away —
> `MODEL="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"` — and saying so
> plainly beats pretending the uncensored one is just as quick.

**Both are installed with `./install.sh` and switched with `./model_download.sh --switch`.**
The MoE is the default; the dense 4-bit 27B is the pick if you want the specific
HauhauCS fine-tune.

MTP depth 1 is optimal for the MoE and depth 2 for the dense model — tuning
results are stored per model in `run/tuning-<model>.json`, so switching models
does not silently reuse the wrong depth.

---

### Making it faster

Ordered by what actually moves the numbers, based on measurement rather than
convention:

1. **Keep contexts short.** This is by far the biggest lever — 51 t/s at 400
   tokens against 17 t/s at 10,000 on the same machine. Use retrieval and
   targeted file reads instead of loading whole repositories. Nothing else on
   this list comes close.

2. **Turn thinking off** (`THINKING="off"`, the default). On a trivial question
   that is 4 output tokens instead of 62, and thinking tokens are generated at
   the same speed as answer tokens. See [docs/TUNING.md](docs/TUNING.md) for why
   `"low"` does not reliably reduce it.

3. **Keep prefixes stable so the cache hits.** A shared prefix prefilled at
   5,857 t/s against 444 t/s cold in testing. An agent that injects a timestamp
   or reorders its system prompt every turn throws that away.

4. **Re-tune MTP depth** (`./bench/bench.sh --tune`) — worth a few percent, and
   the optimum is machine-specific.

5. **Switch to the MoE model** if you want a step change rather than a few
   percent. This is the only lever here that changes the order of magnitude.

6. **`FAN_MODE="smart"`** for long runs. A MacBook decoding for seven minutes
   straight will throttle; the dashboard's throughput chart makes it visible.

Be sceptical of tuning folklore beyond this list. `KV_QUANT`, batching presets
and stream intervals are real, but each is worth single-digit percent, while
context length and model choice are worth multiples.

### Benchmarking

```bash
./bench/bench.sh --quick      # ~30 s sanity check
./bench/bench.sh              # 512 / 8k / 32k / your configured context
./bench/bench.sh --tune       # re-measure AR vs depth 1/2/3
```

The harness times a streaming request client-side and reports TTFT, prefill
rate, and steady-state decode, which is what a client actually feels. Samples
under 32 generated tokens are flagged as noise.

---

---

## Running it day to day

### Updating the code

This is a git checkout, so updating means pulling. There is no update script:
`./model_download.sh` is for models, and this is for code.

```bash
git pull
./restart.sh          # to run the new code
```

**If `git pull` refuses**, that is expected and easy to fix. `env.conf` is
tracked, and editing it is the whole point of the project, so git stops with:

```
error: Your local changes to the following files would be overwritten by merge:
        env.conf
```

Set your settings aside, pull, and put them back:

```bash
git stash push -m "my settings" -- env.conf
git pull
git stash pop
./restart.sh
```

If upstream changed the same lines you did, `git stash pop` reports a conflict
and leaves `<<<<<<<` markers in `env.conf`. **Do not leave it like that** —
`env.conf` is what every script sources, so a conflicted one breaks the whole
project. Take upstream's version and recover yours from the stash:

```bash
git checkout HEAD -- env.conf      # upstream's, known good
git stash show -p stash@{0}        # see what you had
git checkout stash@{0} -- env.conf # or just restore yours and edit it
git stash drop                     # once you are happy
./restart.sh
```

**To see what changed before you commit to it:**

```bash
git fetch origin
git log --oneline HEAD..origin/main         # what is coming
git diff --stat HEAD..origin/main           # which files
```

Your models are never touched by any of this. `models/`, `run/` and `outputs/`
are gitignored, so an update cannot disturb what you have downloaded — only
`./model_download.sh` changes those.

> **`./install.sh` is only needed when a requirement changed** — a new
> dependency or a new setup step. A code-only update just needs
> `./restart.sh`.

### Watching it work

`./status.sh` is a live dashboard, refreshed once a second:

```
  UpinelAIOS-MLX                                        ● serving  up 02:14:33
  ──────────────────────────────────────────────────────────────────────────────
  itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit
  sustained · MTP d2 · think low · hist scoped · KV q8 · ctx 131072

  HOST                                ENDPOINT PROCESS
  CPU   █████░░░░░░░░░░░   32.6%      weights        14.9G ███████░░░
  GPU   ████████████████    100%      session bank   2.55G █░░░░░░░░░
  ANE   ░░░░░░░░░░░░░░░░   n/a        generation     3.82G ██░░░░░░░░
  RAM   ███████████░░░░░   44.9G      kv cache       5.76G ███░░░░░░░
  SWAP  ██████████████░░   11.4G      ───────────────────────────────
  cpu 17u/15s  nominal  pressure L1   total          21.3G  peak 28.9G
                                      host free      5.33G  (+13.8G cached)

  CONCURRENT ACTIVITY                                  CLIENTS  connected peers
  in-flight 1   sessions 1   lane solo_mtp             10.0.1.69   1 conn  python3.1
  chatcmpl-2b476…   219s anon-1125623 23865 ctx  1867 tok @ 16.7 t/s
    “please just write something, it seems like you are n…”

  TOKEN RATE  live, last 126 samples  min 14.5 avg 16.1 max 22.9   TOKENS GENERATED
    17.0                                      ▄▃▃▃▃▃▂▂▁▁▁▁▁  ▁   output        16,105
                                        ▁▃▆█▅▆▅▆▇████████████  input        384,355
        ▅ ▂▃▂▁              ▁▁   ▇▇▇▅▄▄▃▅▅▆▅▅▆▇▇█████████████  total        400,460
        ████████▇▇▂▄▅▅▅▁▃▄▄▄▃▃▅▆▇▇██▇▆▇▆▆▆████▇▇▇▅▆████████████  requests          38
    15.3 now 16.7 t/s                                            since restart      0
```

| Panel | What it shows | Source |
|---|---|---|
| HOST | CPU (with user/sys split), GPU, ANE, RAM, swap, thermal, memory pressure | `host_statistics`, IOKit, `vm_stat` |
| ENDPOINT PROCESS | the inference process's own MLX allocation: weights, session bank, generation working set, KV cache, total and peak | MTPLX telemetry `mem` |
| CONCURRENT ACTIVITY | in-flight requests with live per-request token counts and decode rate, session count, scheduler lane | MTPLX `in_flight`, `scheduler` |
| CLIENTS | TCP peers currently connected, with process names | `lsof` |
| TOKEN RATE | live decode rate, charted, with min/avg/max over the window | MTPLX `rolling.live_history` |
| TOKENS GENERATED | output, input, total and request counts | incremental scan of `run/server.log` |

The full API key and the model identity are printed in the header, untruncated,
because both are things you copy into a client.

#### Keyboard control

While the dashboard is running:

| key | action |
|---|---|
| `t` | cycle the thinking level |
| `m` | cycle the downloaded models |
| `Enter` | apply the pending change now |
| `Esc` | cancel the pending change |
| `q` | quit |

A toggle does **not** apply on the keystroke that chose it. It arms a
**2-second countdown** and applies when the countdown expires, so pressing the
key again moves to the next option without committing to the one you just
passed. The footer shows exactly what is about to happen:

```
── thinking: low → medium   applying in 1.9s   [same key] next  [Enter] now  [Esc] cancel
```

Thinking changes are applied live — the model stays loaded, so it is instant and
safe mid-generation. Model switches persist to `env.conf` and restart the
server, which takes 30–90s to load the new weights; the dashboard survives the
restart and shows `not running` until it is back.

`--no-keys` turns the toggles off for a passive display, and the bindings are
silently disabled when stdout is not a terminal.

`--once` prints a plain summary for logs and scripts, `--json` a
machine-readable snapshot, `--key` prints only the API key (handy for
`export KEY=$(./status.sh --key)`), and `--interval N` slows the refresh.

The window heading is pinned to **UpinelAIOS-MLX Status** for as long as the
dashboard runs, and released when you exit. This needs saying because it is not
automatic: Terminal.app titles the window from whichever process is in front of
the tty, and a dashboard that shells out to `lsof`, `ioreg` and `pmset` once a
second would otherwise make the heading flicker between those names and
`python3`. The probes are launched with `setsid` so they have no controlling
terminal at all, and the title is re-asserted on every frame.

The layout adapts to your window. Panels are admitted in priority order —
identity, then host and process memory, then the token rate, then concurrent
activity — and anything that will not fit is dropped rather than drawn past the
bottom of the screen. If the activity panel is dropped, its essentials (who is
connected, how much is running) move into the footer line, so nothing important
disappears silently. On a 24-row terminal you get the header, the host and
process panels, and the token chart; on an 80-row terminal you get everything.

#### Two limits, stated rather than hidden

**Client IPs come from the socket table, not the request.** MTPLX does not
record which request came from which peer, so `CLIENTS` answers "who is
connected to this port right now", not "who sent this particular prompt". It is
still the right tool for "is anything else on my LAN using my Mac?" — which is
usually the actual question. Each row names the *peer* process, not the server's
own, so a local agent shows as `this Mac  DSH De` rather than the runtime's
process name.

**The ANE is shown as `n/a`.** Apple exposes the Neural Engine only through
`powermetrics`, which needs root — and MLX is GPU-only, so it genuinely is idle
for this workload. `--power` reads the real figures when you have passwordless
sudo. The GPU number needs no such compromise: IOKit publishes the GPU's own
busy counter unprivileged. Note that counter is **system-wide**, so WindowServer
and browser compositing are included; it answers "is the machine busy", not
"how much is the model using".

---

## Repo layout

```
env.conf              the only file you edit
install.sh            deps + model download + MTP depth tune
start.sh / stop.sh    server lifecycle
restart.sh            apply an env.conf change (diffs old vs new)
status.sh             live dashboard (--once / --json for scripts)
lib/dashboard.py      metrics collection and TUI rendering
lib/common.sh         config loading, memory math, health helpers
lib/fetch-model.sh    resumable HF downloader, no pip dependency
bench/bench.sh        sweep / tune entrypoint
bench/bench.py        streaming benchmark harness
bench/verify-tools.sh tool-calling diagnostic — run this if an agent misbehaves
bench/ab.sh           A/B two settings against each other
service.sh            optional launchd login service
lib/preflight.sh      hardware scan and per-machine configuration suggestions
lib/diff_config.py    pairs the config diff that restart.sh prints
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

### This project

**UpinelAIOS-MLX is © 2026 Nova Upinel Chow, released under the
[Upinel Personal Free License](LICENSE).** In short:

| | |
|---|---|
| **Personal use** | Free. Use it, change it, share it. |
| **Creators — YouTubers, KOLs, streamers, bloggers** | Free — **you keep the money you make from the content**. Videos, streams, posts, articles, images, tutorials — and any text or images you generate with it and publish. Just email [dev@upinel.com](mailto:dev@upinel.com) to say you're doing it. **You do not need a reply and should not wait for one** — the permission takes effect the moment you hit send. |
| **Other commercial use** | Needs the author's written permission — email [dev@upinel.com](mailto:dev@upinel.com). This covers use inside a company, selling a product or service built on it, and hosting it for others as a paid service. |
| **Derivatives** | Must credit the author and keep this licence. Fork it, port it, improve it — just leave the name on it. |
| **Scope** | Covers this project's own code only. llama.cpp, MLX, MTPLX, Gemma 4, Qwen, and every model weight belong to other people and keep their own licences — see [section 3 of the licence](LICENSE) for the full list. |
| **Warranty** | None. It is a local inference server; you are responsible for what you run on it. |

So: making a video about it, reviewing it, or using it on stream to generate
content is free, forever, no permission needed — tell us and carry on. Selling
it, or selling something that runs on it, is the thing to ask about.

> **One licence file, both projects.** `LICENSE` is byte-identical in
> UpinelAIOS-MLX and UpinelAIOS-GGUF, and its scope clause covers both editions,
> so you never have to work out which terms apply to which repo. To confirm the
> two have not drifted:
>
> ```bash
> shasum -a 256 LICENSE
> # 0906eccc1ebc7e22f8de876997a4f33b7b85c3516d2c5a28796aad759fde7ff7
> ```

This is a **source-available** licence, not an open-source one: the
[OSI definition](https://opensource.org/osd) requires that a licence permit
commercial use, which this one deliberately does not. If you need a commercial
licence, or a different arrangement for your organisation, ask — the answer is
usually yes, and it is a short conversation.

The author is not a lawyer and this licence has not been reviewed by one. It is
written to be read and understood rather than to be maximally clever, but if you
are relying on it commercially, get your own advice.

### Third-party, and a second attribution you must not drop

This bundle is glue: the hard parts belong to other people.

- **[MTPLX](https://github.com/youssofal/MTPLX)** by Youssof Altoukhi — Apache-2.0,
  with a **mandatory in-product attribution requirement**. If you ship something
  built on this, it must display *"Powered by MTPLX — https://github.com/youssofal/MTPLX"*
  somewhere a user can see it. That obligation is independent of this project's
  licence and outlives it: a derivative of UpinelAIOS-MLX owes credit to
  **both** authors.
- **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — Apache-2.0.
- **HauhauCS** — the Aggressive uncensored fine-tune.
- **itrejomx** — the MLX 4-bit MTP conversion used by default.

Third-party components keep their own licences; this project's licence does not
cover them, and it covers no model weights at all.

### Does this licence conflict with what the project uses?

No, and the reasoning is worth stating rather than asserting:

- **Nothing third-party is redistributed here.** `install.sh` runs
  `brew install`, and the runtime is started as a separate process and spoken to
  over HTTP. MIT and Apache-2.0 impose obligations when you distribute their
  code; nothing is vendored, so nothing of theirs is relicensed, and their
  notices stay with them.
- **Everything upstream is permissive** — llama.cpp (MIT), MLX (MIT), MTPLX
  (Apache-2.0), Gemma 4 and Qwen (Apache-2.0). There is no copyleft anywhere in
  the dependency tree, which is precisely what makes it possible to license this
  project's own code restrictively.
- **Every Python import here is standard library.** There are no third-party
  Python packages to account for.

The one real incompatibility: because this licence forbids commercial use and
requires derivatives to stay no more permissive, this code **cannot be combined
into a GPL or AGPL project**. Both of those permit commercial use, so the terms
conflict in both directions. Nothing used here is GPL, so it costs nothing
today — but it does close that door.

MTPLX's own attribution requirement, described above, is a live obligation
carried by its NOTICE file under Apache-2.0 §4(d). If you *ship* MTPLX — as
opposed to installing it for yourself — that notice travels with it, and nothing
in this project's licence can waive it.

Model weights carry their own upstream licences. Uncensored fine-tunes are
uncensored: you are responsible for how you use the endpoint, and for the fact
that you just exposed it to your LAN.
