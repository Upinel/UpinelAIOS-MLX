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
# Serving Qwen3.8-27B (`qwen3_5` hybrid + 1-layer MTP head) on MLX / Apple Silicon — technical report

**Target:** M5 Pro, 20-core GPU, 64 GB unified memory, OpenAI-compatible LAN endpoint.
**Date of research:** current session. All claims cited; speculations labelled.

---

## 0. Ground truth about the model

`Qwen/Qwen3.8-27B`'s `config.json` declares:

```json
{ "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": { "model_type": "qwen3_5_text", "num_hidden_layers": 64,
                   "mtp_num_hidden_layers": 1,
                   "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention", ...],
                   "full_attention_interval": 4, "hidden_size": 5120, "vocab_size": 248320,
                   "max_position_embeddings": 262144 } }
```
Source: <https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json>

So "Qwen3.8-27B" IS the `qwen3_5` architecture (`Qwen3_5ForConditionalGeneration`, hybrid Gated-DeltaNet
linear attention interleaved 3:1 with full attention, 64 layers) with **one native MTP layer**
(`mtp_num_hidden_layers: 1`). Every runtime below keys off `model_type` starting with `qwen3_5`.

---

## 1. What "MTPLX" is

**MTPLX is a runtime (an inference engine + a model-packaging convention), not a quant scheme and not
merely a format variant.** It is an MLX-native, Apple-Silicon-only Python runtime that *uses* the
model's own MTP head as a speculative drafter.

- Repo: <https://github.com/youssofal/MTPLX> · site <https://mtplx.com> · PyPI project `mtplx`
  (repo `pyproject.toml` currently declares `version = "2.11.2"`).
- Its own description: *"MTPLX is a native Mac app and a command line for running local language models
  with multi-token prediction… the model drafts several tokens ahead of itself, verifies each drafted
  block in a single batched forward pass, and commits tokens through exact rejection sampling with
  residual correction."* It explicitly is **not** an external-drafter system and **not** a greedy-argmax
  trick — acceptance uses the Leviathan/Chen rejection-sampling + residual correction so sampling at
  `temperature=0.6, top_p=0.95` is distributionally exact.
  Source: <https://raw.githubusercontent.com/youssofal/MTPLX/main/README.md>

### 1.1 The `-MTPLX-` in a HF repo name means "MTPLX-branded pack"

A pack built by MTPLX Forge is an MLX safetensors trunk **plus a sidecar MTP head and a runtime contract**:

- trunk shards + `mtp.safetensors` (bf16 MTP sidecar, ~810 MB for the 27B) + `model-vision.safetensors`
  + `mtplx_runtime.json`
- `config.json` may also carry `mlx_lm_extra_tensors.mtp_file`, `mtplx_mtp_contract`,
  `mtplx_mtp_quantization`, `mtplx_mtp_payload_audit`.

Evidence (barozp card): *"Recipe: `body_bits=4`, `body_group_size=64`, `body_mode=affine`,
`mtp_policy=keep_bf16`, quantized trunk + bf16 MTP sidecar … Output: 3 safetensors shards +
`model-vision.safetensors` … + `mtp.safetensors` (810 MB, bf16 sidecar) + `model.safetensors.index.json`
… `mtplx_runtime.json` ships in-repo as provenance."*
Source: <https://huggingface.co/barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit/raw/main/README.md>

**Critical architectural note (this bites everyone):** MTPLX's own loader *refuses* packs that keep
`mtp.*` tensors embedded in the trunk shards, and conversely a stock mlx-lm/omlx load of a *sidecar*
pack will not see the head at all. From `docs/model-compatibility.md`: *"A MTPLX-branded pack stores its
MTP head as a standalone `mtp.safetensors` sidecar. Do not brand or redistribute an artifact that keeps
`mtp.*` tensors embedded in the trunk shards … MTPLX's own loader refuses such a trunk at load with the
cause named."* Source: <https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/model-compatibility.md>

### 1.2 How you actually RUN an MTPLX model

```bash
# Recommended on macOS
brew install youssofal/mtplx/mtplx

# or Python-only
python3 -m pip install -U mtplx
python3 -m pip install mlx            # MLX must be importable in the same env

mtplx doctor --summary                # hardware/dependency check
mtplx pull barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit
mtplx inspect <repo> --json           # compatibility tier before anything runs
mtplx tune --model <repo> --retune    # measure AR vs D1/D2/D3 on THIS Mac, save the winner
```

Sources: <https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/install.md>,
<https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/quickstart.md>,
<https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/cli.md> (404 — CLI reference lives in README
+ `docs/quickstart.md`).

**Serving (OpenAI-compatible):**
```bash
mtplx serve --model barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit --depth 3 --port 8080
# or
mtplx quickstart --port 8000
```
Endpoints documented: `GET /health`, `GET /metrics`, `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions`, `POST /v1/messages` (Anthropic), `GET /admin/sessions`, `POST /admin/cache/clear`.
Source: <https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/server.md>

**LAN binding** (default is `127.0.0.1`; non-localhost **requires** an API key):
```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key-file ~/.mtplx/api-key
# or inline:
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_API_KEY"
```
Startup prints a `Network OpenAI API Base URL`, e.g. `http://192.168.1.20:8000/v1`. Clients send
`Authorization: Bearer <key>` or `X-API-Key: <key>`. Sources: `docs/server.md`, `TROUBLESHOOTING.md`
(<https://raw.githubusercontent.com/youssofal/MTPLX/main/TROUBLESHOOTING.md>).

Depth controls: `--depth N` (default 3), `mtplx tune --retune`, in-session `/mtp on|off|status`,
`--no-mtp` for target-only AR. Concurrency: `--scheduler-mode serial|cooperative|ar_batch|mtp_batch|mtp_cohort_experimental|hyper`
(default `serial`). Source: <https://raw.githubusercontent.com/youssofal/MTPLX/main/docs/concurrency.md>

**Licensing caveat:** Apache-2.0 **with a mandatory in-product attribution** ("Powered by MTPLX",
<https://github.com/youssofal/MTPLX>) if you ship a product built on it. Source: MTPLX `README.md` /
`NOTICE`.

### 1.3 The three named repos, decoded

| Repo | What it actually is |
|---|---|
| `itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` | Community MLX 4-bit conversion (group 64) of a HauhauCS abliterated fine-tune, quant lineage bf16 → Q8_K_P GGUF → f16 → MLX 4-bit. MTP head kept bf16, `contract=true`, forged & verified with MTPLX. Thinking **off by default** in its chat template. |
| `barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit` | MTPLX 4-bit (16.9 GB / 15.7 GiB) conversion of an Opus-distill v2 fine-tune, vision included, forged with `mtplx==2.9.1` **on an M5 Pro 18CPU/20GPU 24 GB**. |
| `PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed` | Byte-for-byte repack of the official `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` recipe, built with MTPLX 2.7.1: bulk 4-bit/gs32, embeddings + LM head + GDN output projections + last 8 MLP blocks at 8-bit/gs64, norms/state/MTP head bf16. |

Sources: the three raw READMEs —
<https://huggingface.co/itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit/raw/main/README.md>,
<https://huggingface.co/barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit/raw/main/README.md>,
<https://huggingface.co/PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md>.

Note the `itrejomx` card is explicit that it is *"a community conversion, not an official HauhauCS
release (their releases carry Ed25519-signed manifests; this one does not)"* and that the quantization
lineage stacks a third quant layer on an already-ablated fine-tune.

---

## 2. `jundot/omlx`

- Repo <https://github.com/jundot/omlx>, site <https://omlx.ai>, Apache-2.0, macOS 15+, Python 3.11–3.13,
  Apple Silicon M1–M5. FastAPI + uvicorn, `mlx==0.32.2`, `mlx-lm` git-pinned at `ab1806e8` (v0.31.3),
  plus `mlx-vlm`, `mlx-embeddings`, and a forked `dflash-mlx`.
  Sources: <https://raw.githubusercontent.com/jundot/omlx/main/README.md>,
  <https://raw.githubusercontent.com/jundot/omlx/main/pyproject.toml>

### 2.1 Install

```bash
# Homebrew (recommended)
brew tap jundot/omlx https://github.com/jundot/omlx
brew install jundot/omlx/omlx
omlx start                      # managed background service

# Custom kernels (GLM-5.2 / MiniMax M3 / Qwen3.5 families) need a HEAD build
brew install jundot/omlx/omlx --HEAD --with-custom-kernel

# From source
git clone https://github.com/jundot/omlx.git && cd omlx
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .     # needs FULL Xcode for the Metal toolchain
# (plain `pip install -e .` silently skips native kernels)
```
Also ships a signed macOS menu-bar app (`.dmg` from Releases) that installs a `~/.omlx/bin/omlx` shim.
Source: omlx `README.md`.

### 2.2 Serve command

```bash
# Zero-config default: 127.0.0.1:8000, models in ~/.omlx/models
omlx serve --model-dir ~/models

# LAN bind + auth
omlx serve --model-dir ~/models --host 0.0.0.0 --port 8000 --api-key your-secret-key

# Useful tuning flags
omlx serve --model-dir ~/models \
  --memory-guard balanced \
  --max-concurrent-requests 16 \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --hot-cache-max-size 20%
```
**Default bind is `127.0.0.1`** (`omlx/settings.py: host: str = "127.0.0.1"`), override with
`--host 0.0.0.0` or `OMLX_HOST`. Sources: omlx `README.md` and repo source
(`omlx/settings.py`, `omlx/cli.py`).

### 2.3 OpenAI-compatible API

Yes — and more. `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/messages` (Anthropic),
`POST /v1/embeddings`, `POST /v1/rerank`, `GET /v1/models`, plus a chat UI at `/admin/chat` and an admin
dashboard at `/admin`. Any OpenAI-compatible client connects to `http://localhost:8000/v1`.
Source: omlx `README.md`.

### 2.4 MTP speculative decoding for Qwen3.8-27B — YES, opt-in

omlx implements what it calls **Lightning MTP**, a monkey-patch of mlx-lm (upstream "PR 990 / PR 15")
that adds an MTP head to `mlx_lm.models.qwen3_5.TextModel`:

- Setting: per-model `mtp_enabled` (**default `False`**), with `mtp_num_draft_tokens` for depth;
  global ceiling `MAX_LIGHTNING_MTP_DRAFT_TOKENS = 8`.
- Gate `_is_mtp_compatible()` accepts `model_type` starting with **`qwen3_5`**, `qwen3_6`,
  `deepseek_v4`, `nemotron_h`, plus `glm_moe_dsa`, `glm5_next`, `gemma4`, `gemma4_unified`,
  `inkling`, `step3p7` — **but only if `_checkpoint_has_mtp_weights()` finds `mtp.*` tensors (or native
  nextn layers) in the weight index**.
- Faithful patch files exist for the qwen3.5 family: `omlx/patches/mlx_lm_mtp/qwen35_model.py`,
  `batch_generator.py`, `cache_rollback.py`, `prompt_priming.py`, `norm_repair.py`.
- `mtp_enabled` is mutually exclusive with `dflash_enabled` and with MoE expert offload.
  Sources (repo source): `omlx/model_settings.py`, `omlx/utils/model_loading.py`,
  `omlx/patches/mlx_lm_mtp/*`, `omlx/admin/routes.py`.

**Failure mode you will actually hit with plain MLX 4-bit conversions:** the admin gate returns
*"Config declares MTP layers but the weight files contain neither `mtp.*` tensors nor native nextn
layers. Re-convert from HF with a converter that preserves MTP weights."* — i.e. many community 4-bit
conversions (including `mlx-community/Qwen3.8-27B-4bit` and `orcarouter/Qwen3.8-27B-Uncensored-MLX`)
simply dropped the head. Confirmed independently by a user report:
<http://reatang.com/p/qwen38-27b-omlx-dflash-mac/> (M5 Pro / 64 GB, oMLX 0.6.3rc3): *"oMLX 有原生 MTP
支持，但这个模型用不了"* with the exact runtime message above.

**Good news — oMLX can import MTPLX sidecar packs.** `omlx/oq.py` has `import_mtplx_sidecar()` plus
`_resolve_mtplx_sidecar()` and `_validate_mtplx_runtime_contract()` (fail-closed allowlist:
`arch_id == "qwen3-next-mtp"`, `base_hidden_variant`/`hidden_variant == "post_norm"`,
`concat_order == "embedding_hidden"`, `mtp_position_mode ∈ {local, cache}`). It normalizes the sidecar
onto a `model-mtp.safetensors` shard so mlx-lm's index-based MTP detection sees it. Exposed at
`POST /admin/api/models/{model_id}/import-mtplx` (and as a one-click dashboard button that appears with
the marker *"MTPLX side-car detected but not imported. Import it to merge the MTP head into the
checkpoint index."*). Sources: `omlx/oq.py`, `omlx/admin/routes.py`.

So the practical omlx path to MTP on a Qwen3.8-27B MTPLX pack is: place the repo in `--model-dir`,
then import the sidecar, then reload the model.

### 2.5 Benchmarks page

`https://omlx.ai/benchmarks` (and `/benchmarks/performance/<id>`) currently returns **HTTP 503** with
the message *"Community Benchmarks will be back soon. Recent growth exceeded the capacity of our current
database design… expect to reopen benchmark browsing within the next few days."* The repo's
`benchmarks/` directory contains only measurement scripts (`bench_m5_sorted_gather_chunk.py`,
`bench_qwen4_qsa_sparse_gqa.py`, `qwen35_ane_*_bench.py`, `tp_identity_probe.py`, …), not result tables.
Community numbers from those pages are still discoverable via search snippets — see §7.

---

## 3. What is `mlx-serve`? (and decoding your manifest string)

`mlx-serve` is **`ddalcu/mlx-serve`** — <https://github.com/ddalcu/mlx-serve>, MIT, site
<https://mlxserve.com>. It is *"a native Zig server that runs any LLM on Apple Silicon"*: a single ~7 MB
Zig binary linking MLX through `mlx-c` FFI (no Python at runtime) **plus an embedded llama.cpp** for
GGUF. It ships MLX Core, a signed/notarized macOS menu-bar app. Default port **11234**.

Source: <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/README.md>

### 3.1 The manifest string, decoded

`mlx-serve-26.8.7|mlx-0.32.0|mlx-c-fba4470b8907|llama.cpp-b10034` is a flattened form of
`mlx-serve --version` output. That report is produced by `src/version.zig`, which prints one
`name value` line per component in a stable order:

```
mlx-serve 26.8.7
mlx 0.32.0
mlx-c fba4470
nax on (...)
ggml 0.16.0 (47c786924)
llama.cpp b10034
gguf ...
ds4 ...
```

- `src/version.zig` documents the format and that `mlx`, `ggml`, `ggml_commit` are runtime values while
  **`mlx-c` is the pinned submodule revision** and **`llama.cpp` is the pinned `LLAMA_TAG`**.
- `docs/gotchas/engine-mlx.md` confirms the mlx-c pin is the short SHA **`fba4470`** from mlx-c main.
- **v26.8.7 is precisely the release that added Qwen 3.8 27B support.** CHANGELOG: *"v26.8.7 — Qwen 3.8
  27B, Ling 3.0 … Qwen 3.8 27B runs on your Mac, hours after Qwen released it. Text, vision and tools, in
  an 18.2 GB 4-bit build with the draft head baked in: about 75 tok/s on code and 40 on prose on an
  M4 Max."*
- `scripts/fetch-llama.sh` currently pins `LLAMA_TAG="${LLAMA_TAG:-b10472}"` (HEAD has moved on from
  b10034, your string is a v26.8.7-era build), and `Formula/mlx-serve.rb` / `Casks/mlx-core.rb` are at
  `26.9.2`.

Sources: repo files `src/version.zig`, `src/mtp.zig`, `CHANGELOG.md`, `docs/gotchas/engine-mlx.md`,
`scripts/fetch-llama.sh`, `Formula/mlx-serve.rb`, `Casks/mlx-core.rb` in
<https://github.com/ddalcu/mlx-serve>.

### 3.2 Install

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install --cask mlx-core     # the menu-bar app (recommended)
brew install mlx-serve           # CLI + server only

# From source (Zig binary; needs Xcode 26.2+ with Metal Toolchain)
git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
brew bundle install --file=Brewfile
./app/build.sh                   # or: zig build -Doptimize=ReleaseFast  (NEVER a bare `zig build`)
```
Requires macOS 26.2+ on Apple Silicon. Sources: `README.md`, `docs/building.md`.

### 3.3 Serve command (LAN-friendly by default)

```bash
mlx-serve run qwen3.8                      # Ollama-style: download + serve + REPL
mlx-serve serve                            # serve everything pulled; load on demand by name
mlx-serve --model ddalcu/Qwen3.8-27B-MLX-Serve-4bit --serve --port 11234
mlx-serve --serve --model-dir ~/.mlx-serve/models --host 0.0.0.0 --port 11234 --api-key SECRET
```
**Defaults are `--host 0.0.0.0 --port 11234`** (`docs/cli.md`: *"Bind address (all interfaces — set
`127.0.0.1` for strictly local)"*). `--api-key` is required only for non-localhost requests; loopback is
trusted. Source: <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/docs/cli.md>

### 3.4 MTP for `qwen3_5` — YES, auto, zero setup

- `src/mtp.zig` header: *"Architecture (matches mlx-lm `qwen3_5` MTP contract)"*.
- `docs/models.md` lists `qwen3_5`, `qwen3_5_moe` as natively dispatched model types and calls out
  `Qwen3.8-27B` (`ddalcu/Qwen3.8-27B-MLX-Serve-4bit`, 18.2 GB, "draft head baked in").
- `docs/performance.md`: *"**Native MTP** (Qwen 3.5/3.6/3.8) — checkpoints with a trained
  multi-token-prediction head (sidecar or baked in, like the Qwen 3.8 27B build) draft with the model's
  own head, with a controller that self-tunes depth per request. Auto-loads, zero setup."*
- CLI flags: `--mtp` / `--no-mtp` (default: **on when a sidecar is present**, off for MoE trunks
  unless per-request `enable_mtp`), `--mtp-depth N` (default 3), `--mtp-history-window N`.
- **It resolves MTPLX-format sidecars directly**: `src/mtp.zig` `resolveMtpSidecarInDir()` searches
  `mtp.safetensors`, `model-mtp.safetensors`, `optiq/mtp.safetensors`, plus `mtp/weights.safetensors`.
  FAQ: *"mlx-serve loads the same MTP sidecar artifacts (including MTPLX-published ones) with zero
  setup."*
- Coexistence with PLD is routed by an n-gram prompt score; the MTP head is the default, and the head
  holds ~93%+ per-draft on novel and echo content.
  Sources: <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/docs/performance.md>,
  <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/docs/models.md>,
  <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/docs/faq.md>, plus `src/mtp.zig`, `src/server.zig`.

**Bottom line for §3:** `mlx-serve` is the only one of the three that runs an MTPLX-branded Qwen3.8-27B
pack *as shipped*, with MTP, no import step and no Python.

---

## 4. Comparison: `mlx-lm` vs `mlx-serve` vs `omlx` vs `MTPLX`

| Capability | `mlx-lm` (ml-explore) | `mlx-serve` (ddalcu) | `omlx` (jundot) | `MTPLX` (youssofal) |
|---|---|---|---|---|
| **Kind** | Python library + CLI + FastAPI-ish server | Single Zig binary (mlx-c FFI) + embedded llama.cpp + GUI app | Python/FastAPI server + GUI app | Python runtime + GUI app |
| **Install** | `pip install mlx-lm` | `brew install mlx-serve` / `--cask mlx-core` | `brew install jundot/omlx/omlx` or `pip install -e .` | `brew install youssofal/mtplx/mtplx` or `pip install mtplx` |
| **`/v1/chat/completions`** | ✅ `mlx_lm.server` | ✅ + Anthropic `/v1/messages` + **Ollama API** + OpenAI Responses/WS | ✅ + Anthropic `/v1/messages`, `/v1/embeddings`, `/v1/rerank` | ✅ + Anthropic `/v1/messages`, `/v1/embeddings`, `/v1/rerank` |
| **MTP for `qwen3_5`** | ❌ (drafter-side spec-decode only; no native MTP in the released package) | ✅ **native, auto**, incl. MTPLX sidecars | ✅ opt-in `mtp_enabled` ("Lightning MTP", mlx-lm PR 990 patch), needs `mtp.*` in the index; MTPLX sidecar importable | ✅ its core feature |
| **Other spec-decode** | `--draft-model` / `-m` small drafter | PLD (on), DFlash drafter, Gemma-4 assistant, DSpark | DFlash2 (`dflash_enabled`, `dflash_draft_model`), SpecPrefill | — (own head only; refuses greedy shortcuts) |
| **KV cache quantization** | ✅ `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` | ✅ `--kv-quant {off,4,8,turbo2,turbo4}` + `--kv-attn-mode {auto,dense,fused}` | ✅ TurboQuant KV (`turboquant_kv_enabled`, `turboquant_kv_bits` 2/2.5/3/3.5/4/6/8) | ✅ internal `kv_quant_config` path |
| **Prompt caching** | ✅ `mlx_lm.cache_prompt` + prompt cache files | ✅ shared-prefix KV cache (`--prefix-cache-entries`, `--prefix-cache-mem`, `--prefix-cache-disk`, `--tokenize-cache-entries`); 11K-token restart TTFT 5.9 s → 0.7 s | ✅ **tiered** block KV cache: hot RAM tier + cold SSD tier (safetensors), prefix sharing + CoW, survives restarts | ✅ warm-prefix session bank + default-on SSD session cache |
| **Continuous batching** | ✅ `BatchGenerator` (Python API) | ✅ `--max-concurrent` (default **1**) — *"~1.6× total throughput at 4-way on dense models"* | ✅ mlx-lm `BatchGenerator`; `--max-concurrent-requests` default **8** | `--scheduler-mode` (default `serial`); `ar_batch` / `mtp_batch` / `hyper` |
| **Default bind** | `mlx_lm.server --host` (you must pass it) | **`0.0.0.0`** | `127.0.0.1` | `127.0.0.1` |
| **LAN auth** | none built in | `--api-key` for non-localhost; `--api-key-strict` to include loopback | `--api-key`; localhost-exempt by default | non-localhost bind **requires** an API key |
| **Runs MTPLX packs as-is** | ❌ | ✅ | ⚠️ after `/import-mtplx` | ✅ |

### 4.1 Exact install + serve commands

**mlx-lm** (reference baseline; no native MTP):
```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.8-27B-4bit --host 0.0.0.0 --port 8080 \
  --kv-bits 8 --kv-group-size 64 --quantized-kv-start 5000
```
`mlx_lm.server` accepts per-request `repetition_context_size` (`mlx_lm/server.py`: `body.get("repetition_context_size", 20)`, validated `min_val=0`). Sources:
<https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md>, `mlx_lm/server.py`.

**mlx-serve**:
```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install mlx-serve

mlx-serve --model barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit \
  --serve --host 0.0.0.0 --port 11234 --api-key "$MLXSERVE_KEY" \
  --mtp-depth 3 --max-concurrent 4 --kv-quant off \
  --prefix-cache-entries auto --prefix-cache-mem 4GB --prefix-cache-disk 32GB
```

**omlx**:
```bash
brew tap jundot/omlx https://github.com/jundot/omlx
OMLX_WITH_CUSTOM_KERNEL=1 brew install jundot/omlx/omlx --HEAD --with-custom-kernel

omlx serve --model-dir ~/models --host 0.0.0.0 --port 8000 --api-key "$OMLX_KEY" \
  --memory-guard balanced --max-concurrent-requests 8

# enable Lightning MTP for the Qwen3.8 model (admin panel or settings.json):
#   mtp_enabled: true, mtp_num_draft_tokens: 3
# for an MTPLX pack, first import the sidecar:
curl -X POST http://127.0.0.1:8000/admin/api/models/<model_id>/import-mtplx \
  -H "Authorization: Bearer $OMLX_KEY"      # then reload the model
```

**MTPLX**:
```bash
brew install youssofal/mtplx/mtplx
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed \
  --host 0.0.0.0 --port 8000 --api-key "$MTPLX_KEY" --depth 3
```

---

## 5. The repetition-penalty bug and MLX looping

### 5.1 `jundot/omlx` issue #3011 — summary

Title: **"repetition_penalty is inert for verbatim loop units longer than mlx-lm's default 20-token
window."** Author `xunlinkx` (CONTRIBUTOR), opened 2026-08-21, **closed by `jundot` 2026-08-23**,
0 comments, `state_reason: completed`. <https://github.com/jundot/omlx/issues/3011>

Environment: two-Mac TP2 cluster (M4 Max 48 GB each), oMLX HEAD-fa3e94b / 0.6.3rc2, **mlx-lm pinned at
`ab1806e8`**, `Qwen3.8-27B-oQ4e-mtp` over JACCL.

Reported failure: the reasoning trace started coherent, then *"collapsed into a fixed verbatim
oscillation (~100 repetitions of the same indecision flip) and burned the entire 8192-token completion
budget with `finish_reason=length` and no answer."*

Root cause: mlx-lm's repetition penalty only looks back `repetition_context_size` tokens —
`sample_utils.py`, `tokens[-context_size:]` — which **defaults to 20**. The loop unit was ~50 tokens, so
a token whose previous occurrence sat outside that window was never penalized, making the penalty value
irrelevant for this failure mode.

Their deterministic repro against the pinned mlx-lm, `make_repetition_penalty(1.5, ...)`, logit −2 on a
token whose most recent occurrence is N tokens back:

| repeat distance | default window (20) | window 128 |
|---|---|---|
| 10 | −3.0 (penalized) | −3.0 |
| 20 | −3.0 | −3.0 |
| 25 | **−2.0 (untouched)** | −3.0 |
| 45 | **−2.0 (untouched)** | −3.0 |

Second half of the bug: oMLX gave clients no way to widen the window — neither `ChatCompletionRequest`
nor `CompletionRequest` carried `repetition_context_size`, so pydantic silently dropped the key —
*while mlx-lm's own server already accepted it per request.* Proposed fix: forward
`repetition_context_size` end-to-end (request models → distributed rank payloads → `SamplingParams` →
`make_logits_processors`).

**Verification of the fix (I checked the current source):** oMLX now forwards it in `omlx/server.py` at
five call sites (`gen_kwargs["repetition_context_size"]` / `chat_kwargs[...]`), the field exists in
`omlx/request.py`, and the scheduler carries it. So the issue is genuinely resolved on `main`.
Sources: the issue JSON above, and `omlx/server.py`, `omlx/request.py`, `omlx/scheduler.py` in
<https://github.com/jundot/omlx>.

**I also confirmed the upstream claim in mlx-lm's own source:** `mlx_lm/sample_utils.py`
`def make_repetition_penalty(penalty: float, context_size: int = 20)` with `tokens = tokens[-context_size:]`,
and `mlx_lm/server.py` line 1179 `self.repetition_context_size = self.body.get("repetition_context_size", 20)`.
Source: <https://github.com/ml-explore/mlx-lm> (checked out at `main`).

### 5.2 Is MLX more loop-prone than llama.cpp for Qwen3.8? — Evidence found

**Direct evidence that this pathology is real and MLX-specific in practice (strong):**

1. **MTPLX ships a purpose-built Loop Guard whose docstring names the exact failure.**
   `mtplx/loop_guard.py`: *"Why this exists (2026-07-08, chess execute-plan marathons): Qwen3.6-class
   hybrid models can collapse into verbatim sentence/paragraph cycling during long reasoning segments
   ('OK, I'm going to start creating files...' repeated dozens of times). Qwen's own model card
   recommends presence_penalty 0-2 against 'endless repetitions', but static presence penalties degrade
   coding quality (they tax every reused identifier), so MTPLX refuses to ship one."* The guard is
   DRY-style (per `oobabooga/text-generation-webui#5677`), **armed only when a real loop is detected**,
   bit-exact when disarmed, and masks tool-call spans because *"tool-call payloads are the one place
   verbatim repetition is legitimate at loop density."* Config: `ngram=12`, `occurrences=4`,
   `window=2048`, `allowed_length=12`, `penalty=2.0`. `MTPLX_LOOP_GUARD=1` enables it (**default OFF**).
   Sources: `mtplx/loop_guard.py`, MTPLX `CHANGELOG.md` (Added: *"Loop Guard (opt-in)…"*).
2. **MTPLX has a hard repetition *stop* and a streaming holdback** for loops it can't steer away:
   `MTPLX_REPETITION_STREAM_HOLDBACK=candidate|strict|off`, plus documented cases where an uncapped
   repetition stop *"held a fixed ~448-token tail off the stream on every uncapped request"*, and a note
   that a two-turn terminal session on Flash-Next *"degenerated into repetition and a later run ended in a
   Metal GPU address fault"* (CHANGELOG).
3. **mlx-serve has the same class of feature**: a repetition-loop cut with
   `finish_details: {"type":"repetition_loop"}` and `loopTrimmedIds()` trimming tokens before the loop
   start (`src/server.zig`), i.e. two of three MLX runtimes built dedicated loop machinery.
4. **The fine-tune ecosystem blames trained data for deterministic loops.** `barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit`:
   v1's training data induced *"a deterministic self-verification loop on stacked format constraints
   ('no prose' + 'no markdown' → 3000/3000 tokens, zero visible output)"*, fixed by retraining on a
   cleaned dataset. That is a model/training issue rather than a runtime issue, and it's an important
   confound.
5. **The M5 Pro sustained-throughput report shows MTP `finish=parked` with 70–84% acceptance and 6–11
   tok/s** (issue #2976), which the maintainer fixed in #3004 by *"allowing MTP to retry after a
   cooldown"* — after which the same user measured 33.6 tok/s and 21.1 tok/s on long responses.

**What I could NOT find (label as unverified):** no controlled, same-machine, same-prompt A/B study of
MLX vs llama.cpp repetition/looping rates for Qwen3.8 specifically. The claim "MLX suffers more looping
than llama.cpp on Qwen3.8" is **not established by any source I found**. What *is* established is:

- the **mechanism** by which a `repetition_penalty` set by the user can be a no-op on MLX runtimes built
  on mlx-lm (issue #3011 + mlx-lm source), and
- that **MLX-side runtimes found it necessary to add dedicated loop detection** (MTPLX Loop Guard,
  mlx-serve repetition-loop cut), which is circumstantial evidence that the pathology is common enough to
  warrant engineering.

**Mitigations that are source-backed:**
- Widen the penalty window: send `repetition_context_size` per request (supported by mlx-lm's own server
  and now by oMLX; **not** exposed by mlx-serve or MTPLX as of this research — verify before relying).
- Presence penalty per request: MTPLX exposes `presence_penalty` (OpenAI field), `--default-presence-penalty`,
  `mtplx settings set`, and an app dial; *"values around 0.5–1.5 break repetition; 0 (the default) is an
  exact no-op."* Qwen's guidance per MTPLX: keep penalties at 0 for coding/agent work.
- MTPLX Loop Guard: `MTPLX_LOOP_GUARD=1` (+ `MTPLX_LOOP_GUARD_WINDOW`, `_NGRAM`, `_MASK_TOOL_CALLS`).
- Choose a checkpoint that fixed its loop pathology (the `-v2` Opus-Distill card documents a full
  before/after repro table).

---

## 6. GPU memory / wired-limit settings on macOS for MLX

### 6.1 The knobs

| Knob | Scope | Notes |
|---|---|---|
| `sudo sysctl iogpu.wired_limit_mb=N` | **kernel**, per-boot, system-wide | The real ceiling. mlx-lm's README: *"If the model fits in RAM then it can often be sped up by increasing the system wired memory limit… The value N should be larger than the size of the model in megabytes but smaller than the memory size of the machine."* |
| `mlx.core.set_wired_limit(bytes)` | per-process | Current API; `mx.metal.set_wired_limit` is the deprecated alias. MTPLX prefers the new name and falls back. |
| `mlx.core.set_memory_limit(bytes)` | per-process | Soft allocator budget (`mx.metal.set_memory_limit` legacy alias). |
| `mlx.core.set_cache_limit(bytes)` | per-process | Reclaimable MLX buffer cache (`mx.metal.set_cache_limit` legacy alias). |

Sources: <https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md>; MTPLX
`mtplx/server/openai.py` (`_set_metal_memory_limit(mx, "set_wired_limit", …)`, *"Prefer the new top-level
mx.set_memory_limit / mx.set_wired_limit; fall back to the deprecated mx.metal.* names"*);
MTPLX `docs/server.md` (`--mlx-cache-limit`, env `MTPLX_MLX_CACHE_LIMIT`, default
`max(1 GiB, min(8 GiB, memory_budget / 8))` when `--memory-budget` is set, else 8 GiB at ≥100 GB RAM).

### 6.2 Recommended values for a 64 GB Mac

The three runtimes compute different defaults, which is worth knowing before you copy a number from the
internet:

- **omlx** — ceiling = `min(static_ceiling, dynamic_ceiling, metal_cap)`, where
  `static_ceiling = total_ram − tier.static_reserve` and the reserves are
  `safe: 8 GiB`, `balanced: 6 GiB` (default), `aggressive: 4 GiB`, `custom: 2 GiB`
  (systems ≥ 24 GiB; below that a flat 4 GiB reserve). It clamps the value it *suggests* for
  `iogpu.wired_limit_mb` to **leave 5% of RAM free**, because wiring within a few GiB of physical RAM
  invites jetsam during large-model load bursts (issue #2184). Source: `omlx/process_memory_enforcer.py`.
  → For a 64 GiB Mac at the default `balanced` tier: **`sudo sysctl iogpu.wired_limit_mb=59392`**
  (58 GiB; the 5% clamp would allow up to ~62 259 MiB, but 58 GiB is what oMLX actually targets).
  *This specific number is my arithmetic on the source constants, not a quoted value.*
- **MTPLX** — on a 64 GiB Mac it sets `memory_limit = min(64 GiB, max(8 GiB, 0.75·RAM), 192 GiB) = 48 GiB`
  and `wired_limit = min(memory_limit, max(4 GiB, 0.60·RAM), 160 GiB) = 38.4 GiB`, then clamps
  `wired_limit ≤ memory_limit`. It never lowers the kernel sysctl; those are MLX allocator caps.
  Source: `mtplx/server/openai.py` (lines ~2427–2445).
- **mlx-serve** — `--max-resident-mem` auto = **80% of the MLX wired limit**; `--skip-mem-preflight` skips
  the free-RAM pre-flight. Source: `docs/cli.md`.

### 6.3 Caveats (important)

1. **Never wire close to physical RAM.** omlx issue #2184 (<https://github.com/jundot/omlx/issues/2184>,
   still open): if a process holding a large wired model is **hard-killed** (SIGKILL, `launchctl kickstart -k`,
   or jetsam), *"the entire wired allocation leaks at kernel level — it survives process death and is only
   reclaimed by a reboot."* On a Mac Studio M3 Ultra 512 GB with `iogpu.wired_limit_mb=499712` they saw
   *"233 consecutive startups ~2 minutes apart"* crash-loop, `vm_stat` wired = **414 GiB** with 0 omlx
   processes running, and one kernel panic. **`522240` caused the loop; `499712` was stable.** Also:
   *"Raising `iogpu.wired_limit_mb` toward physical RAM … makes jetsam during load more likely."*
   Practical rule on 64 GB: stay a few GiB below RAM, always stop the server **gracefully** (SIGTERM /
   `omlx stop` / `brew services stop`), and never `kill -9` a loaded server before rebooting.
2. **`iogpu.wired_limit_mb` is not persisted across reboots** by sysctl alone; you need a launchd
   daemon or `/etc/sysctl.conf` to make it stick (standard macOS `sysctl` behavior — not confirmed by a
   source in this research; label as conventional knowledge).
3. **Unset sysctl is a real state.** oMLX explicitly *skips* `mx.set_wired_limit` when
   `iogpu.wired_limit_mb` is unset (0) and leaves Apple's default Metal cap
   (`max_recommended_working_set_size`) active, logging an actionable warning with the exact
   `sudo sysctl iogpu.wired_limit_mb=N` command to run.
4. **MTPLX's 27B Optimized Speed measured peak unified memory = 23.6 GB** (weights + runtime) on the
   actual artifact, and 29.0 GB peak footprint for a 4K-context D3 run. A 64 GB Mac is not
   memory-constrained for this model — do not over-wire for no reason.
   Source: <https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md>,
   <https://huggingface.co/PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md>.
5. **Thermal drift is a measurement trap, not a memory setting** — but it destroys benchmarks. Measured
   −23% on the same mxfp4 config across three minutes without cooldowns; interleaved A/B with 90 s
   cooldowns took spread from 23% to under 2%. Source:
   <https://huggingface.co/KaedeTai/dflash2-mlx-quantized-draft/raw/main/README.md>.

---

## 7. Measured tokens/sec for Qwen3.8-27B 4-bit on Apple Silicon

**All numbers are as published by the cited source; hardware and protocol differ between rows and they
are NOT mutually comparable.** I found no published MTP-on/off number measured on an **M5 Pro 20-core
GPU with 64 GB** from a first-party source; the closest are the M5 Pro 24 GB and M5 Pro 48 GB rows and
the third-party M5 Pro 64 GB report.

### 7.1 MTPLX (MTP via the model's own head)

| Hardware | Quant | AR (depth 0) | D1 | D2 | D3 | Source |
|---|---|---:|---:|---:|---:|---|
| **M5 Pro, 18 CPU / 20 GPU, 24 GB**, macOS 27.0, mtplx 2.9.1 | Opus-Distill-v2 MTPLX 4-bit (16.9 GB) | 17.4 | 29.0 (1.67×) | 38.1 (2.18×) | **41.6 (2.39×)** | [barozp card](https://huggingface.co/barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit/raw/main/README.md) |
| same, 4K-token prompt (4,099 tok) | same | 26.4 decode / 907.8 prefill | — | — | **70.5 decode** / 810.5 prefill | same card |
| Unspecified M-series, 48 GB | Uncensored-HauhauCS MTPLX 4-bit (~16 GB) | 14.9 | 31.0 (2.08×) | **37.3 (2.51×)** | 32.5 (2.18×) | [itrejomx card](https://huggingface.co/itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit/raw/main/README.md) |
| **M5 Max, 40-core GPU, 128 GB**, macOS 26.4 | MTPLX Optimized Speed 4-bit (21.31 GB) | 24.62 | 41.69 (1.69×) | 52.85 (2.15×) | **58.02 (2.36×)** | [PocketAiHub card](https://huggingface.co/PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md) |
| M5 Max, fans pinned, official Qwen sampling | MTPLX Optimized Speed | — | — | — | **58.7** coding task via `mtplx serve`; 55.5 in the app; 35.1 / 37.3 at xhigh long reasoning | [Youssofal card](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md) |
| M4 Mac mini 16 GB (9B model, for scale) | MTP 9B | 14.4 | 23.0 | — | — | [MTPLX README](https://raw.githubusercontent.com/youssofal/MTPLX/main/README.md) |

Notable cross-engine data point from the Youssofal card, same night / same task: *"oMLX 0.5.7 with its
own Qwen 3.8 4-bit MTP quant ran 63.3 tok/s"* vs MTPLX 58.7, MTPLX Bare Speed 65.2, and Qwen 3.6 27B
Optimized Speed V2 59.9–60.1.

### 7.2 mlx-serve (native MTP, M4 Max 128 GB)

| Workload | MTP on | MTP off (`--no-mtp`) | Ratio | Source |
|---|---:|---:|---:|---|
| write a class from scratch (temp 0, median of 3) | **75.3 tok/s** | 26.3 tok/s | 2.86× | [ddalcu/Qwen3.8-27B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit/raw/main/README.md) |
| explanatory prose | 39.7 tok/s | 26.7 tok/s | 1.49× | same |
| 8-bit build, code | 53.6 tok/s | — | — | same card (8-bit = 31.2 GB, 0.71% weight noise vs 9.25%) |

`mlx-serve/benchmarks.md` decode-by-release (M4 Max 128 GB only, `llmprobe`, median of 3, fastest config):

| Model | 26.8.6 | 26.8.11 | 26.9.1 | 26.9.2 |
|---|---:|---:|---:|---:|
| **Qwen3.8 27B 4b (ddalcu MTP)** | · | 70 mtp | 71 mtp | 68 mtp |
| Qwen3.8 27B oQ4e (Jundot MTP) | · | · | 67 mtp | 61 mtp |
| Qwen3.6 35B-A3B 4b (MTP) | 191 mtp | · | · | 259 mtp |

Source: <https://raw.githubusercontent.com/ddalcu/mlx-serve/main/benchmarks.md>

### 7.3 omlx

- **M5 Pro (18 CPU / 20 GPU), 48 GB, macOS 26.4.1, oMLX 0.6.2, `root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp`,
  4-bit affine g64, Lightning MTP on, temp 0:** short 384-token runs **27.5 / 31.8 tok/s**, but real
  1,000–1,200-token responses sustained only **6.0–10.8 tok/s**, with MTP reporting `finish=parked` and
  69.8–83.7% draft acceptance. After the maintainer's fix (#3004, allow MTP to retry after a cooldown):
  **33.6 tok/s** on a 1,631-token response and **21.1 tok/s** on a 1,988-token response with a 27.5K-token
  prompt, MTP staying active. Source: <https://github.com/jundot/omlx/issues/2976>.
- **M5 Pro, 20-core GPU, 64 GB, oMLX 0.6.3rc3, `orcarouter/Qwen3.8-27B-Uncensored-MLX` 4-bit (15.70 GB),
  32K context, DFlash2 drafter `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB), block size 5, 1,400-token output:**
  plain decode **13.4–16.9 tok/s** (paired-test mean 16.4); DFlash2 **30.6 / 34.4 / 41.5 / 45.2 tok/s**
  (+86.8% to +171.5%). Chinese prose: **no stable benefit** (−3.81 … +0.07 tok/s, sign inconsistent).
  Sustained 5-run decay: plain 14.33→13.52 (−6%) vs DFlash2 40.34→21.38 (**−47%**); author recommends
  planning on **16–21 tok/s** for long sessions, not 40. MTP was **unavailable on this model** (weights
  lack `mtp.*`). Source: <http://reatang.com/p/qwen38-27b-omlx-dflash-mac/>.
- **M5 Max 128 GB, MLX 0.32.0 / mlx-lm 0.31.3, `tozp/Qwen3.8-27B-OBLITERATED-V2-mlx-4bit` target,
  greedy, 300 tokens, K=4, interleaved with 90 s cooldowns (mlx-lm-based DFlash2 server, not oMLX
  itself):** dense **33.3 tok/s**; + bf16 DFlash2 drafter 61.4 (1.84×, 88% accept); + quantized Q4
  drafter **76.1 tok/s (2.29×, 86.9% accept)**, 16.1 GB resident. Compare against oMLX on the same
  protocol: the same card reports `mlx-community/Qwen3.8-27B-mxfp4` at 35.4 dense. Source:
  <https://huggingface.co/KaedeTai/dflash2-mlx-quantized-draft/raw/main/README.md>.
- **omlx.ai benchmark pages** for "Qwen3.8-27B-oQ4e-mtp on M5 Pro (20c)", "Qwen3.8-27B-MTPLX-Optimized-Quality
  on M5 Pro (20c)", "Qwen3.8-27B-MLX-6bit on M5 Pro (20c)" exist in search indexes but currently return
  **503 maintenance** — the numeric values could not be retrieved. Re-check
  <https://omlx.ai/benchmarks/performance> later.

### 7.4 Practical expectation for an M5 Pro 20-core / 64 GB

Label this a **synthesis, not a measurement**: naive interpolation between the M5 Pro 24 GB D3 = 41.6
and M5 Max D3 = 58.0 rows puts a well-formed MTPLX 4-bit pack at roughly **40–50 tok/s sustained burst**
on an M5 Pro with a 4-bit MTP pack, tempered downward by the M5 Pro 64 GB oMLX report (16–21 tok/s
sustained for non-MTP decode, 21–34 tok/s for a healthy MTP path) and by thermal decay under continuous
load. **MTP roughly doubles to triples code decode and 1.5–1.7× prose decode** across every credible
source; the gain is much smaller on open-ended prose because acceptance falls from ~85–95% to ~38%.

---

## 8. Recommended configuration for the stated goal

**Primary recommendation: `mlx-serve` on port 11234, binding `0.0.0.0` with `--api-key`.**
It is the only runtime in this set that (a) supports the `qwen3_5` native MTP head automatically,
(b) loads MTPLX-branded sidecar packs with zero import step, (c) binds all interfaces by default, and
(d) needs no Python. Add `--kv-quant off` (default) unless memory-constrained, keep `--max-concurrent`
modest (default 1; raise for multi-client), and use `--prefix-cache-disk` for restart-persistent prefixes.

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve && brew install mlx-serve
sudo sysctl iogpu.wired_limit_mb=59392        # 58 GiB on a 64 GiB Mac; do not go near 65536

mlx-serve --model barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit \
  --serve --host 0.0.0.0 --port 11234 --api-key "$(openssl rand -hex 24)" \
  --mtp-depth 3 --max-concurrent 4 \
  --prefix-cache-mem 4GB --prefix-cache-disk 32GB
```

**Alternative if you want the richest serving stack (multi-model LRU, tiered SSD KV cache, embeddings +
rerankers, admin UI): `omlx`.** Requires enabling `mtp_enabled` per model and, for an MTPLX pack,
importing the sidecar first.

**Alternative if you want maximum verified MTP throughput on MTPLX's own artifacts: `MTPLX` itself** —
but note the LAN bind requires an API key and the license requires an in-product attribution.

---

## 9. Open questions / things I could not verify

1. **oMLX community benchmark numbers** — `omlx.ai/benchmarks` and all `/benchmarks/performance/<id>`
   pages return **HTTP 503** maintenance as of this research. Re-check.
2. **No published first-party M5 Pro 20-core / 64 GB MTP on/off measurement** for Qwen3.8-27B from
   MTPLX, mlx-serve, or omlx. §7.4 is extrapolation.
3. **No controlled MLX-vs-llama.cpp repetition-rate study for Qwen3.8.** The mechanism (mlx-lm's 20-token
   penalty window) is proven; the comparative claim is not.
4. **`repetition_context_size` exposure** in mlx-serve and MTPLX was not found in their docs/CLI
   references — verify in source before relying on it as the loop mitigation there.
5. **`iogpu.wired_limit_mb` persistence mechanism** across reboots was not source-verified in this pass.
6. **`PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed` byte-identity claim** to the official Youssofal
   artifact is asserted by the card only; no independent checksum comparison was performed.
