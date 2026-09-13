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
# Research index

Two independent deep-dives back the tuning decisions in this bundle. Both are
included in full because the reasoning is worth more than the conclusions — a
future MTPLX or llama.cpp release may invalidate a conclusion while leaving the
measurements valid.

| File | What it covers |
|---|---|
| [RESEARCH-MLX-STACK.md](RESEARCH-MLX-STACK.md) | MTPLX vs oMLX vs mlx-serve vs mlx-lm: MTP support, KV quantization, prompt caching, concurrency, LAN binding, wired-memory limits, and published tok/s across Apple Silicon. |
| [RESEARCH-LLAMACPP-MTP.md](RESEARCH-LLAMACPP-MTP.md) | llama.cpp's MTP support for the `qwen35` architecture: exact flags, build numbers, the Metal batch-amortisation problem, and the hybrid KV/recurrent-state memory math. |

---

## The findings that shaped this bundle

**1. llama.cpp's MTP is not worth enabling on Apple Silicon.**
`--spec-type draft-mtp` is the correct flag (`--mtp` is a download-only flag, a
common trap). It works. It is just slower: Metal does not amortise a 3–5 row
verify batch the way CUDA does, and PR #27776 measures this model's K-quant
matmuls dropping from 113 GB/s at batch 1 to 31.7 GB/s at batch 5 — which is
exactly the shape of an MTP verify batch. Reports range from −28% to +71%
depending on the Mac; the honest summary is *do not expect a speedup*. An MLX
runtime clears that bar by a wide margin, which is why this bundle is MLX-native.

**2. The MTP head is already inside the main GGUF — and the sidecar is a trap.**
The 15.7 GB `IQ4_XS` file contains `blk.64.*` MTP tensors, so no extra file is
needed. The 903 MB `FastMTP-32K.gguf` sidecar requires a *patched* llama.cpp
(pinned commit `4df29be4` plus a `d2t` draft-vocab trim) and has zero published
Apple Silicon numbers. It is not used here.

**3. The KV memory math, validated against three independent sources.**
`KV = 32,768 elements/token × n_ctx`, and the recurrent state is ≈150 MiB,
context-independent, multiplied by `(1 + n_max)` when MTP is on. These matched
measured logs to the byte (`512.00 MiB`, `144.00 MiB`, `KV 576`,
`recurrent 150→449`). The practical consequence — 262,144 context fits in ~24 GB
on a 64 GB Mac — is the reason `CONTEXT_WINDOW` defaults so high.

**4. Exactness is real.**
MTPLX accepts drafts through Leviathan–Chen rejection sampling with residual
correction, so `temperature=1.0, top_p=0.95` produces the same distribution as
ordinary decoding. The speedup does not cost you output quality. Note the
corollary: presence/frequency penalties default to 0 and are an *exact no-op*;
turning them on breaks that guarantee, so only do it to break a repetition loop.

---

## Where the research is uncertain

Reported honestly, because the difference matters when you are deciding whether
to trust a number:

- **The llama.cpp-on-Metal verdict is contested.** The upstream issue was closed
  as "not a bug" without a fix, and maintainers report users — including one of
  the llama.cpp maintainers — seeing real speedups on Macs. The conclusion here is
  "verify on your own hardware", not "impossible".
- **No first-party M5 Pro 20-core / 64 GB MTP measurement existed** before this
  bundle. The numbers in the README were measured here.
- **The MLX-vs-llama.cpp repetition-loop claim is not established.** The
  `repetition_penalty` inertness bug in mlx-lm (`context_size=20` truncation) was
  real and is fixed. But it is not evidence that MLX loops more than llama.cpp.
  What is true: this fine-tune's own conversion deliberately disables thinking by
  default to avoid greedy-decoding degeneration, and thinking is the setting most
  likely to produce a loop.
- **oMLX's community benchmark site was returning HTTP 503** throughout this
  research, so its published numbers could not be independently checked.

---

## Primary sources

- MTPLX — <https://github.com/youssofal/MTPLX>, docs at `docs/profiles.md`,
  `docs/server.md`, `docs/model-compatibility.md`
- llama.cpp MTP — PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)
  (merged as build `b9180`), flag documented in `docs/speculative.md`
- Metal batch amortisation — PRs [#27776](https://github.com/ggml-org/llama.cpp/pull/27776),
  [#27441](https://github.com/ggml-org/llama.cpp/pull/27441)
- Weight amortisation on Metal — issue [#23752](https://github.com/ggml-org/llama.cpp/issues/23752)
- The model — <https://huggingface.co/Qwen/Qwen3.8-27B>
- The MLX conversion — <https://huggingface.co/itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit>
- The GGUF fine-tune — <https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF>
- mlx-lm repetition window bug — <https://github.com/jundot/omlx/issues/3011>
