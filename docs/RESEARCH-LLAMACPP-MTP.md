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
# Qwen3.8-27B (`qwen35`) MTP on Apple Silicon — Technical Report

**Scope:** llama.cpp MTP (multi-token prediction / NextN) speculative decoding for the `qwen35` architecture, and what it actually does on Apple Silicon/Metal.
**Evidence date:** llama.cpp master as of 2026-09-12 (latest release `b10927`).
**Method note:** every claim below carries a URL. Claims I could not source are explicitly labelled **[SPECULATION]**, **[ESTIMATE]**, or **[INFERRED]**. Corrections made after initial drafting are marked **CORRECTION**.
**Bottom line up front:** the flag is `--spec-type draft-mtp` (not `--mtp`); the MTP head is already in your GGUF so **you do not need the 903 MB sidecar**; and on Apple Silicon the result ranges from **−28 % to +71 %** depending on chip, quant family, and `--spec-draft-n-max` — it must be measured, not assumed.

---

## 0. Model facts (verified from config + GGUF metadata)

| Property | Value | Source |
|---|---|---|
| HF `model_type` | `qwen3_5` | [Qwen/Qwen3.8-27B config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) |
| llama.cpp arch name | `qwen35` | [llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp) (`LLM_ARCH_QWEN35, "qwen35"`) |
| Layers | 64 (`case 64: type = LLM_TYPE_27B`) | [qwen35.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen35.cpp) |
| Layer layout | 16 × (3 × Gated DeltaNet → FFN, then 1 × Gated Attention → FFN) | [unsloth GGUF model card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/raw/main/README.md) |
| `full_attention_interval` | 4 → **16 full-attention layers**, 48 linear-attention (GDN) layers | config.json; [qwen35.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen35.cpp) (`uint32_t full_attn_interval = 4;`) |
| `num_key_value_heads` | 4 (`qwen35.attention.head_count_kv = 4`) | config.json; [issue #27282](https://github.com/ggml-org/llama.cpp/issues/27282) |
| `head_dim` (K/V length) | 256 | config.json |
| `hidden_size` / FFN | 5120 / 17408 | config.json |
| `vocab_size` | 248320 | config.json |
| MTP | `mtp_num_hidden_layers: 1`; `qwen35.nextn_predict_layers = 1` | config.json; [issue #27282](https://github.com/ggml-org/llama.cpp/issues/27282) |
| GDN dims | key head dim 128 × 16 heads, value head dim 128 × 48 heads, conv kernel 4 | config.json |

The MTP layer is **appended as block index 64** (`qwen35.block_count = 65` = 64 trunk + 1 MTP). This indexing matters for auto-detection (§3).

---

## 1. Which llama.cpp build added MTP for `qwen35`

### The exact commit

**PR [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673) — "llama + spec: MTP Support"**, merged **2026-05-16T12:06:24Z**, at merge commit `255582687b8dd211fdbc582e43ab842491554e94`.

### ⇒ **The exact build is `b9180`** — [release page with macOS arm64 + xcframework assets](https://github.com/ggml-org/llama.cpp/releases/tag/b9180)

This is **not an estimate**. The git tag `b9180` points directly at that merge commit:
```
$ git ls-remote --tags https://github.com/ggml-org/llama.cpp
255582687b8dd211fdbc582e43ab842491554e94  refs/tags/b9180
$ git log -1 b9180
255582687  b8d  2026-05-16 20:06:23 +0800  llama + spec: MTP Support  (#22673)
```

**The numbering scheme (reverse-engineered and verified): `build bN` == the number of commits in the master ancestry of the built commit** (`git rev-list --count <commit>`). Verified at `count(b9180)=9180`, `count(b10927)=10927`, `count(b10000)=10000`, `count(b7973)=7973`, `count(b9000)=9000`, `count(b6000)=6000`. Only very early `b1046` shows 1045 (±1 artifact).

**Consequence — and this resolves the "missing release tag" puzzle:** a release+tag is published only for the **head commit of a push batch**, so numbers belonging to non-head commits in the same batch are consumed but never tagged. The push that carried the merge had 6 commits (`42928bc #23046`, `1d9f99a #23134`, `366c5e2 #23127`, `1428004 #22736`, `b81c2cd #23117`, `2555826 #22673`) and the batch head got `b9180`; hence `b9175`–`b9179` have no tags. **That is why a naive `releases?per_page=100` scan shows gaps that do not mean those builds never existed**, and why `b10335`, `b10450`, `b10454`, `b10473` return 404 from `/releases/tags/`.

Same-day builds on 2026-05-16: `b9174` 00:02:40 · **`b9180` 12:06:23 (the merge)** · `b9181` 12:25:21 · `b9186` 13:11:29 · `b9189` 19:21:06 · `b9190` 21:42:16.

I also verified this is the qwen35 commit specifically, not just a generic MTP commit, by reading the commit history of `src/models/qwen35.cpp` via GitHub's atom feed (`https://github.com/ggml-org/llama.cpp/commits/master/src/models/qwen35.cpp.atom`):

```
2026-05-16 | llama + spec: MTP Support  (#22673)      <-- MTP enters qwen35.cpp
2026-05-17 | llama: avoid copying logits during prompt decode in MTP (#23198)
2026-05-19 | model : clarify MTP layer comment in qwen35.cpp (#23338)
2026-05-21 | mtp: use inp_out_ids for skipping logit computation (#23433)
2026-05-23 | model : add NVFP4 MTP scale tensors (#23563)
2026-05-29 | llama: add llm_graph_input_mtp (#23643)
2026-06-03 | qwen35: use post-norm hidden state for MTP (#24025)   <-- correctness fix
2026-07-31 | llama : load MTP tensors only if they are really used (#26296)
2026-09-01 | model : load hparams.n_layer_nextn before n_layer() calls (#28159)
2026-09-06 | models : fix GDN normalization from `max` to `rsqrt` (#28068)
```

PR #22673's own body states it was "tested on Qwen3.6 27B and Qwen3.6 35BA3B" and describes the design: *"The MTP model is a separate model which loads from the same GGUF, the idea is that MTP should automatically start and we shouldn't need to distribute the MTP gguf separately but also it has it's own context/kv-cache etc."*

**Key point:** `Qwen/Qwen3.6-27B` is *also* `qwen3_5` architecture (I fetched its config: `model_type: qwen3_5`, 64 layers, `full_attention_interval: 4`, `mtp_num_hidden_layers: 1`, https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json). So PR #22673's testing was already on the qwen35 architecture — there was no separate "qwen35 MTP" PR.

### Build number mapping

| Build | Release date | Source |
|---|---|---|
| `b9180` | **2026-05-16** | **PR #22673 merge — the build that added qwen35 MTP** (`git ls-remote` tag identity) |
| `b9239` | 2026-05-19 | #23338 MTP comment clarify (#23338's own count is 9236; first released build 9239) |
| `b9495` | 2026-06-03 | #24025 "qwen35: use post-norm hidden state for MTP" |
| `b9745` | 2026-06-21 | #24340 "spec : Support Step3.5/3.7 flash mtp3" — introduced the three-mode MTP driver |
| `b10034` | **2026-07-15** | GitHub Releases API `/releases/tags/b10034` |
| `b10068` | 2026-07-18 | API |
| `b10194` | ~2026-07-31 (est.) | [issue #27282](https://github.com/ggml-org/llama.cpp/issues/27282) reports `version: 10194` running qwen35 native MTP |
| `b10212` | 2026-07-31 | #26296 "load MTP tensors only if they are really used" |
| `b10358` | 2026-08-11 | API |
| `b10430` | 2026-08-14 | API |
| `b10431` | 2026-08-14 | #26623 ggml `SSM_SCAN` op rollback (Nemotron-motivated, not qwen35) |
| `b10453` | 2026-08-16 | API |
| `b10456` | 2026-08-17 | API |
| `b10927` | 2026-09-12 | API (**latest**; master tip is 3 commits ahead, count 10930) |

Build rate in the window: ≈ **14.7 builds/day** (1747 build numbers between `b9180` and `b10927` over 118.8 days).

⚠️ **Trap:** `GET /repos/ggml-org/llama.cpp/releases/latest` returns **`v0.4.0`**, not a `bNNNN` build. llama.cpp introduced semantic versioning on 2026-08-12 (`cmake : introduce semantic versioning (#26839)`) and now publishes semver tags alongside the b-stream (v0.1.0 → v0.4.0, published 2026-09-04). Do not use that endpoint to answer "latest build".

### qwen35 architecture support (for completeness)

| PR | Date | Build | What |
|---|---|---|---|
| [#19435](https://github.com/ggml-org/llama.cpp/pull/19435) | 2026-02-09 | b7973 | First Qwen3.5 dense+MoE attempt |
| [#19453](https://github.com/ggml-org/llama.cpp/pull/19453) | 2026-02-09 | b7976 | Immediate revert of #19435 |
| [#19468](https://github.com/ggml-org/llama.cpp/pull/19468) | 2026-02-10 | **b7990** | **"models : support qwen3.5 series" — registers `qwen35` AND `qwen35moe`** |
| [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) | 2026-05-16 | **b9180** | **MTP for qwen35/qwen35moe** |
| [#25589](https://github.com/ggml-org/llama.cpp/pull/25589) | 2026-08-03 | b10238 | MTP for the *Qwen3-Next* arch (`qwen3next`) — a different arch, not this one |
| [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) | 2026-08-27 | b10660 | `qwen4exp` = **Qwen3.8-Flash-Next**, a *different* architecture from Qwen3.8-27B |

### Verdict on your `b10034`

**`b10034` (2026-07-15) already supports `qwen35` MTP.** It is two months newer than the merge of PR #22673 and it includes the important correctness fix #24025 ("qwen35: use post-norm hidden state for MTP", 2026-06-03).

**A newer build is not *required*.** However:
- `b10034` predates #26296 (2026-07-31, "load MTP tensors only if they are really used") and #28159/#28068 (Sep 2026 correctness fixes). Rebuilding is recommended if only because, per the community tracking repo, "upstream is optimizing this arch weekly" ([qwen38-mtp rule 6](https://github.com/lsunay/qwen38-mtp)).
- ⚠️ `b10034` is **not** sufficient for the HauhauCS FastMTP sidecar — that needs a **custom patch** (§3).
- Caveat on build numbers: not every build number cited in the wild has a GitHub release object. `b10335`, `b10450`, `b10454`, `b10473` all return "Not Found" from `/releases/tags/<tag>`, even though other sources cite them. Build numbers printed by `llama-server --version` and release tags are not perfectly 1:1.

### Notable open bugs around qwen35 MTP (as of this writing)

- [#27282](https://github.com/ggml-org/llama.cpp/issues/27282) — native MTP reserves a **separate compute arena** and OOMs; at `n_ubatch = 512` the MTP context tried to `allocating 1040.28 MiB on device 0` and failed.
- [#27781](https://github.com/ggml-org/llama.cpp/issues/27781) — "speculative : fix shared-KV detection for draft-mtp contexts".
- [#28158](https://github.com/ggml-org/llama.cpp/issues/28158) — "Qwen3.8 DFlash/MTP speculative emits OOB token id == n_vocab (248320) on Vulkan".
- [#23577](https://github.com/ggml-org/llama.cpp/issues/23577) — "MTP with Qwen3.6 27B outputs repeated `////` after long session".
- [#26765](https://github.com/ggml-org/llama.cpp/issues/26765) — "qwen35: TENSOR_SKIP on MTP block triggers 'unused tensor' warnings".

---

## 2. Exact flags to enable MTP

### ⚠️ First, the important correction: the flag is `--spec-type draft-mtp`, **not** `--mtp`

`--mtp` **does exist** in llama.cpp but it is a **download-only** flag — it does not enable anything at serve time. From `common/arg.cpp` (master):

```cpp
add_opt(common_arg(
    {"--mtp"},
    "also download the multi-token prediction (MTP) head, if available (default: unused)",
    [](common_params & params) {
        params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }
).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
```

Note `.set_examples({LLAMA_EXAMPLE_DOWNLOAD})` — it is only valid for the download tool. Confirmed by the official docs, which list `--mtp` under none of the server/speculative option groups: [docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md).

### The official flag set

From [docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md) (verbatim):

```
--spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
--spec-draft-model, -md, --model-draft  FNAME
--spec-draft-n-max                      N     (default: 3)
--spec-draft-n-min                      N     (default: 0)
--spec-draft-p-split, --draft-p-split   P     (default: 0.10)
--spec-draft-p-min, --draft-p-min       P     (default: 0.00)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
--spec-draft-type-k, -ctkd, --cache-type-k-draft  TYPE
--spec-draft-type-v, -ctvd, --cache-type-v-draft  TYPE
```

The docs' type table says verbatim:

> | `draft-mtp` | Use Multi Token Prediction (MTP) heads **from the main model** |

There is **no dedicated `--mtp`-style "load a head from another file" flag**. The separate-file case is handled by `--spec-draft-model` (§3).

### The minimal working invocation (embedded MTP, no sidecar)

From [github.com/lsunay/qwen38-mtp](https://github.com/lsunay/qwen38-mtp) — note this repo's default branch is **`master`**, there is no `main` branch (`main` → HTTP 404). Its `serve_mtp.sh` is:

```bash
llama-server -m "$MODEL" \
  -c 131072 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 \
  --host 127.0.0.1 --port 8080
```

Its README distils the flag to:

> ```
> --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
> ```

Repo metadata (GitHub API) confirms the project name: *"One llama.cpp flag unlocks +33-39% decode speed for Qwen3.8-27B on consumer GPUs. The MTP head already ships inside your GGUF. Recipe, paired benchmarks, probe tool."* — created 2026-08-15, Apache-2.0, 0 stars, 53 benchmark rows from 40 contributors.

**Tuning guidance from that repo's seven rules** (all GPU numbers; see §4 for why this does not transfer to Apple):
1. `--spec-draft-n-max` sweet spot is card-dependent: 24 GB cards peak at 2, larger/faster cards at 3–4; re-sweep after any config change.
2. `--spec-draft-p-min` helps bandwidth-starved cards (0.60–0.75) and *hurts* fast ones. Sweep it, don't adopt it.
3. The gain scales with generation length; short generations can lose.
4. Multi-GPU: fix `--split-mode` first (`tensor` beat `layer` by 68% on its own for a 5060 Ti pair).
5. Speculative decode is a single-stream optimization — the advantage is gone by `--parallel 4`, and a `--parallel 2` baseline reads ~20% low. **Measure both arms at `--parallel 1`.**
6. Rebuild llama.cpp before tuning; upstream is improving this arch weekly.
7. **"A shared desktop halves everything, silently."** A live compositor/browser spilled weights to host RAM while `/health` stayed green.

Rule 7 is directly transferable to macOS: on a Mac, `iogpu.wired_limit_mb` and other resident processes are your equivalent hazard.

---

## 3. Separate draft GGUF vs. embedded head — both are real, and they are different paths

### Answer in one line

The MTP head is **normally embedded in the main GGUF** (`--spec-type draft-mtp` alone is enough). A **separate MTP-head GGUF is supported** and is loaded through the ordinary **`--spec-draft-model`** (draft-model) path — it is *not* a special `--mtp` head flag. Your 903 MB FastMTP file is a **third** thing: a custom draft sidecar with a 32 K reduced vocabulary that additionally **requires a patched llama.cpp**.

### Source-level proof (llama.cpp master)

`common/speculative.cpp` — `common_speculative_init_result`:

```cpp
std::string model_path;
if (has_draft) {
    model_path = params.speculative.draft.mparams.path;
    LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());
    llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
    ...
} else if (spec_mtp) {
    model_path = params.model.path;
    LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());
    llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
    ...
}
```

So MTP is implemented as **"a second context over a model"** — the draft context is built either from an explicit draft model file (`has_draft`, i.e. `--spec-draft-model`) or directly against the target model (`spec_mtp`). Either way it is a `draft-mtp` spec type with its own context and KV cache. `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP` is set when MTP is on.

**Auto-detection of a separate MTP GGUF** — `common_speculative_types_from_gguf()`:

```cpp
const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(),
    gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

if (gguf_find_tensor(gguf_ctx.get(),
        ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
    return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
}
```

Detection is on the presence of tensor **`blk.{block_count-1}.nextn.eh_proj.weight`**. For Qwen3.8-27B that is `blk.64.nextn.eh_proj.weight`. Note the inverse relation between `--spec-draft-model` and the MTP struct: because the patch reads `params.speculative.draft.mparams` via `has_dft()`, **`--spec-draft-model` is the mechanism** — there is no separate MTP file flag.

**Standalone (trunk-less) MTP GGUFs are explicitly supported** — `qwen35.cpp`:

```cpp
const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;
```

When the file has MTP layers but no block-0 attention norm, the trunk tensors become `TENSOR_NOT_REQUIRED` — i.e. a file containing *only* the MTP head loads fine as a draft. The `mtp_only` flag is also what the FastMTP d2t patch keys off.

**MTP gets its own KV cache, not the hybrid wrapper** — `llama-model.cpp`:

```cpp
// Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
const bool mtp_on_hybrid_qwen =
    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || ...);
...
if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
    const llama_memory_i::layer_filter_cb filter_mtp = [&](int32_t il) {
        return il >= (int32_t) hparams.n_layer();   // only the NextN layer(s)
    };
    res = new llama_kv_cache_iswa(...);
```

So the MTP context allocates KV for **only layer 64**, as a plain (non-hybrid) attention cache.

### Your specific files — exact identification

Your filenames match **`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`** exactly (repo, 2,078,044 downloads, tags include `mtp`, `speculative-decoding`, `fastmtp`):
- `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` — 15.71 GB ✓ (your "15.7 GB IQ4_XS")
- `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf` — 903 MB ✓

Independent GGUF header parse of the FastMTP sidecar (retrieved by HTTP range request) confirms your metadata and adds the tensor list:

- 19 tensors, `general.architecture = qwen35`, `general.file_type = 12`
- `general.description = "Optional FastMTP draft for the Qwen3.8-27B Aggressive variant by HauhauCS."`
- Tensors: `output.weight` **[5120, 32768] Q6_K**, `output_norm.weight`, `token_embd.weight` [5120, 248320] Q3_K, the whole `blk.64.*` MTP layer (`attn_*`, `ffn_*`, `nextn.*`, `post_attention_norm`), and **`d2t` [32768] I32**
- SHA-256 `115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b`

The main target GGUF by contrast has **866 tensors**, `qwen35.block_count = 65`, `nextn_predict_layers = 1`, full `blk.64.nextn.*`, `output.weight` [5120, **248320**], and **no `d2t`**.

### The decisive quote from the HauhauCS model card

> **"Every text GGUF preserves Qwen3.8's native NextN head, and this release adds HauhauCS FastMTP: a specific acceleration sidecar qualified across the complete quant lineup at maximum native context."**

> "There are two acceleration paths:
> - **Embedded MTP:** use any target GGUF by itself with `--spec-type draft-mtp` in a current upstream llama.cpp build.
> - **HauhauCS FastMTP:** pair that same target with `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf` and the HauhauCS runtime patch below."

Source: [HauhauCS README](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/raw/main/README.md)

**Therefore: you do not need the 903 MB file to use MTP.** Your IQ4_XS alone supports `--spec-type draft-mtp`.

### The documented FastMTP recipe (requires a patched build)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 4df29be4f4c3673f428170fda944a5b19f743bb8

curl -L -o HauhauCS-FastMTP-llama.cpp.patch \
  https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch
git apply --check HauhauCS-FastMTP-llama.cpp.patch
git apply HauhauCS-FastMTP-llama.cpp.patch

cmake -S . -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release   # -DGGML_METAL=ON is the Apple change from the doc's -DGGML_CUDA=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"

MODEL=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
DRAFT=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf

./build/bin/llama-server \
  --model "$MODEL" \
  --spec-draft-model "$DRAFT" \
  --spec-draft-ngl all \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --n-gpu-layers all --flash-attn on \
  --ctx-size 32768 --parallel 1 \
  --jinja --host 127.0.0.1 --port 8080
```

> ⚠️ The model card's `git clone https://github.com/ggerganov/llama.cpp` is a stale org name; upstream is now `ggml-org/llama.cpp`. Also note `--spec-draft-ngl all` and `--split-mode none` are the card's CUDA-shaped flags.

> ⚠️⚠️ **The card's recipe contains a REMOVED FLAG and will fail on a current build.** It passes `--no-mmap`, which no longer exists — it was replaced by `-lm/--load-mode`. Unknown arguments throw `error: invalid argument: <arg>` ([arg.cpp:824](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)), so this is a hard failure, not a silent no-op. **Replace `--no-mmap` with `-lm none`** (or just drop it — `auto` is the default and maps to mmap). The card was evidently written against the pinned `4df29be4f` build, which still had the old flag.

**Diagnostic for a missing patch:**

> "If draft loading reports `expected 5120, 248320, got 5120, 32768`, the FastMTP sidecar is correct but the executable is unpatched."

### What the patch actually does (I read it; SHA-256 verified)

File: `HauhauCS-FastMTP-llama.cpp.patch`, 2445 bytes, SHA-256 `981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615` (matches `FastMTP-PROVENANCE.json` exactly). **It touches only `src/models/qwen35.cpp`.**

It adds a **`d2t` draft-vocabulary trim**:
1. Reads the `d2t` tensor meta; if `mtp_only` and present, sets `n_vocab_out = d2t->ne[0]` (32768) and creates the `d2t` tensor.
2. Builds `output.weight` as `[n_embd, n_vocab_out]` instead of `[n_embd, n_vocab]` — so the draft LM head is 7.6× smaller (5120×32768 vs 5120×248320).
3. In the MTP graph, allocates a full-vocab `-INFINITY` logits tensor and scatters the 32768 draft logits into it with `ggml_set_rows` using `d2t` as the index map.

Provenance (`FastMTP-PROVENANCE.json`, schema `hauhaucs.fastmtp.provenance.v1`, Ed25519-signed):
- `runtime_base: ggerganov/llama.cpp@4df29be4f4c3673f428170fda944a5b19f743bb8` (≈ build `b10454` per third-party reports; no release object exists for that tag)
- `artifact` 903,453,952 bytes, `sha256: 115e618e…8968b` (matches)

The `d2t` concept is **already upstream**, used by EAGLE-3 — the official docs say EAGLE-3 "shares the target model's tokenizer and, optionally, uses a reduced draft vocabulary with its own `lm_head`, which is mapped back using a `d2t` table" ([docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)). `LLM_TENSOR_D2T` and `llama_model::d2t` exist in master ([llama-arch.h:696](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.h), [llama-model.h:675](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.h)). HauhauCS's patch is the qwen35 wiring.

Prior art: **PR [#20700](https://github.com/ggml-org/llama.cpp/pull/20700) "feat: MTP support for dense Qwen 3.5 with FastMTP vocabulary trimming"** — a 79 KB / 20-file PR that introduced `--spec-type mtp`. It was **closed unmerged**: current master's `common_speculative_type_from_name_map` accepts `draft-mtp` but **not** `mtp` (verified in [common/speculative.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/speculative.cpp)). This is why the mechanism ships as an out-of-tree patch rather than upstream.

### HauhauCS FastMTP claimed speedups (all NVIDIA — see §4)

| Comparison (final Q8_K_P) | Document TG | Reasoning TG |
|---|---:|---:|
| Standard embedded MTP vs MTP disabled (depth 2) | 2.23× | 1.60× |
| FastMTP vs standard embedded MTP (depth 3 vs 2) | +35.2% | +21.1% |
| **FastMTP vs MTP disabled (service)** | **3.02×** | **1.93×** |

Measured "on one RTX PRO 6000 Blackwell 96 GB per isolated lane at `204800` configured context, full CUDA offload, `--no-mmap`". Per-quant IQ4_XS row: PP 3445.30 tok/s, doc TG 211.09, reasoning TG 135.77. Ada RTX 6000 reference: FastMTP Q3_K_P → 138.37 doc / 87.95 reasoning TG. **There are zero Apple Silicon numbers in the card.**

---

## 4. Measured MTP speedup on Apple Silicon — genuinely inconsistent (−28 % to +71 %), mechanism now explained

**This is the headline finding and it inverts the marketing.** On Metal, MTP is slower than no MTP at every tested configuration.

### Primary evidence: upstream issue

**[ggml-org/llama.cpp#23752](https://github.com/ggml-org/llama.cpp/issues/23752) — "MTP speculative decoding degrades throughput on Metal (Apple Silicon) — net loss at every configuration"** (opened 2026-05-27, **closed**).

Hardware: MacBook Pro 14" 2021, **Apple M1 Max, 24-core GPU, 32 GB unified**, macOS 26.4.1. Build: `llama-server b9330 (328874d)`, AppleClang 21.0.0.21000099, Darwin arm64. Model: `Qwen3.5-9B-MTP Q4_K_M` (same `qwen35` arch family). 2048 tokens, temperature 0.

| Config | tok/s (think ON) | tok/s (think OFF) | Draft accept |
|---|---:|---:|---:|
| **Baseline (non-MTP model)** | **25.3** | **25.1** | — |
| `--spec-draft-n-max 0` | 22.4 (−11%) | 22.1 | 100% |
| `--spec-draft-n-max 2` | 21.9 (−13%) | 21.3 | 76% / 73% |
| `--spec-draft-n-max 6` | 19.3 (−24%) | 18.3 (−27%) | 44% / 41% |

Reporter's verbatim conclusions:
- "The draft evaluation overhead on Metal exceeds the speculative gain."
- "Ceiling (`n-max`) drives the regression. Floor (`n-min`) has no effect."
- "Higher ceiling = worse acceptance AND more overhead. No configuration beats baseline."
- First-bad-commit: **"Unknown. MTP on Metal, as a new feature, may never have provided a speedup. Not a regression from a known working state."**

**Note the `n-max 0` row**: even with *zero* drafts and *100 % acceptance*, Metal is 11 % slower. That is pure MTP-context overhead with no speculative benefit at all. Also from that issue's logs: `srv load_model: [spec] estimated memory usage of MTP context is 1808.02 MiB` — on a 9B model.

The same reporter filed a companion HF discussion with the same data ([Qwen3.5-9B discussion #56](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/56)) adding: "The 35B MoE variant (Qwen3.6-35B-A3B-MTP) shows **5–14× regression**."

### Corroborating: the community benchmark table

From [lsunay/qwen38-mtp README](https://github.com/lsunay/qwen38-mtp) (Qwen3.8-27B specifically):

| Machine | Baseline | With flag | n-max | Acceptance |
|---|---:|---:|---:|---|
| **Apple M4 24GB (Metal)** | 5.8 | **5.8** | 2 | 0.48–0.95 |
| **Mac Studio M3 Ultra 96GB (Metal)**, Q6_K | 22.8 | **24.2** | 2 | 0.53–0.95 (79.8%) |

Apple Silicon rows land at **0 % to +6.1 %**, versus +33 % to +145 % for the NVIDIA rows in the same table.

### ⭐ Important counter-datapoint: **+71 % on an M5 Max with llama.cpp**

This is the closest available hardware to an M5 Pro, and it is **positive** — which is why the "net loss" framing must not be treated as universal. From a community M5 Max benchmark ([KGP Talkie, Qwen3.8-27B on MacBook M5 Max](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-on-macbook-m5-max)), Q4_K_M, 8K context, best-of-3:

| Config | tok/s | vs baseline |
|---|---:|---:|
| MTP off | 18.2 | — |
| `--spec-draft-n-max 1` | 23.4 | +29 % |
| `--spec-draft-n-max 2` | 20.8 | +14 % |
| **`--spec-draft-n-max 3`** | **31.1** | **+71 %** (acceptance 0.575) |
| `--spec-draft-n-max 4` | 24.7 | +36 % |

Their recommended command:
```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -ngl 999 -fa on --jinja \
  -np 1 -c 8192 --spec-type draft-mtp --spec-draft-n-max 3
```

Note the shape: **the optimum is n-max 3, not 2 or 4** — matching `qwen38-mtp` rule 1 ("re-sweep after any config change"). The n-max 2 community default would have *underperformed* here (20.8 vs 31.1).

⚠️ This is a **single machine, single blog post**, with best-of-3 medians that the source itself notes were affected by swap stalls. Treat it as existence proof that llama.cpp MTP *can* pay on Apple Silicon — not as a number to expect.

**The same post independently validates this report's memory math** (§6). Its server log for M5 Max at 8K context reads:
```
llama_kv_cache: size = 512.00 MiB (8192 cells, 16 layers)
llama_memory_recurrent: size = 149.62 MiB ... S (f32): 144.00 MiB
```
My formulas predict: KV = 8192 × 65,536 B (f16) = **512.0 MiB** ✓, and `S` = 48 layers × 786,432 elements × 4 B = **144.0 MiB** ✓, recurrent total ≈ 149.6 MiB ✓. **Three independent quantities match to the digit**, from a completely separate source.

The same log also confirms the embedded head is present but dormant by default: `model has unused tensor blk.64.nextn.eh_proj.weight -- ignoring` — exactly the `load_mtp`/`TENSOR_SKIP` behaviour described in §3.

### The root-cause measurement (this is the valuable part)

From [sweeps/apple-silicon.md](https://raw.githubusercontent.com/lsunay/qwen38-mtp/master/sweeps/apple-silicon.md) — a matched two-backend control (same GGUF, flags, commit, operator):

| | baseline | n-max 2 | delta |
|---|---:|---:|---:|
| RTX 3090, run A | 43.8 | 73.9 | **+69%** |
| RTX 3090, run B | 43.2 | 78.5 | **+82%** |
| Apple M4, run A | 5.8 | 5.6 | −3.4% |
| Apple M4, run B | 5.8 | 5.9 | +1.7% |

Acceptance is comparable on both (0.46–0.93 CUDA vs 0.48–0.95 Metal) — **"so the drafts are equally good and only the verify differs."**

The mechanism, measured with `llama-batched-bench` (`-npp 128 -ntg 64 -npl 1,2,4,8 -fa 1 -ngl 999`), decode throughput at batch 8 versus batch 1 (8× would be perfect amortization):

| Model / format | Metal (M4, b10450) | CUDA (RTX 3090, master) |
|---|---:|---:|
| Qwen3-0.6B BF16 | 4.99× | 5.89× |
| Qwen3-0.6B Q4_0 | **1.66×** | 4.18× |
| Qwen3-0.6B Q4_K_M | **1.74×** | 3.85× |
| **Qwen3.8-27B (Q3_K_XL / Q4_0)** | **1.22×** | **3.34×** |

> "On the 27B, eight rows cost 6.54× one row on Metal versus 2.39× on CUDA. **Quantized decode on Metal barely amortizes across a batch; BF16 does.** That is the whole result: verification costs close to `n+1` full passes, so the drafts are paid for and nothing comes back."

The sweep names two source-level leads (explicitly flagged there as "leads, not a diagnosis"), both in `ggml/src/ggml-metal/ggml-metal-ops.cpp`: `ne11_mm_min = 8` gates the batched mat-mat kernel to `ne11 > 8`, and a spec-verify batch is only `n_draft + 1` rows (3 at n-max 2, 5 at n-max 4), so it never qualifies; and the small-batch mat-vec path admits F16/BF16/Q4_0/Q8_0/IQ4_NL at `ne11 >= 2` but K-quants only at `ne11 >= 4`.

#### ⭐ The mechanism is now confirmed, with numbers, **on this exact model class**

Two open upstream PRs contain direct `test-export-graph-ops` / `test-backend-ops` measurements that turn the above from a hypothesis into a measured fact.

**[PR #27776](https://github.com/ggml-org/llama.cpp/pull/27776)** ([Metal/Apple Silicon Performance] Disable K-quant mul_mv_ext for M3 Pro…) — benchmarks the **`qwen3.8:27b-mtp-q4_K_M`** blob on an **M3 Pro**, and the numbers are damning for exactly the ops MTP exercises:

| Op, M3 Pro | batch `ub=1` | batch `ub=5` (an n-max 4 verify batch) |
|---|---|---|
| `ffn_gate-0` q4_K [5120, 17408] | 412 µs/run @ **113 GB/s** | 1485 µs/run @ **31.7 GB/s** |
| `result_output` q6_K [5120, 248320] | 8607 µs @ **113 GB/s** | 27320 µs @ **35.7 GB/s** |

Per-row cost on `ffn_gate-0` goes 412 → 297 µs, i.e. **5 rows cost 3.6× one row** — only ~1.4× better per token. Bandwidth collapses from 113 GB/s to ~32 GB/s. **This is precisely "verification costs close to `n+1` full passes," measured on the same model, same quant family.** Note also that `result_output` is the 248,320-vocab LM head — the single biggest tensor in the MTP verify path.

**[PR #27441](https://github.com/ggml-org/llama.cpp/pull/27441)** (draft: skinny-batch mul_mm tiles) — M4 Pro, gemma4, q4_0, throughput vs batch size on stock master:

| batch | 4 | 5 | 6 | 8 | 10 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 94.6 | 91.0 | 102.9 | 108.4 | 147.4 | 221.4 |
| **per row** | **23.7** | **18.2** | **17.2** | **13.6** | 14.7 | 13.8 |

**Batch 8 is *worse per token* than batch 4**, and the real jump only arrives at `b ≥ 10`. With that PR's alternative kernel selection, per-row throughput is flat (27.2 / 25.0 / 26.0 / 24.8 / 23.8 / 23.2). The PR author's own comment: *"4-8 is speculative decoding sweet spot, so this efficiency would be needed there."* Its end-to-end spec-decode result on M4 Max improved **1.84× → 2.27×** once the ≥10-row verify path was fixed.

Current master thresholds (verified in source): `mul_mat_use_mm()` requires `ne00 >= 64 && ne11 > 8` ([ggml-metal-common.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal-common.cpp)); `mul_mv_ext` covers `ne11` **2–8** for F16/F32/Q4_0/Q8_0/IQ4_NL and **4–8** for K-quants.

**Why this bites Qwen3.8-27B specifically:** the n-max 2/3/4 verify batches used by MTP land at 3/4/5 rows — squarely inside the 2–8 dead zone. And critically, the MTP draft head's own LM head is `output.weight`, a K-quant in a K-quant GGUF (K-quants only get the 4–8 window), multiplied by the 248,320-row vocab. That combination is the worst case for this kernel family, which is exactly what the M3 Pro numbers show.

**Three salvage levers to consider [SPECULATION unless noted]:**
1. **Use non-K-quant cache/weights where possible** — Q4_0/IQ4_NL get the wider `ne11 ≥ 2` window. (This is plausibly why the sweep saw Q4_0 behave differently from Q3_K_XL in an earlier build.)
2. **Try `GGML_OP_OFFLOAD_MIN_BATCH=1`** if any weights are host-resident.
3. **Watch PRs [#27441](https://github.com/ggml-org/llama.cpp/pull/27441), [#27350](https://github.com/ggml-org/llama.cpp/pull/27350), [#27776](https://github.com/ggml-org/llama.cpp/pull/27776)** — **all three are still open/draft and unmerged as of 2026-09-12. Do not present any of them as available behaviour.** #27441 is the one that directly fixes the spec-decode window.

Disclosed nulls from the same sweep (do not over-read these):
- A claimed quant-family effect (Q3_K_XL −11.7 % vs Q4_0 +7.5 %) **did not reproduce** on stock b10450 (both −1.7 %) — "Not reproducible, so not a rule."
- PR [#23114](https://github.com/ggml-org/llama.cpp/pull/23114) ("metal: reuse K/V in flash-attn vec for spec-decode", closed unmerged) gave only +3.8 % — "Right subsystem, wrong term — flash attention is not where the cost is." **Correction:** the `GGML_METAL_FA_DISABLE_Q2` env var referenced by that sweep **does not exist in master** — it was a benchmark switch only on that closed PR's branch. The underlying idea did ship, in generalised form, as per-device tuned `(Q, NE)` tables ([PR #26570](https://github.com/ggml-org/llama.cpp/pull/26570) → `ggml-metal-tuning.cpp`).
- `n-max 4` was worse than 2 on both quants (−21 % on Q4_0).

**Context length makes Metal MTP worse, not better** (same sweep, M4, `--spec-draft-n-max 2`):

| context | `-b` | baseline | n-max 2 | delta |
|---|---:|---:|---:|---:|
| 8K | 512 | 6.0 | 5.9 | −1.7% |
| 32K | 512 | 5.8 | 5.6 / 5.9 | parity |
| 32K | 2048 | 5.9 | 5.4 | −8.5% |

### ⚠️ The MTP arm OOMs at llama.cpp's default batch size on a 24 GB Mac

Verbatim from the same sweep:

> On 24GB, `--spec-type draft-mtp` dies mid-generation at the default `-b 2048`:
> ```
> ggml_metal_synchronize: error: command buffer 1 failed with status 5
> error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
> ```
> "Load-time allocation is only 13.9 GB against a 21.5 GB `iogpu.wired_limit_mb`, so nothing warns you — it is the transient prompt-batch allocation that overflows, and it takes the server down with a `GGML_ASSERT` rather than an error response. **`-b 512 -ub 512` fixes it.**"

This is consistent with [#27282](https://github.com/ggml-org/llama.cpp/issues/27282): the MTP context reserves its own compute arena sized by `n_ubatch`.

### The counterpoint: a non-llama.cpp runtime *does* win on Apple

[weschera/Qwen3.8-27B-oMLX-MTP-Mac](https://raw.githubusercontent.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/main/README.md) — oMLX (MLX-based) + **ANE prefill**, M4 Max 128 GB, macOS 26, measured 2026-08-21, temperature 0, thinking off, 320 generated tokens:

| Config | Prose tok/s | Code tok/s |
|---|---:|---:|
| **oMLX 0.6.3rc2 + ANE prefill + native MTP, k=3** | **53.3** | **72.1** |
| oMLX 0.6.1 + native MTP, k=3 | 48.0 | 65.5 |
| oMLX 0.6.3rc2 + MTP k=3, ANE off | 47.9 | 47.9 |
| vllm-metal 0.3.0 (8-bit, no MTP) | — | 13.2 |

That is 2.2–2.8× the M4's llama.cpp number on the same model class. The repo also notes oMLX 0.6.3rc2 "fixes hybrid-model KV-cache sizing (4× overestimate that caused false OOM and prefill throttling on Qwen3.5/3.6/3.8)".

**Interpretation [SPECULATION]:** since a same-machine-class competing runtime gets large MTP wins while llama.cpp Metal gets none, this looks like a **llama.cpp Metal kernel gap** (specifically the non-amortizing small-batch verify path) rather than an Apple hardware limit. The `qwen38-mtp` Apple sweep lists this as its #1 open thread for exactly this reason.

### ⚖️ The counter-evidence: maintainers dispute this, and claims exist both ways

I want to be scrupulously fair here, because the two Apple reports were **closed without a fix** and the maintainers disagree with the conclusion. Read this as **"reported and contested," not "confirmed regression."**

**Maintainer response on [#23752](https://github.com/ggml-org/llama.cpp/issues/23752)** (closed 2026-05-27, 48 minutes after it was opened, as *completed*):
- **am17an**: "**This is not a bug**"; "MTP is optional. It is not baked in by default, so you can choose not to enable it."; "There are multiple users (**include Georgi himself**) which have reported a sizeable speed-up on Macs. I currently don't have the bandwidth to debug your particular setup."

So there are credible reports of Mac MTP speedups that I could not retrieve numbers for. My Apple conclusion rests on (a) issue #23752's M1 Max sweep, (b) the `qwen38-mtp` M4 and M3 Ultra rows (parity and +6.1 %), and (c) the `llama-batched-bench` amortization measurements. Three independent sources agree on "roughly neutral-to-negative," but this is **not unanimous**, and the deciding variable is very likely machine-specific (GPU core count / memory bandwidth / quant family).

**The other report, [#23011](https://github.com/ggml-org/llama.cpp/issues/23011)** — "Qwen3.6-35B-A3B self-MTP is much slower than baseline on Apple Metal despite high acceptance" (created 2026-05-13, closed 2026-05-14 as invalid):
- Apple M1 Pro 32 GB, version `b9117-ebe4fca4b`, baseline 26.23 tok/s → self-MTP **1.93 tok/s** (~13.6× slower) at ~95 % acceptance, with `warning: current allocated size is greater than the recommended max working set size`, and OOM at default parallelism (completes with `-np 1`).
- ⚠️ **This number should not be trusted as an MTP result**: `b9117` **predates the MTP merge (`b9180`)** — it was a PR-branch build. The maintainer closed it with: "Why is this an issue? **The feature is not even in llama.cpp yet.** Testing PRs is fine, but if you find a problem report it there, not here."
- Corroboration of the *pattern*, at least: brechtm commented on #23752 (2026-06-11) "also seeing consistently slower token generation speeds with Gemma 4 26B MTP enabled on M1 Max."

**No merged PR or release note fixing Metal MTP performance was found** in the window. Nearest adjacent Metal work: #26647 "ggml-metal: add chunked SSD MMA for Mamba-2 prefill optimization" and #26561 "metal: per-op source split + parallel compile".

### Practical verdict for Apple Silicon

**The honest summary: llama.cpp MTP on Apple Silicon is genuinely inconsistent — reported all the way from −28 % to +71 % — and the outcome is decided by your specific chip, quant family, and n-max.** The mechanism is now understood (§4.3), which makes the variance predictable rather than mysterious.

| Measured on | Result |
|---|---|
| M1 Max, Qwen3.5-9B Q4_K_M, b9330 | **−11 % to −28 %** at every n-max |
| Apple M4 24 GB, UD-Q3_K_XL | **parity** (code +9 %, prose −22 %) |
| Mac Studio M3 Ultra 96 GB, Q6_K | **+6.1 %** (n-max 2) |
| **M5 Max, Q4_K_M, 8K** | **+71 %** (n-max **3**) |
| MTPLX / mlx-serve (different runtime) | **+136 % to +186 %** |

1. **Do not assume either sign — measure.** The M5 Max result (+71 %, §4.2) rules out "MTP is broken on Metal" as a blanket claim, and the M1 Max result rules out "MTP works on Metal" as a blanket claim. Both are real.
2. **`--spec-draft-n-max` is the highest-leverage knob and the optimum is machine-specific**: 3 on M5 Max, 2 on M3 Ultra, ≤2 (or off) on M1 Max. Sweep 1→4. The upstream default of 3 is not a bad starting point.
3. **Quant family plausibly decides it**, per §4.3: K-quants only get the `mul_mv_ext` window at `ne11 ≥ 4`, Q4_0/IQ4_NL at `ne11 ≥ 2`. **A Q4_0-family or IQ4 quant may pay where a K-quant does not.** **[SPECULATION]** — untested end-to-end, but now grounded in measured kernel bandwidths.
4. **Pure code generation is where the gain lives** (+9–10 % on M4 code vs −22 % prose; prose acceptance falls to ~38 % vs 85–95 % on code). If your workload is prose/chat, expect much less or a loss.
5. **`-b 512 -ub 512` if you hit OOM** — the default `-b 2048` OOM-kills the server on a 24 GB Mac (§4.4).
6. ⭐ **If llama.cpp underdelivers on your machine, an MLX runtime is currently the reliable path to large MTP gains. See §4.6 and Recommendation D.**

### 4.6 ⭐ Decisive cross-validation: MLX runtimes get 2.4–2.9× MTP speedups on the *same* Apple hardware

This is the finding that resolves the whole question, and it strongly supports the "llama.cpp Metal kernel gap" reading over "Apple hardware can't do it."

Independent, first-party model cards for **MLX-family** runtimes serving this exact model (`qwen3_5`, 1-layer MTP head) report large MTP gains on the same class of Apple Silicon where llama.cpp reports none:

| Runtime / hardware | Quant | MTP off | MTP on | Speedup | Source |
|---|---|---:|---:|---:|---|
| **MTPLX** — M5 Pro, 20-core GPU, 24 GB | MTPLX 4-bit (16.9 GB) | 17.4 | **41.6** (depth 3) | **2.39×** | [barozp card](https://huggingface.co/barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit/raw/main/README.md) |
| **MTPLX** — M5 Max, 40-core GPU, 128 GB | MTPLX Optimized Speed 4-bit | 24.62 | **58.02** (depth 3) | **2.36×** | [PocketAiHub card](https://huggingface.co/PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed/raw/main/README.md) |
| **MTPLX** — unspecified M-series, 48 GB, **HauhauCS Aggressive** | MTPLX 4-bit | 14.9 | **37.3** (depth 2) | **2.51×** | [itrejomx card](https://huggingface.co/itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit/raw/main/README.md) |
| **mlx-serve** — M4 Max 128 GB, code task, temp 0, median of 3 | MLX 4-bit | 26.3 | **75.3** | **2.86×** | [ddalcu card](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit/raw/main/README.md) |
| **mlx-serve** — same, prose | MLX 4-bit | 26.7 | 39.7 | 1.49× | same |
| llama.cpp — M4 (comparison) | UD-Q3_K_XL | 5.8 | 5.8 | **1.00×** | [apple-silicon.md](https://raw.githubusercontent.com/lsunay/qwen38-mtp/master/sweeps/apple-silicon.md) |

`mlx-serve`'s own benchmark log (`https://raw.githubusercontent.com/ddalcu/mlx-serve/main/benchmarks.md`, M4 Max 128 GB, `llmprobe`, median of 3) is consistent across releases: Qwen3.8 27B 4-bit with MTP at **68–71 tok/s** across builds 26.8.11 → 26.9.2.

**Two things follow, and both are actionable:**
1. **The Apple Silicon limitation is runtime-specific, not hardware-specific.** An M4 Max gets 26.3 → 75.3 tok/s with MTP under mlx-serve while an M4 gets 5.8 → 5.8 under llama.cpp. That is direct evidence that llama.cpp's Metal *verify* path — not the Mac — is what fails to amortize, exactly as the `llama-batched-bench` numbers in §4.3 predict.
2. **The gain is code-shaped everywhere.** Prose gains ~1.5–1.7× vs code ~2.4–2.9×, because draft acceptance falls from ~85–95 % on code to ~38 % on prose. This matches llama.cpp's `+9 % code / −22 % prose` split in direction, just with a much larger positive code term.

⚠️ **Provenance caveat:** these are first-party model-card numbers with differing hardware, quants, protocols, and thermal handling — they are **not mutually comparable** and I did not reproduce them. Treat the *ratio* (MTP roughly doubles-to-triples code decode on MLX) as the durable claim, not any single tok/s figure.

**Also relevant to the HauhauCS line of work:** the same uncensored Aggressive model exists in MTPLX/MLX form with a working MTP head (`itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit`, with a separate `mtp.safetensors`). So if the goal is Mac MTP speed on the Aggressive variant specifically, the MLX pack is the path that has published numbers — the GGUF + FastMTP sidecar does not.

---

## 5. llama.cpp flags that matter for Apple Silicon on this model

### `--swa-full` is a **no-op** on Qwen3.8-27B

This is worth stating flatly because it is commonly recommended. `qwen35` never reads a sliding-window key and never sets `swa_type`:

- `hparams.swa_type = LLAMA_SWA_TYPE_NONE` and `hparams.n_swa = 0` are the defaults ([llama-hparams.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-hparams.h)).
- `src/models/qwen35.cpp` `load_arch_hparams()` reads `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`, `LLM_KV_ROPE_DIMENSION_SECTIONS` and the five SSM keys — **there is no `swa` reference anywhere in the file**.
- The SWA-aware hybrid memory (`llama_memory_hybrid_iswa`, which is where `params.swa_full` is passed) is only constructed `if (hparams.swa_type != LLAMA_SWA_TYPE_NONE)` ([llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp)). The plain `llama_memory_hybrid` constructor used by qwen35 does not receive `swa_full` at all.

So `--swa-full` changes nothing for this model. Source: [arg.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp).

### Flag-by-flag for M-series + hybrid models

| Flag | Status for Apple Silicon + qwen35 | Notes |
|---|---|---|
| `-ngl 999` / `-ngl all` | **Use.** | Metal is the only backend device; unified memory means "offload" is really "keep it in the shared pool". |
| `-fa on` / `--flash-attn on` | On. | Required in every published working recipe. Faithful attention here has 24 Q heads / 4 KV heads / head_dim 256. |
| `-ctk q4_0 -ctv q4_0` | **Strongly recommended.** | Only affects the 16 full-attention layers (§6). Used in every Apple row: M4 32K, M3 Ultra 131K, and the `serve_mtp.sh` reference. |
| `-ctk q8_0 -ctv q8_0` | Reasonable if memory allows. | The 2×24 GB test found "KV f16 was within 2% of q4_0 on all arms… q4_0 KV buys headroom, not speed". |
| `-ctk f16 -ctv f16` | Default; costs 2.67× q4_0 KV. | The HauhauCS card says "Keep default F16 K/V on the lower tiers unless memory pressure requires otherwise." |
| `-b` / `-ub` | **Critical with MTP.** Use `-b 512 -ub 512`. | Default `-b 2048` **OOM-kills the server** on a 24 GB Mac with `--spec-type draft-mtp`. MTP's compute arena scales with `n_ubatch` ([#27282](https://github.com/ggml-org/llama.cpp/issues/27282)). |
| `-nkvo` / `--no-kv-offload` | **Avoid.** | On Apple Silicon this pushes the KV cache to host buffers while weights stay on the GPU, adding per-token transfers. No published Apple recipe uses it. **[SPECULATION]** on the exact magnitude — not benchmarked by me. |
| `-ngld` / `--spec-draft-ngl all` | Use when using a draft sidecar. | From the FastMTP recipe. |
| `--spec-draft-n-max 2` | Start here if you test MTP at all. | Upstream default is **3**; the community sweet spot on constrained hardware is 2. |
| `--spec-draft-p-min` | Leave at default 0 for Metal. | Gating raised acceptance and *lowered* throughput on the M3 Ultra row (24.2 → 19.9 tok/s) and on fast NVIDIA cards. |
| ~~`--mlock`~~ | ❌ **THIS FLAG NO LONGER EXISTS.** | **CORRECTION** to an earlier draft of this report. Use `-lm/--load-mode`: `auto`\|`none`\|`mmap`\|`mlock`\|`mmap+mlock`\|`dio` ([arg.cpp:2687](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp), env `LLAMA_ARG_LOAD_MODE`). Verified: `grep '"--mlock"' common/arg.cpp` → **zero hits**. |
| ~~`--no-mmap`~~ | ❌ **THIS FLAG NO LONGER EXISTS.** | Same correction. The modern equivalent is **`-lm none`** (no mmap) or `-lm mmap`. `-lm mlock` / `-lm mmap+mlock` force the model to stay resident. The help text itself says mmap-disabled "may reduce pageouts if not using mlock". |
| `-t` / `--threads` | Default is already correct. | On `__APPLE__` the default reads `sysctlbyname("hw.perflevel0.physicalcpu")` — i.e. **performance cores only**, not all cores ([common.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/common.cpp)). With full offload, threads barely matter. |
| `--prio` | **Irrelevant to Metal decode.** | It sets `params.cpuparams.priority` — a **CPU thread** priority only. There is no GPU priority knob. |
| `--swa-full` | **No-op.** | See above. |
| `--op-offload` / `--no-op-offload` | Niche, but a real lever. | `ggml_backend_metal_device_offload_op()` returns true only for `MUL_MAT`/`MUL_MAT_ID` with batch (`ne[1]`; `ne[2]` for `MUL_MAT_ID`) `>= op_offload_min_batch_size` (**default 32**). The scheduler consults it **only when the op's weights sit in a host buffer** — so this decides whether such ops get *promoted* to the GPU or run on the CPU. A spec-verify batch is 3 rows, far below 32. **[SPECULATION]** that lowering it (`GGML_OP_OFFLOAD_MIN_BATCH=1`) helps — untested, and only relevant if weights are host-resident. |
| `-ot` / `--override-tensor` | Advanced. | Metal buffer types are `MTL0` (shared), `MTL0_Private`, `MTL0_Mapped`. There is **no** buffer type named "Metal"; `CPU` is the standard offload target. |
| `-nr` / `--no-repack` | ❌ **No effect on Metal.** | **CORRECTION** to an earlier draft. Repacking exists **only in the CPU backend** (`ggml-cpu.cpp` registers the repack buffer type); there is zero repack code under `ggml/src/ggml-metal/`. `-nr` only drops extra bufts from the CPU buft list. |
| `--no-host` | ~No-op on single-GPU Metal. | Metal's device registers `get_host_buffer_type = NULL`, so on a single MTL0 setup the host buft entry comes from another backend. **[INFERRED]** |
| `-dev` / `--device`, `-sm` / `--split-mode`, `-ts` / `--tensor-split` | Not useful. | Metal normally exposes exactly one device (`MTL0`); split mode is for multi-device only. |
| `-lzm` / `--lazy-mode` | New; leave default. | `on`\|`auto`\|`off` for on-demand tensor reads (requires mmap). |

⚠️ **There is no `--metal` CLI flag.** Metal is a build-time switch (`cmake -DGGML_METAL=OFF`). To force CPU inference use `--n-gpu-layers 0` or `--device none` ([docs/build.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md)).

### `GGML_METAL_*` environment variables (from source, `ggml/src/ggml-metal/`)

Extracted via `getenv()` from `ggml-metal-device.m`, `ggml-metal.cpp`, `ggml-metal-ops.cpp` — **this is the complete set, and they are undocumented in `docs/`**:

| Env var | Effect | Source |
|---|---|---|
| `GGML_METAL_PATH_RESOURCES` | Directory to load per-kind `kernels/*.metal` sources from instead of the embedded/precompiled metallib | ggml-metal-device.m:541 |
| `GGML_METAL_DEVICES` | Number of (virtual) Metal devices to expose, `MTL0..MTLn` | [ggml-metal.cpp:1014](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal.cpp) |
| `GGML_METAL_FUSION_DISABLE` | Disable Metal op fusion | [ggml-metal-device.m:1282](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal-device.m) |
| `GGML_METAL_FUSION_DEBUG` | Fusion debug level (integer) | same |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | Residency-set keep-alive seconds (**default 180**) | ggml-metal-device.m:963 |
| `GGML_METAL_NO_RESIDENCY` | Disable `MTLResidencySet` usage (on by default on macOS 15+) | ggml-metal-device.m:1252 |
| `GGML_METAL_BF16_DISABLE` | Disable BF16 support | ggml-metal-device.m:1127 |
| `GGML_METAL_TENSOR_DISABLE` / `GGML_METAL_TENSOR_ENABLE` | Metal 4 tensor API (MPP). **Default: disabled unless the device name contains M5/M6/A19/A20** — so it is already ON for an M5-series Mac | ggml-metal-device.m:1132/1141 |
| `GGML_METAL_SHARED_BUFFERS_DISABLE` / `..._ENABLE` | Force `MTLBuffer` private vs shared (shared is default on unified memory) | ggml-metal-device.m:1260/1263 |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` | Disable graph optimisation | ggml-metal-context.m:149 |
| `GGML_METAL_GRAPH_DEBUG` | Graph debug output | ggml-metal-context.m:143 |
| `GGML_METAL_CONCURRENCY_DISABLE` | Serialize command-buffer encoding | ggml-metal-context.m:140 |
| `GGML_METAL_CAPTURE_COMPUTE` | Metal capture scope (with Apple's `MTL_CAPTURE_ENABLED=1`) | ggml-metal-context.m:168 |
| `GGML_OP_OFFLOAD_MIN_BATCH` | Min batch for offloading `MUL_MAT`/`MUL_MAT_ID` from host buffers to Metal (**default 32**) — not Metal-prefixed but Metal-only | ggml-metal-device.m:1271, ggml-metal.cpp:758 |

**Negative findings (grep-verified against master, worth knowing so you don't chase ghosts):**
- `GGML_METAL_FA_DISABLE_Q2` **does NOT exist in master** — it was a benchmark switch only on the closed-unmerged PR #23114 branch. The idea shipped differently via per-device tuned `(Q, NE)` tables ([#26570](https://github.com/ggml-org/llama.cpp/pull/26570) → `ggml-metal-tuning.cpp`).
- `GGML_METAL_FUSE_GDN_CACHE` is **not an env var** — it is an enum constant (`GGML_METAL_FUSION_GDN_CACHE`).
- `GGML_METAL_MM_MIN`, `GGML_METAL_MV_EXT_MAX`, `GGML_METAL_MV_EXT_R1`, `GGML_METAL_MM_SPLITK`, `GGML_METAL_MM_SKINNY_*` exist **only on draft PR #27441's branch**, not in master.
- **No `GGML_METAL_*` env var is documented anywhere in `docs/`** — source is the only source of truth.

**Relevant to the MTP verify path [RETRIEVED]:** the per-device FA-vec tuning table (`ggml-metal-tuning.cpp`) picks, for `dk=dv=256` (this model's head dim), `{Q=1, NE=2}` for decode on M5, M5 Max, M4 and M3 Pro — i.e. **Q>1 multi-row amortization is *not* used for full-attention decode at head_dim 256 on these SKUs**. That is one more reason the verify batch does not amortize.
**Also [RETRIEVED]:** `-fa auto` is overridden to **ENABLED** when the V cache is quantized, and a quantized **K** cache requires `head_dim % block_size == 0`. Metal supports `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` KV with in-kernel dequant to f16 when the attention batch `ne[1] >= 32`.

On `GGML_OP_OFFLOAD_MIN_BATCH`: `get_op_batch_size()` returns `op->ne[1]` for `MUL_MAT`, and an op is only offloaded if that is `>= op_offload_min_batch_size` ([ggml-metal.cpp:769-775](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal.cpp)). This gate applies to ops whose tensors live in **host/CPU buffers** (the scheduler's fallback path), not to a fully-offloaded model whose weights are already in Metal buffers. A spec-verify batch is only `n_max + 1` = 3 rows, so **if** any weights were in host buffers those matmuls would run on CPU — but for a normal `-ngl all` Metal setup this gate is not on the critical path. **[SPECULATION]** — worth a controlled test (`GGML_OP_OFFLOAD_MIN_BATCH=1`), but I have not measured it and I do not claim it is the fix.

Related open Metal PRs worth watching — **all three are still OPEN/DRAFT and unmerged as of 2026-09-12; do not treat any as available behaviour:**
- [#27441](https://github.com/ggml-org/llama.cpp/pull/27441) — "metal : skinny-batch mul_mm tiles + deterministic split-K for small-batch decode" (**draft**) — the one that directly targets the `ne11` 4–16 spec-decode window. Measured end-to-end spec-decode gain on M4 Max: **1.84× → 2.27×**.
- [#27350](https://github.com/ggml-org/llama.cpp/pull/27350) — "metal : add i-quant support to mul_mv_ext (small-batch decode path)" (**draft**) — adds IQ2/IQ3/IQ4_XS to the `ne11 4..8` K-quant group. No benchmarks in the PR body.
- [#27776](https://github.com/ggml-org/llama.cpp/pull/27776) — "[Metal/Apple Silicon Performance] Disable K-quant mul_mv_ext for M3 Pro…" (**open**) — adds a per-device disable table; source of the M3 Pro bandwidth numbers in §4.3.
- [#28210](https://github.com/ggml-org/llama.cpp/pull/28210) — "ggml-metal: fix int32 overflow in kernel_mul_mm batched dst offsets (deep-context corruption)" (closed) — relevant if you run very long contexts.

Also Metal-relevant and **already merged** (in `b10909`, 2026-09-11): "metal : single-source fusion table + fusion debug rework (#28164)", which fixed a bug that "silently disabled the norm/MUL fusion and caused a ~5% token-generation regression" **and added a `GATED_DELTA_NET + CPY` fusion** — recurrent-state snapshots are now written straight into the cache buffer with the trailing copy elided. That is directly relevant to qwen35's 48 linear-attention layers and to MTP's snapshot machinery. **If you are on an older build, this alone is a reason to update.**

### macOS system-level tuning

The one setting that appears in every Apple row is the **GPU wired-memory limit**. On macOS Sonoma (14)+ the unit is MB:

```bash
# inspect the current cap (0 = system default)
sysctl iogpu.wired_limit_mb

# raise it — e.g. 120 GB on a 128 GB machine; does not need a reboot
sudo sysctl iogpu.wired_limit_mb=122880

# reset to the system default
sudo sysctl iogpu.wired_limit_mb=0
```

Older macOS used **bytes** under a different key: `sudo sysctl debug.iogpu.wired_limit=60129542144`. Persist via `/etc/sysctl.conf` (may require disabling SIP). Guidance from [ivanopcode/devnote-override-macos-metal-vram-cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap) (derived from [stencel.io](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)): leave 8–16 GB headroom — 32 GB → 28–30 GB, 64 GB → 56 GB, 128 GB → 120 GB.

**How to verify the cap llama.cpp actually sees:** it logs `recommendedMaxWorkingSetSize` at startup ([ggml-metal-device.m](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal-device.m)) — that is the number that matters, and it is often well below physical RAM (a 36 GB M5 Max reports ~28.7 GiB usable by MTL0). The M4 24 GB row reports a default of **18000 MB** and calls it "too low for a 12.5 GiB model plus spec contexts"; that reporter set 21504.

**`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`:** this env var works around the macOS GPU interactivity watchdog failing command buffers with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` / `...InnocentVictim` ([issue #20441-style reports](https://github.com/ggml-org/llama.cpp/issues/20141), referencing [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267)). **You no longer need to set it manually** — current master sets it for you at backend registration: `setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true)` with a comment explaining it is a macOS limitation workaround ([ggml-metal.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-metal/ggml-metal.cpp)). If you are on an older build and seeing those errors, exporting it yourself is the fix.

macOS has no equivalent of CUDA persistence mode; the two settings above are the practical set. **[INFERRED]**

---

## 6. KV cache and recurrent-state memory for this hybrid model

### 6.1 The KV cache formula (only 16 of 64 layers have one)

`llama_memory_hybrid` splits memory into an attention KV cache filtered to non-recurrent layers and a recurrent state for the GDN layers ([llama-memory-hybrid.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-memory-hybrid.cpp)). `qwen35.cpp` marks recurrent layers as:

```cpp
// Mark recurrent layers (linear attention layers). MTP layers are dense
// attention-only and must be flagged non-recurrent.
if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
    }
}
```

→ full attention at block indices **3, 7, 11, … 63** = 16 layers.

**Formula:**

```
KV bytes = 16 layers × (K + V) × head_count_kv(4) × head_dim(256) × n_ctx × bytes_per_element
         = 32,768 elements/token × n_ctx × bytes_per_element
```

Bytes/row: **f16 = 2.00** → 65,536 B/token (64 KiB) · **q8_0 = 1.0625** → 34,816 B/token (34 KiB) · **q4_0 = 0.5625** → 18,432 B/token (18 KiB).

| Context | f16 | q8_0 | q4_0 |
|---:|---:|---:|---:|
| 8,192 | 0.50 GiB | 0.27 GiB | 0.14 GiB |
| 32,768 | 2.00 GiB | 1.06 GiB | **0.56 GiB** |
| 65,536 | 4.00 GiB | 2.13 GiB | 1.13 GiB |
| 131,072 | 8.00 GiB | 4.25 GiB | 2.25 GiB |
| 204,800 | 12.50 GiB | 6.64 GiB | 3.52 GiB |
| 262,144 | 16.00 GiB | 8.50 GiB | 4.50 GiB |

**Validated against real measurements:**
- Apple M4 24 GB row, 32K context, q4_0 KV logs **`KV 576`** MiB. Formula: 32,768 × 18,432 B = 603,979,776 B = **576.0 MiB**. ✓ *Exact.*
- Mac Studio M3 Ultra, 131K context, q4_0 KV, Q6_K (21 GB file): predicted KV = 2.25 GiB; process RSS reported 23.20 GiB ≈ 21 GB model + 2.25 GiB KV + 0.15 GiB recurrent. ✓

Because only full-attention layers are counted, **KV here costs ~1/(4×)** what a same-shape all-attention 64-layer model would — this is the single biggest memory advantage of the hybrid design, and it is why 131K–262K context fits on a Mac.

### 6.2 Recurrent (Gated DeltaNet) state — how llama.cpp handles it

The GDN layers do **not** have a KV cache. They carry a fixed-size recurrent state (plus a short conv history), allocated by `llama_memory_recurrent` with three tensor sets: `r_l` (recurrent state), `s_l`, and `p_l` ([llama-memory-recurrent.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-memory-recurrent.h)).

Sizing is the sum of `ggml_nbytes()` over those tensors, logged as:

```
"%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): ... S (%s): ... P (%s): ..."
```

**Derivation for Qwen3.8-27B** (48 linear layers):

| Component | Per layer | × 48 layers |
|---|---|---|
| Recurrent state `S` = `head_k_dim × head_v_dim × n_v_heads` = 128 × 128 × 48 = 786,432 elements × 4 B (F32) | 3.00 MiB | 144.0 MiB |
| Conv state = `(conv_kernel−1) × conv_channels` = 3 × (6144 + 2×16×128 = 10,240) = 30,720 elements × 4 B | 0.117 MiB | 5.6 MiB |
| **Total** | | **≈ 149.6 MiB** |

**Validated:** the Apple M4 sweep logs **`recurrent state 150`** MiB. ✓ *Exact.*

**Key property: the recurrent state is context-length independent.** It is 150 MiB at 8K, 32K, or 262K. Only the attention KV grows with context.

### 6.3 ⚠️ MTP multiplies the recurrent state by `(1 + n_max)`

This is the subtle, undocumented interaction, and it is the thing to know before enabling MTP on a hybrid model.

`common.h`:
```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
        return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || ...;
    });
    return needs_rs_seq ? draft.n_max : 0u;
}
```

`llama-memory-recurrent.h`:
```cpp
// number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
uint32_t n_rs_seq = 0;
```

That value is threaded through as `cparams.n_rs_seq` into both `llama_memory_recurrent` and `llama_memory_hybrid` ([llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp)).

**Therefore:** `recurrent_state ≈ 150 MiB × (1 + --spec-draft-n-max)`.

| `--spec-draft-n-max` | `n_rs_seq` | Recurrent state |
|---:|---:|---:|
| off | 0 | **150 MiB** |
| 1 | 1 | 300 MiB |
| **2** | 2 | **450 MiB** |
| 4 | 4 | 750 MiB |
| 8 | 8 | 1,350 MiB |

**Validated:** the Apple M4 sweep logs **`recurrent state 150→449`** for baseline→n-max 2. ✓ 150 × 3 = 450 ≈ 449. *Exact.*

The purpose is draft rollback: rejected drafts must rewind the GDN state, which requires keeping `n_max` prior snapshots. `llm_arch_supports_rs_rollback()` includes `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` ([llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp)). Server guards fall back to full checkpoints when `draft.size() > llama_n_rs_seq(ctx)` (`tools/server/server-context.cpp:3073`) or `n_rollback > llama_n_rs_seq(ctx)` (`:3928`).

**Provenance correction (I initially mis-attributed this):** for **qwen35's** DeltaNet/GDN hybrid layers the rollback shipped *in PR #22673 itself* (`b9180`) — #22673 added `cparams.n_rs_seq`, `llama_n_rs_seq()`, `need_n_rs_seq()`, the `COMMON_CONTEXT_SEQ_RM_TYPE_RS` seq_rm mode, and `tests/test-recurrent-state-rollback.cpp`. At `b9180`, `llm_arch_supports_rs_rollback()` returns true for **only** `QWEN35` and `QWEN35MOE`.

Two separate things are often conflated here:
- The design PR [#22400](https://github.com/ggml-org/llama.cpp/pull/22400) ("llama: allow partial seq_rm for GDN models for speculative decoding", opened 2026-04-26) is **closed unmerged**, and the branch commits `gg/metal-gdn-partial-rollback` (`4e732e0`, `6eb6d84 "metal: add GDN partial rollback"`, `8c05923 "vulkan: add GDN partial rollback"`, all 2026-05-14) are **not ancestors of master** — they never merged as-is. The shipped version arrived via #22673.
- The **ggml `SSM_SCAN` op-level rollback** is a *later, unrelated* PR: [#26623](https://github.com/ggml-org/llama.cpp/pull/26623) "Initial changes for Recurrent state rollback for nemotron for cuda", merged 2026-08-14 → **`b10431`**. Its motivation is **Nemotron/Mamba-2, not qwen35**. So the widely-repeated claim that "an `ssm_scan` state-rollback is what makes deep MTP drafts work on the DeltaNet layers" is **wrong for qwen35** — qwen35's rollback does not route through `ggml_ssm_scan`.

Later `llm_arch_supports_rs_rollback` extensions: `qwen4exp` #28123 (b10731), Kimi-K3 #28466 (b10853); master now also lists DEEPSEEK4, NEMOTRON_H(_MOE), LFM2(_MOE), BAILINGMOE3.

Also worth knowing: the "single trained MTP head" driver comment at `common/speculative.cpp:1345` (`// neither (qwen35 / qwen35moe): a single trained MTP head.`) was introduced by **[#24340](https://github.com/ggml-org/llama.cpp/pull/24340) "spec : Support Step3.5/3.7 flash mtp3"**, merged 2026-06-21 → **`b9745`** — *not* by #23338. #24340 added the three-mode driver (`is_mem_shared` gemma4 / `chain_heads` step35 / neither = qwen35), i.e. it formalised how the qwen35 single-shared-head path is reused across draft steps.

### 6.4 Is there a flag to limit the recurrent state?

**No.** There is **no CLI flag** for the recurrent state's size or dtype:

- The dtype is **hardcoded to F32** for both `type_r` and `type_s` in every construction site in `llama-model.cpp` (`GGML_TYPE_F32, GGML_TYPE_F32`) — including the `llama_memory_hybrid` and `llama_memory_hybrid_iswa` constructors, which also hardcode the recurrent KV type to F32.
- `common/arg.cpp` exposes exactly four cache-type flags — `-ctk/--cache-type-k`, `-ctv/--cache-type-v`, `--spec-draft-type-k`, `--spec-draft-type-v` — **none of which touch the recurrent state.** I grepped the whole argument table for `cache-type-r`, `type-r`, `-ctr`, `--state`; there is nothing.
- `--swa-full` is a no-op here (§5).
- `llama-memory-hybrid.cpp` is explicit: `// Shifting is trivially supported for recurrent`.

**The only lever that changes recurrent-state memory is `--spec-draft-n-max`.** Lowering it from 3 (upstream default) to 2 or 1 directly shrinks the state.

### 6.5 Total memory budget — worked example (32K context)

Using your IQ4_XS (≈15.71 GB) at `-c 32768`, MTP off vs. on:

| Component | MTP off | MTP on, `n-max 2` |
|---|---:|---:|
| Weights (IQ4_XS) | 15.71 GB | 15.71 GB |
| Attention KV (16 layers, q4_0) | 0.56 GiB | 0.56 GiB |
| Recurrent state (48 GDN layers, F32) | 0.15 GiB | **0.45 GiB** |
| MTP draft KV (1 layer, q4_0) | — | ~0.04 GiB |
| MTP compute arena (scales with `-ub`) | — | **0.5–1.8 GiB** (measured range: 1808 MiB on a 9B model at large ubatch, [#23752](https://github.com/ggml-org/llama.cpp/issues/23752); "allocating 1040.28 MiB" at ubatch 512, [#27282](https://github.com/ggml-org/llama.cpp/issues/27282)) |
| **Approx. total** | **~16.4 GB** | **~17.2–17.6 GB** |

For reference, the M4 24 GB row measured 13.2 GB baseline → 13.9 GB with spec. The delta is small at load **but the transient compute arena is not** — that is exactly what OOMs at `-b 2048`.

**Note on the `-bs`/`-ub` interaction:** both the MTP compute arena and the prompt-batch allocation scale with batch size, so `-ub` is the single most effective memory knob for a Mac running MTP.

---

## 7. Recommended configuration for an M-series Mac

### Recommendation A — highest throughput (MTP off)

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf \
  -ngl 999 -fa on \
  -c 32768 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -b 2048 -ub 512 \
  --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
```

Expected: matches the published Apple rows (M4 24 GB ≈ 5.8 tok/s; M3 Ultra 96 GB ≈ 22.8 tok/s at Q6_K/131K).

### Recommendation B — MTP on Metal, tuned for an M5-class Mac

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf \
  -ngl 999 -fa on \
  -c 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
  -b 512 -ub 512 \
  --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
```

Rationale, in order of how much it matters:
- **`--spec-draft-n-max 3`** — this is what produced **+71 %** on an M5 Max (§4.2). Do not blindly copy the n-max 2 default from NVIDIA-oriented guides.
- **Sweep n-max 1→4.** The optimum is machine-specific (3 on M5 Max, 2 on M3 Ultra, ≤2/off on M1 Max). There is no universal value.
- **`-b 512 -ub 512`** — defensive. The default `-b 2048` OOM-kills the server on a 24 GB Mac (§4.4). If you have headroom, a larger `-ub` is fine.
- Note the M5 Max reference ran at `-c 8192`. Shorter context helps because verify cost grows with KV.
- If this underperforms, **try a Q4_0-family or IQ4 quant rather than a K-quant** — §4.3 explains why K-quants get the narrower kernel window.

Then measure honestly:
```bash
# both arms, same prompts, --parallel 1
python3 probe.py http://127.0.0.1:8080     # from github.com/lsunay/qwen38-mtp
```

No `--spec-draft-model` is needed — the MTP head is embedded in the IQ4_XS, and llama.cpp will log `creating MTP draft context against the target model`.

### Recommendation C — the FastMTP sidecar (only if you accept a patched build)

Requires the pinned commit + HauhauCS patch (§3), and **no Apple Silicon speedup has ever been published for it.** The pinned base is from 2026-08 (~b10454) and would not include the Aug/Sep upstream qwen35 fixes (#26296, #28159, #28068). **I would not recommend this on a Mac today.**

### ⭐ Recommendation D — the configuration that actually delivers MTP speed on Apple Silicon

**If the goal is *fast* Qwen3.8-27B *with* MTP on an M-series Mac, change runtime rather than fight llama.cpp's Metal verify path.** §4.6 shows MLX-family runtimes getting **2.36–2.86×** on the same hardware class where llama.cpp gets 1.00×:

| Runtime | Hardware | MTP off → on | Speedup |
|---|---|---:|---:|
| MTPLX | M5 Pro 20-core GPU, 24 GB | 17.4 → 41.6 tok/s | 2.39× |
| MTPLX | M5 Max 40-core, 128 GB | 24.62 → 58.02 tok/s | 2.36× |
| mlx-serve | M4 Max 128 GB (code) | 26.3 → 75.3 tok/s | 2.86× |
| **llama.cpp** | **M4** | **5.8 → 5.8 tok/s** | **1.00×** |

For the HauhauCS Aggressive variant specifically, a pack already exists with a working MTP head:
`itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit` (4-bit MTPLX, separate `mtp.safetensors`; published 14.9 → 37.3 tok/s, 2.51× at depth 2). Candidate runtimes: `mtplx`, `mlx-serve`, `omlx`.

⚠️ I did **not** benchmark these myself; the numbers are from first-party model cards with differing hardware, quants, and protocols, and are not mutually comparable. Verify on your own machine.

---

## 8. Open questions / what I could not source

1. **No Apple Silicon benchmark of the HauhauCS FastMTP sidecar exists** anywhere I could find. All its numbers are RTX PRO 6000 Blackwell / RTX 6000 Ada.
2. ~~No M5-generation measurement of Qwen3.8-27B MTP in llama.cpp.~~ **PARTIALLY RESOLVED:** a community M5 Max llama.cpp run is now included (§4.2, +71 % at n-max 3). Still **no M5 Pro measurement**, and no independent replication — it is a single blog post. **A controlled MTP-on/off A/B on an M5 Pro with llama.cpp is the single most valuable missing experiment.**
3. ~~PR #22673's exact build number is an estimate.~~ **RESOLVED: it is `b9180`**, verified two independent ways (tag→commit identity, and `bN == git rev-list --count`). The `b10335`/`b10450`/`b10454`/`b10473` 404s are **explained, not a gap**: tags are published only for push-batch heads, so non-head build numbers never get a tag.
4. **Whether `GGML_OP_OFFLOAD_MIN_BATCH=1` helps Metal spec-decode** is untested by me. (`-nr`/`--no-repack` is now ruled out — CPU-backend only, no Metal effect.) The genuinely promising fixes are all unmerged upstream kernel work: [#27441](https://github.com/ggml-org/llama.cpp/pull/27441) (draft, the one that matters), [#27350](https://github.com/ggml-org/llama.cpp/pull/27350) (draft), [#27776](https://github.com/ggml-org/llama.cpp/pull/27776) (open).
5. **Issue [#23752](https://github.com/ggml-org/llama.cpp/issues/23752) was closed 48 minutes after it was opened, with am17an stating "This is not a bug"** and citing other Mac users (including Georgi) who report MTP speedups. The M5 Max row in §4.2 is evidence he was right that Mac speedups exist. **The Apple conclusion is therefore "highly machine-dependent, spanning −28 % to +71 %" — not a settled sign.**
6. **`-fa on` and MTP:** I found no source-level requirement that MTP needs flash attention, despite every published recipe including it. Treat `-fa on` as best practice, not a dependency. (Note `-fa auto` becomes ENABLED anyway when the V cache is quantized.)
7. **The quant-family hypothesis is untested end-to-end.** §4.3 predicts a Q4_0/IQ4-family quant should beat a K-quant for MTP verify on Metal, because of the `ne11` window difference. The kernel-level bandwidths are measured; the end-to-end consequence is not. **This is the cheapest high-value experiment for a Mac owner.**
8. **I did not benchmark anything myself.** Every number here is either from llama.cpp source, upstream issues/PRs, or third-party reports, each cited. The memory formulas in §6 are the exception in spirit — I derived them from source and they matched three independent log lines to the digit, which is why I state them as fact rather than report.

---

## Appendix: source index

**llama.cpp source (master, 2026-09-12)**
- `src/models/qwen35.cpp` — arch hparams, `mtp_only`, MTP layer tensors, GDN graph
- `common/arg.cpp` — `--mtp` (download only), `--spec-type`, `--spec-draft-*`, `--swa-full`, `-ctk/-ctv`, `-b/-ub`, `--prio`
- `common/speculative.cpp` — `draft-mtp` impl, `common_speculative_types_from_gguf`, context creation
- `common/common.h` — `common_speculative_type`, `need_n_rs_seq()`
- `src/llama-memory-recurrent.h` / `.cpp` — `n_rs_seq`, state sizing/logging
- `src/llama-memory-hybrid.cpp` — hybrid attention+recurrent split
- `src/llama-model.cpp` — memory construction, hardcoded `GGML_TYPE_F32` recurrent types, `mtp_on_hybrid_qwen`
- `src/llama-arch.cpp` — `qwen35`, `llm_arch_supports_rs_rollback`
- `src/llama-hparams.h` — `swa_type`/`n_swa` defaults
- `ggml/src/ggml-metal/ggml-metal-device.m`, `ggml-metal.cpp` — `GGML_METAL_*` env vars
- `docs/speculative.md` — official flag reference

**PRs / issues**
- [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) **MTP Support, merged 2026-05-16 = build `b9180`** ([release](https://github.com/ggml-org/llama.cpp/releases/tag/b9180)) · [#19468](https://github.com/ggml-org/llama.cpp/pull/19468) qwen3.5 series (b7990) · [#24340](https://github.com/ggml-org/llama.cpp/pull/24340) three-mode MTP driver (b9745) · [#24025](https://github.com/ggml-org/llama.cpp/pull/24025) qwen35 post-norm MTP fix (b9495) · [#26296](https://github.com/ggml-org/llama.cpp/pull/26296) load MTP tensors only if used (b10212) · [#26623](https://github.com/ggml-org/llama.cpp/pull/26623) ggml SSM_SCAN rollback, Nemotron (b10431) · [#20700](https://github.com/ggml-org/llama.cpp/pull/20700) FastMTP vocab trimming (closed unmerged) · [#22400](https://github.com/ggml-org/llama.cpp/pull/22400) GDN partial seq_rm (closed unmerged) · [#25589](https://github.com/ggml-org/llama.cpp/pull/25589) Qwen3-Next MTP (b10238) · [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) qwen4exp / Qwen3.8-Flash-Next (b10660) · [#27350](https://github.com/ggml-org/llama.cpp/pull/27350) metal i-quant mul_mv_ext · [#27776](https://github.com/ggml-org/llama.cpp/pull/27776) M3 Pro K-quant mul_mv_ext
- [#23752](https://github.com/ggml-org/llama.cpp/issues/23752) **Metal MTP net loss (contested, closed)** · [#23011](https://github.com/ggml-org/llama.cpp/issues/23011) M1 Pro self-MTP (pre-merge build, closed invalid) · [#27282](https://github.com/ggml-org/llama.cpp/issues/27282) MTP compute arena OOM · [#27781](https://github.com/ggml-org/llama.cpp/issues/27781) shared-KV detection · [#23577](https://github.com/ggml-org/llama.cpp/issues/23577) Qwen3.6-27B repeat bug · [#28158](https://github.com/ggml-org/llama.cpp/issues/28158) OOB token id · [#26765](https://github.com/ggml-org/llama.cpp/issues/26765) qwen35 unused-tensor warnings (open)

**Model / GGUF / benchmarks**
- [Qwen/Qwen3.8-27B config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) · [Qwen/Qwen3.6-27B config.json](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json)
- [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/raw/main/README.md)
- [HauhauCS FastMTP repo](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) + [README](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/raw/main/README.md) + [patch](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch)
- [lsunay/qwen38-mtp](https://github.com/lsunay/qwen38-mtp) (master) + [apple-silicon.md](https://raw.githubusercontent.com/lsunay/qwen38-mtp/master/sweeps/apple-silicon.md)
- [Qwen3.5-9B discussion #56](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/56) · [oMLX Mac MTP recipe](https://raw.githubusercontent.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac/main/README.md)
