#!/usr/bin/env bash
# Shared helpers for the UpinelAIOS-MLX bundle.
# Sourced by install.sh / start.sh / stop.sh / status.sh / bench/bench.sh

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
COMMON_SH_SOURCE="${BASH_SOURCE[0]}"
LIB_DIR="$(cd "$(dirname "$COMMON_SH_SOURCE")" && pwd)"
REPO_DIR="$(cd "$LIB_DIR/.." && pwd)"
export REPO_DIR

RUN_DIR="$REPO_DIR/run"
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/server.pid"
TUNE_FILE="$RUN_DIR/tuning.json"
ENV_FILE="$REPO_DIR/env.conf"

# ── pretty output ────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'; C_MAGENTA=$'\033[35m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''
  C_CYAN=''; C_MAGENTA=''
fi

log()   { printf '%s\n' "$*"; }
info()  { printf '%s\n' "${C_BLUE}==>${C_RESET} $*"; }
ok()    { printf '%s\n' "${C_GREEN} ok${C_RESET} $*"; }
warn()  { printf '%s\n' "${C_YELLOW}warn${C_RESET} $*" >&2; }
die()   { printf '%s\n' "${C_RED}fail${C_RESET} $*" >&2; exit 1; }
step()  { printf '\n%s\n' "${C_BOLD}$*${C_RESET}"; }

# ── usage ────────────────────────────────────────────────────────────────────
# Print the leading comment block of a script as its --help text.
show_usage() {
  local script="$1"
  awk 'NR == 1 { next }
       /^#/ { sub(/^# ?/, ""); print; next }
       /^[[:space:]]*$/ { if (seen) exit; else next }
       { exit }' "$script"
}

# ── config ───────────────────────────────────────────────────────────────────
# Model registry.
#
# EVERY entry here is an uncensored fine-tune. That is a deliberate product
# rule, not a coincidence: UpinelAIOS-MLX does not ship or suggest aligned models.
# Each one is an MLX conversion packaged for MTPLX with a verified MTP head,
# which is what makes the speculative decoding worth having.
#
#   alias      size    repo
#   ---------  ------  -------------------------------------------------------
#   4bit       15 GB   itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit
#   6bit       22 GB   itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-6bit
#   27b-3bit   13 GB   barozp/Qwen3.8-27B-Uncensored-MTPLX-3bit
#   27b-4bit   16 GB   barozp/Qwen3.8-27B-Uncensored-MTPLX-4bit
#   9b          5 GB   Foresee/Qwen3.8-9B-heretic-uncensored-4bit-MTPLX
#   moe        22 GB   hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16
#
#   "moe" is the speed pick: a mixture-of-experts model with 3B active per
#   token, so it is several times faster than the dense 27B at the same memory
#   footprint. Its publisher measured depth 1 at 89 tok/s.
#
# Nothing smaller than 9B is listed because a 27B is what this bundle is built
# and measured around; smaller uncensored MTPLX packs exist if you want to try
# one, and any owner/name repo id is accepted.
#
# NOTE: a case statement, not an associative array. macOS ships bash 3.2,
# which has no `declare -A`, and this bundle must run on a stock Mac.
MODEL_ALIASES="4bit 6bit 27b-3bit 27b-4bit 9b moe"

model_repo_for() {
  case "$1" in
    4bit)      echo "itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit" ;;
    6bit)      echo "itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-6bit" ;;
    27b-3bit)  echo "barozp/Qwen3.8-27B-Uncensored-MTPLX-3bit" ;;
    27b-4bit)  echo "barozp/Qwen3.8-27B-Uncensored-MTPLX-4bit" ;;
    9b)        echo "Foresee/Qwen3.8-9B-heretic-uncensored-4bit-MTPLX" ;;
    moe)       echo "hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16" ;;
    */*)       echo "$1" ;;
    *)         die "MODEL=\"$1\" is neither a known alias nor an owner/name repo id.
    Known aliases: $MODEL_ALIASES" ;;
  esac
}

# Is this alias/repo one we have measured and endorse?
model_is_known_alias() {
  case "$1" in
    4bit|6bit|27b-3bit|27b-4bit|9b|moe) return 0 ;;
    *) return 1 ;;
  esac
}

# Everything a human-facing script needs, derived from env.conf.
load_config() {
  [[ -f "$ENV_FILE" ]] || die "env.conf not found at $ENV_FILE"

  # Defaults first, so a trimmed env.conf still works.
  # moe, not 4bit: the MoE is the measured fastest thing here (~83 tok/s vs
  # 42-51 for the dense 27B) and is what the README documents as the default.
  MODEL="moe"
  MODELS_DIR="$REPO_DIR/models"
  CONTEXT_WINDOW=131072
  MAX_RESPONSE_TOKENS=32768
  KV_QUANT="q8"
  THINKING="minimal"
  THINKING_BUDGET_TOKENS=0
  THINKING_NOVELTY_CLOSE=1
  PRESERVE_THINKING="scoped"
  MTP_DEPTH="auto"
  PROFILE="turbo"
  BATCHING_PRESET="agent"
  MAX_CONCURRENT=2
  STREAM_INTERVAL=1
  PREFILL_CHUNK_TOKENS=0
  SSD_SESSION_CACHE="on"
  MEMORY_LIMIT_GB=48
  WIRED_LIMIT_GB=0
  SESSION_BANK_GB=8
  MLX_CACHE_LIMIT_GB=0
  HOST="0.0.0.0"
  PORT=8000
  API_KEY_FILE="$REPO_DIR/run/api-key"
  SERVED_MODEL_NAME="Upinel-AIOS-MLX"
  RATE_LIMIT=0
  FAN_MODE="default"
  NGRAM_PREWARM="auto"
  WARMUP_TOKENS=8
  LOG_FILE="$REPO_DIR/run/server.log"

  # shellcheck disable=SC1090
  source "$ENV_FILE"

  MODEL_REPO="$(model_repo_for "$MODEL")"
  MODEL_DIR="$MODELS_DIR/${MODEL_REPO//\//--}"
  export MODEL MODEL_REPO MODEL_DIR MODELS_DIR CONTEXT_WINDOW MAX_RESPONSE_TOKENS
  export KV_QUANT THINKING MTP_DEPTH PROFILE BATCHING_PRESET MAX_CONCURRENT
  export PRESERVE_THINKING THINKING_BUDGET_TOKENS THINKING_NOVELTY_CLOSE
  export MEMORY_LIMIT_GB WIRED_LIMIT_GB SESSION_BANK_GB MLX_CACHE_LIMIT_GB
  export HOST PORT API_KEY_FILE SERVED_MODEL_NAME FAN_MODE LOG_FILE
}

# ── machine facts ────────────────────────────────────────────────────────────
is_apple_silicon() { [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; }

total_ram_gb() {
  awk -v b="$(sysctl -n hw.memsize)" 'BEGIN { printf "%d", b/1024/1024/1024 }'
}

gpu_cores() {
  system_profiler SPDisplaysDataType 2>/dev/null \
    | awk -F': ' '/Total Number of Cores/ { print $2; exit }' | tr -d ' '
}

macos_version() { sw_vers -productVersion; }

# ── requirement checks ───────────────────────────────────────────────────────
require_macos() {
  local v; v="$(macos_version)"
  local major="${v%%.*}"
  if (( major < 14 )); then
    die "macOS $v detected. Apple Silicon MLX inference needs macOS 14 or newer."
  fi
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH. $2"
}

# ── API key ──────────────────────────────────────────────────────────────────
ensure_api_key() {
  mkdir -p "$(dirname "$API_KEY_FILE")"
  if [[ ! -s "$API_KEY_FILE" ]]; then
    # 32 hex chars from the OS CSPRNG
    ( umask 077; openssl rand -hex 24 > "$API_KEY_FILE" )
    chmod 600 "$API_KEY_FILE"
    info "Generated a new API key at $API_KEY_FILE"
  fi
  API_KEY="$(tr -d '\n' < "$API_KEY_FILE")"
  [[ -n "$API_KEY" ]] || die "API key file $API_KEY_FILE is empty."
  export API_KEY
}

# ── LAN address ──────────────────────────────────────────────────────────────
lan_ip() {
  local ip=""
  ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
  [[ -z "$ip" ]] && ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
  [[ -z "$ip" ]] && ip="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -1)"
  # Fall back to a route lookup so a single-homed Mac always answers.
  [[ -z "$ip" ]] && ip="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' | xargs -I{} ipconfig getifaddr {} 2>/dev/null || true)"
  echo "${ip:-<your-lan-ip>}"
}

# ── process / port helpers ───────────────────────────────────────────────────
pid_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local p; p="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$p" ]] || return 1
  kill -0 "$p" 2>/dev/null
}

# Anything listening on $PORT (an MTPLX daemon started outside this bundle).
port_pids() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true; }

# /health requires the API key whenever the server is bound to a
# non-loopback address, so always present it when we have one.
auth_header() {
  if [[ -s "${API_KEY_FILE:-}" ]]; then
    printf 'Authorization: Bearer %s' "$(tr -d '\n' < "$API_KEY_FILE")"
  fi
}

# Is the server on $PORT actually OURS?
#
# /health alone is not enough. Any OpenAI-compatible server answers it, so a
# server belonging to a *different* project satisfies the check and start.sh
# reports success while serving the wrong thing. That happened for real: this
# project's start.sh printed "Server is up" while a GGUF llama-server from the
# sister project held port 8000, and the only visible symptom was every client
# getting 401 against a key the running server had never heard of.
#
# The served model id is the cheapest reliable fingerprint - each project
# publishes its own, and it is present as soon as the server can answer.
server_is_ours() {
  local hdr body
  hdr="$(auth_header)"
  if [[ -n "$hdr" ]]; then
    body="$(curl -fsS --max-time 4 -H "$hdr" \
              "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)"
  else
    body="$(curl -fsS --max-time 4 \
              "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)"
  fi
  [[ -n "$body" ]] && grep -qF "$SERVED_MODEL_NAME" <<<"$body"
}

server_healthy() {
  local hdr; hdr="$(auth_header)"
  if [[ -n "$hdr" ]]; then
    curl -fsS --max-time 4 -H "$hdr" "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
  else
    curl -fsS --max-time 4 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
  fi
}

# Raw /health body (or {} on failure).
health_json() {
  local hdr; hdr="$(auth_header)"
  if [[ -n "$hdr" ]]; then
    curl -fsS --max-time 6 -H "$hdr" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo '{}'
  else
    curl -fsS --max-time 6 "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo '{}'
  fi
}

wait_healthy() {
  local timeout="${1:-600}" waited=0
  while (( waited < timeout )); do
    # Both: answering /health is not the same as being our server.
    if server_healthy && server_is_ours; then return 0; fi
    if [[ -f "$PID_FILE" ]] && ! pid_alive; then return 2; fi
    sleep 2; waited=$(( waited + 2 ))
  done
  return 1
}

# Where a tune result for the currently selected model belongs.
tune_file_for_current_model() {
  printf '%s/tuning-%s.json' "$RUN_DIR" "${MODEL_REPO//\//--}"
}

# Read the tuned MTP depth written by `mtplx tune` / install.sh.
#
# Results are per model: depth 2 is right for the dense 27B and depth 1 for the
# MoE, so one shared file would apply the wrong depth the moment MODEL changed.
# A tuning result is only honoured for the model it was measured on. The
# pre-split shared file has no model recorded and is accepted only for the
# model this bundle ships as default; anything else falls through to the
# runtime default rather than inheriting a depth tuned for a different model.
tuned_depth() {
  local path="" recorded=""
  if [[ -n "${MODEL_REPO:-}" ]]; then
    path="$RUN_DIR/tuning-${MODEL_REPO//\//--}.json"
    [[ -f "$path" ]] || path=""
  fi
  if [[ -z "$path" ]] && [[ -f "$TUNE_FILE" ]]; then
    recorded="$(python3 -c '
import json,sys
try: print(json.load(open(sys.argv[1])).get("model_repo") or "")
except Exception: print("")
' "$TUNE_FILE" 2>/dev/null || true)"
    # Empty means a legacy file: usable only for the default model.
    if [[ -n "$recorded" ]]; then
      [[ "$recorded" == "$MODEL_REPO" ]] && path="$TUNE_FILE"
    elif [[ "$MODEL_REPO" == "$(model_repo_for 4bit)" ]]; then
      path="$TUNE_FILE"
    fi
  fi
  [[ -n "$path" ]] || { echo ""; return; }
  python3 - "$path" <<'PY' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
best = d.get("best_depth") or d.get("winner") or d.get("depth")
if best is None:
    for k in ("depths", "results", "rows"):
        rows = d.get(k)
        if isinstance(rows, list) and rows:
            cand = [r for r in rows if isinstance(r, dict) and (r.get("tok_s") or r.get("tok_per_s"))]
            if cand:
                best = max(cand, key=lambda r: r.get("tok_s") or r.get("tok_per_s") or 0).get("depth")
                break
if isinstance(best, dict):
    best = best.get("depth")
print("" if best is None else int(best))
PY
}

# ── memory arithmetic ────────────────────────────────────────────────────────
# KV bytes/token for this hybrid model: only 16 of 64 layers keep a KV cache,
# so one token costs 16 x 2 x 4 kv-heads x 256 head-dim = 32768 elements.
kv_kb_per_token() {
  case "$KV_QUANT" in
    q4|q4_0)  echo 18  ;;   # ~0.56 bytes/element
    q8|q8_0)  echo 34  ;;   # ~1.06 bytes/element
    *)        echo 64  ;;   # f16, 2 bytes/element
  esac
}

# KV cost per token in KB at f16, for one specific model. Divide by 2 for q8,
# by 4 for q4.
#
# The table above is the 27B's own curve, and it is right for the 27B: that
# model is a hybrid that caches KV on only 16 of its 64 layers. But the other
# entries in the registry are not the same shape, and reusing the 27B's numbers
# for them gets the comparison exactly backwards:
#
#   * the dense 9B keeps KV on ALL 32 of its layers, so it is twice as
#     expensive per token as the 27B despite being a quarter of the size;
#   * the 35B MoE is hybrid like the 27B and has only 12 attention layers.
#
# Sizes below are elements per token (K and V, 4 kv-heads x 256 head-dim).
# The 27B row is the measured/model-card figure; the other two are derived
# from their layer counts and are marked as estimates.
#
# Keyed on the FAMILY, not on the quant tag: the MoE's repo id contains
# "MTPLX-4bit" as well, so matching on that would silently hand the 35B the
# 27B's curve. Same trap as the draft-head mixup in the GGUF sister project.
kv_kb_per_token_f16() {
  case "$1" in
    # ~12 of 48 attention layers.
    *Qwen3.6-35B*) echo 48  ;;   # estimate - MoE, hybrid, only some layers cache KV
    # 32 of 32 layers - dense, full attention on every layer.
    *Qwen3.8-9B*)  echo 128 ;;   # estimate - dense: 2x the 27B per token, at a quarter the size
    # 16 of 64 layers - every 27B build here (4bit/6bit/3bit, both owners).
    *)             echo 64  ;;
  esac
}

# How many GB a repo's model files weigh, by repo id. This is the sum of every
# .safetensors file the runtime loads - trunk weights, the MTP head and the
# vision tower - which is also what model_weight_gb() measures on disk, so the
# two agree instead of disagreeing by a few GB.
#
# Read off the published repos (exact byte totals divided by 1e9, truncated):
#   4bit 15.98, 6bit 22.71, 27b-3bit 13.54, 27b-4bit 16.90, 9b 5.52, moe 22.09
#
# Keyed on OWNER as well as quant: the two 4-bit 27B repos are 15 GB and 16 GB
# for the same nominal quant, and a quant-only pattern matches both at once.
model_download_gb() {
  case "$1" in
    *itrejomx*MTPLX-6bit*) echo 22 ;;
    *itrejomx*MTPLX-4bit*) echo 15 ;;
    *barozp*MTPLX-4bit*)   echo 16 ;;
    *MTPLX-3bit*)          echo 13 ;;
    *Qwen3.8-9B*)          echo 5  ;;
    *Qwen3.6-35B*)         echo 22 ;;
    *)                     echo 0  ;;
  esac
}

# Size of the trunk weights in GB (whole numbers). Prefers the real on-disk
# size so a 6-bit build or a custom repo is accounted for correctly; falls back
# to the published size before the model has been downloaded.
model_weight_gb() {
  if [[ -n "${MODEL_DIR:-}" && -d "${MODEL_DIR:-}" ]]; then
    local bytes
    bytes="$(find "$MODEL_DIR" -name '*.safetensors' -type f \
              -exec stat -f%z {} + 2>/dev/null | awk '{n+=$1} END {print n+0}')"
    if (( bytes > 1000000000 )); then
      echo $(( bytes / 1000000000 ))
      return
    fi
  fi
  # Fall back to the published size. This used to guess from the quant tag
  # alone, which returned 15 GB for moe, 9b and 27b-4bit alike - so a fresh
  # install was told a 22 GB model would fit a 16 GB Mac.
  local gb
  gb="$(model_download_gb "${MODEL_REPO:-}")"
  if (( gb > 0 )); then echo "$gb"; return; fi
  case "${MODEL_REPO:-}" in
    *6bit*|*6-bit*|*Q6*|*6bit*) echo 23 ;;
    *8bit*|*8-bit*|*Q8*)        echo 30 ;;
    *2bit*|*Q2*)                echo 9  ;;
    *)                          echo 15 ;;
  esac
}

# ── config identity ──────────────────────────────────────────────────────────
# The effective settings, independent of comments and whitespace, so that
# editing a comment does not look like a configuration change.
config_fingerprint() {
  local depth; depth="$(effective_depth)"
  printf '%s\n' \
    "MODEL=$MODEL_REPO" \
    "CONTEXT_WINDOW=$CONTEXT_WINDOW" \
    "MAX_RESPONSE_TOKENS=$MAX_RESPONSE_TOKENS" \
    "KV_QUANT=$KV_QUANT" \
    "THINKING=$THINKING" \
    "PRESERVE_THINKING=$PRESERVE_THINKING" \
    "MTP_DEPTH=$depth" \
    "PROFILE=$PROFILE" \
    "BATCHING_PRESET=$BATCHING_PRESET" \
    "MAX_CONCURRENT=$MAX_CONCURRENT" \
    "STREAM_INTERVAL=$STREAM_INTERVAL" \
    "PREFILL_CHUNK_TOKENS=$PREFILL_CHUNK_TOKENS" \
    "SSD_SESSION_CACHE=$SSD_SESSION_CACHE" \
    "MEMORY_LIMIT_GB=$MEMORY_LIMIT_GB" \
    "WIRED_LIMIT_GB=$WIRED_LIMIT_GB" \
    "SESSION_BANK_GB=$SESSION_BANK_GB" \
    "MLX_CACHE_LIMIT_GB=$MLX_CACHE_LIMIT_GB" \
    "HOST=$HOST" \
    "PORT=$PORT" \
    "SERVED_MODEL_NAME=$SERVED_MODEL_NAME" \
    "RATE_LIMIT=$RATE_LIMIT" \
    "FAN_MODE=$FAN_MODE" \
    "NGRAM_PREWARM=$NGRAM_PREWARM" \
    "WARMUP_TOKENS=$WARMUP_TOKENS"
}

CONFIG_SNAPSHOT_FILE="$RUN_DIR/config.snapshot"

save_config_snapshot() {
  mkdir -p "$RUN_DIR"
  config_fingerprint > "$CONFIG_SNAPSHOT_FILE"
}

# Print the settings that differ from the last saved snapshot. Returns 0 if
# there were changes, 1 if the config is identical or there is no snapshot.
diff_config_snapshot() {
  [[ -f "$CONFIG_SNAPSHOT_FILE" ]] || return 1
  local changed
  changed="$(diff <(config_fingerprint) "$CONFIG_SNAPSHOT_FILE" 2>/dev/null \
             | grep -E '^[<>]' || true)"
  [[ -n "$changed" ]] || return 1
  printf '%s\n' "$changed"
  return 0
}

# ── argument builders ────────────────────────────────────────────────────────
# Thinking levels, cheapest first. MTPLX's own chat template only distinguishes
# low/medium/xhigh, and "low" is a soft nudge the model ignores. What actually
# bounds it is MTPLX's thinking guard: a token budget, plus a novelty close that
# ends thinking early once it stops saying anything new.
#
#   off      no thinking at all
#   minimal  answer first, think only if forced. The default.
#   low      brief thinking, hard-capped
#   medium   more room, still capped
#   high     no budget (the model's xhigh)
thinking_level_ok() {
  case "$1" in off|minimal|low|medium|high) return 0 ;; *) return 1 ;; esac
}

# Default thinking-token budget per level. 0 means unlimited.
thinking_budget_for() {
  case "$1" in
    off)     echo 0 ;;
    minimal) echo 128 ;;
    low)     echo 512 ;;
    medium)  echo 2048 ;;
    high)    echo 0 ;;
    *)       echo 512 ;;
  esac
}

# Template effort level to ask the model for, per our level.
thinking_effort_for() {
  case "$1" in
    high) echo xhigh ;;
    medium) echo medium ;;
    *) echo low ;;
  esac
}

# Translate THINKING into MTPLX reasoning flags.
thinking_flags() {
  thinking_level_ok "$THINKING" || die "THINKING=\"$THINKING\" is not one of off | minimal | low | medium | high"
  if [[ "$THINKING" == "off" ]]; then
    echo "--reasoning off"
  else
    echo "--reasoning on --reasoning-effort $(thinking_effort_for "$THINKING")"
  fi
}

# Export the thinking guard limits for the level that will be launched.
apply_thinking_budget_env() {
  local budget="$THINKING_BUDGET_TOKENS"
  [[ "$budget" == "0" ]] && budget="$(thinking_budget_for "$THINKING")"
  if [[ "$THINKING" == "off" ]]; then
    unset MTPLX_THINKING_BUDGET MTPLX_THINKING_NOVELTY_CLOSE
    return 0
  fi
  if (( budget > 0 )); then
    export MTPLX_THINKING_BUDGET="$budget"
    # End thinking early once it stops producing new content, so a capped run
    # does not merely truncate mid-thought.
    export MTPLX_THINKING_NOVELTY_CLOSE="${THINKING_NOVELTY_CLOSE:-1}"
  else
    unset MTPLX_THINKING_BUDGET
    export MTPLX_THINKING_NOVELTY_CLOSE="${THINKING_NOVELTY_CLOSE:-1}"
  fi
}

# ── live settings (no restart needed) ───────────────────────────────────────
# MTPLX exposes POST /v1/mtplx/settings accepting reasoning, reasoning_effort,
# enable_thinking, depth and more. This is what makes an on-the-fly toggle
# possible: the model stays loaded and only the decode policy changes.
live_settings_get() {
  local hdr; hdr="$(auth_header)"
  curl -fsS --max-time 8 ${hdr:+-H "$hdr"} "http://127.0.0.1:${PORT}/v1/mtplx/settings" 2>/dev/null
}

live_settings_set() {
  local json="$1"
  local hdr; hdr="$(auth_header)"
  curl -fsS --max-time 10 -X POST     -H "Content-Type: application/json" ${hdr:+-H "$hdr"}     -d "$json" "http://127.0.0.1:${PORT}/v1/mtplx/settings" 2>/dev/null
}

# Map a THINKING level onto the live-settings payload.
live_thinking_payload() {
  local level="$1"
  thinking_level_ok "$level" || die "Unknown thinking level: $level"
  local effort; effort="$(thinking_effort_for "$level")"
  if [[ "$level" == "off" ]]; then
    printf '{"reasoning":"off","enable_thinking":false}'
  else
    printf '{"reasoning":"on","enable_thinking":true,"reasoning_effort":"%s"}' "$effort"
  fi
}

# Resolve MTP_DEPTH=auto against the tuned value.
effective_depth() {
  if [[ "$MTP_DEPTH" == "auto" ]]; then
    local d; d="$(tuned_depth)"
    echo "${d:-3}"     # 3 is the MTPLX default when nothing is measured
  else
    echo "$MTP_DEPTH"
  fi
}
