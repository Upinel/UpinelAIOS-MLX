#!/usr/bin/env bash
# Shared helpers for the UpinelAIOS bundle.
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
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''
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
# Model aliases. MODEL in env.conf may be either one of these short keys or a
# full Hugging Face repo id -- any MTPLX-format repo with a working MTP head.
#
# NOTE: a case statement, not an associative array. macOS ships bash 3.2,
# which has no `declare -A`, and this bundle must run on a stock Mac.
MODEL_ALIASES="4bit 6bit 4bit-opus 6bit-opus official"

model_repo_for() {
  case "$1" in
    4bit)       echo "itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-4bit" ;;
    6bit)       echo "itrejomx/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTPLX-6bit" ;;
    4bit-opus)  echo "barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-4bit" ;;
    6bit-opus)  echo "barozp/Qwen3.8-27B-Opus-Distill-v2-MTPLX-6bit" ;;
    official)   echo "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed" ;;
    */*)        echo "$1" ;;
    *)          die "MODEL=\"$1\" is neither a known alias nor an owner/name repo id.
    Known aliases: $MODEL_ALIASES" ;;
  esac
}

# Everything a human-facing script needs, derived from env.conf.
load_config() {
  [[ -f "$ENV_FILE" ]] || die "env.conf not found at $ENV_FILE"

  # Defaults first, so a trimmed env.conf still works.
  MODEL="4bit"
  MODELS_DIR="$REPO_DIR/models"
  CONTEXT_WINDOW=131072
  MAX_RESPONSE_TOKENS=32768
  KV_QUANT="q8"
  THINKING="low"
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
  SERVED_MODEL_NAME="Upinel-AIOS"
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
  export PRESERVE_THINKING
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
    if server_healthy; then return 0; fi
    if [[ -f "$PID_FILE" ]] && ! pid_alive; then return 2; fi
    sleep 2; waited=$(( waited + 2 ))
  done
  return 1
}

# Read the tuned MTP depth written by `mtplx tune` / install.sh.
tuned_depth() {
  [[ -f "$TUNE_FILE" ]] || { echo ""; return; }
  python3 - "$TUNE_FILE" <<'PY' 2>/dev/null || true
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

# Size of the trunk weights in GB (whole numbers). Prefers the real on-disk
# size so a 6-bit build or a custom repo is accounted for correctly; falls back
# to a guess from the quant tag before the model has been downloaded.
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
  case "${MODEL_REPO:-}" in
    *6bit*|*6-bit*|*Q6*|*6bit*) echo 23 ;;
    *8bit*|*8-bit*|*Q8*)        echo 30 ;;
    *2bit*|*Q2*)                echo 9  ;;
    *)                          echo 15 ;;
  esac
}

# ── argument builders ────────────────────────────────────────────────────────
# Translate THINKING=<off|low|medium|high> into MTPLX reasoning flags.
thinking_flags() {
  case "$THINKING" in
    off)    echo "--reasoning off" ;;
    low)    echo "--reasoning on --reasoning-effort low" ;;
    medium|mid) echo "--reasoning on --reasoning-effort medium" ;;
    high)   echo "--reasoning on --reasoning-effort high" ;;
    *)      die "THINKING=\"$THINKING\" is not one of off | low | medium | high" ;;
  esac
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
