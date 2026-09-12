#!/usr/bin/env bash
# Start the UpinelAIOS server.
#
#   ./start.sh                 start in the background (default)
#   ./start.sh --foreground    run attached to this terminal (Ctrl-C to stop)
#   ./start.sh --print         print the command it would run, then exit
#   ./start.sh --model 9b      serve a different model for this run only
#
# All settings come from env.conf.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

FOREGROUND=0; PRINT_ONLY=0; MODEL_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--foreground) FOREGROUND=1 ;;
    --print)         PRINT_ONLY=1 ;;
    --model)         MODEL_OVERRIDE="$2"; shift ;;
    -h|--help)       show_usage "$0"; exit 0 ;;
    *) die "Unknown argument: $1  (try --help)" ;;
  esac
  shift
done

# A --model flag overrides env.conf for this launch only. It is not written
# back: use ./model_download.sh --switch to make a choice permanent.
if [[ -n "$MODEL_OVERRIDE" ]]; then
  MODEL_REPO="$(model_repo_for "$MODEL_OVERRIDE")"
  MODEL_DIR="$MODELS_DIR/${MODEL_REPO//\//--}"
  export MODEL_REPO MODEL_DIR
  info "Model override for this run: $MODEL_REPO"
fi

step "UpinelAIOS server"

# ── preflight ────────────────────────────────────────────────────────────────
is_apple_silicon || die "This bundle needs an Apple Silicon Mac."
require_bin mtplx "Run ./install.sh first, or: brew install youssofal/mtplx/mtplx"

[[ -d "$MODEL_DIR" ]] || die "Model not found at $MODEL_DIR
    Run ./install.sh, or point MODEL= at a model you already have."

EFFECTIVE_DEPTH="$(effective_depth)"

# Validate enum settings up front, so a typo fails before we check the port or
# load 15 GB of weights.
THINKING_ARGS="$(thinking_flags)"
case "$KV_QUANT" in off|q8|q4) ;; *) die "KV_QUANT=\"$KV_QUANT\" is not one of off | q8 | q4" ;; esac
case "$PROFILE" in turbo|sustained|stable|exact|performance-cold|max-diagnostic) ;;
  *) die "PROFILE=\"$PROFILE\" is not a valid MTPLX profile." ;;
esac
case "$BATCHING_PRESET" in solo|latency|agent|throughput) ;;
  *) die "BATCHING_PRESET=\"$BATCHING_PRESET\" is not one of solo | latency | agent | throughput" ;;
esac
case "$FAN_MODE" in default|smart|max) ;; *) die "FAN_MODE=\"$FAN_MODE\" is not one of default | smart | max" ;; esac
case "$SSD_SESSION_CACHE" in on|off|write-only) ;; *) die "SSD_SESSION_CACHE=\"$SSD_SESSION_CACHE\" is not one of on | off | write-only" ;; esac
case "$PRESERVE_THINKING" in auto|on|off|scoped) ;; *) die "PRESERVE_THINKING=\"$PRESERVE_THINKING\" is not one of auto | on | off | scoped" ;; esac
thinking_level_ok "$THINKING" || die "THINKING=\"$THINKING\" is not one of off | minimal | low | medium | high"

# A depth of 0 or an explicit --no-mtp means plain autoregressive decoding.
# On Apple Silicon that is roughly a 3x slowdown, so warn loudly.
if [[ "$EFFECTIVE_DEPTH" == "0" ]]; then
  warn "MTP_DEPTH=0 disables speculative decoding. Expect ~3x slower output."
fi

# Reject a configuration that cannot fit before we spend a minute loading it.
RAM_GB="$(total_ram_gb)"
KV_KB_PER_TOK="$(kv_kb_per_token)"
KV_GB=$(( CONTEXT_WINDOW * KV_KB_PER_TOK / 1024 / 1024 ))
MODEL_GB="$(model_weight_gb)"
NEED_GB=$(( MODEL_GB + KV_GB + 6 ))
if (( NEED_GB > MEMORY_LIMIT_GB )); then
  warn "Plan needs ~${NEED_GB} GB but MEMORY_LIMIT_GB=${MEMORY_LIMIT_GB}."
  warn "Lower CONTEXT_WINDOW, use KV_QUANT=q4, or raise MEMORY_LIMIT_GB."
fi
if (( NEED_GB > RAM_GB )); then
  die "Plan needs ~${NEED_GB} GB but this Mac has ${RAM_GB} GB. It will not load."
fi

# Loopback binds need no key; anything else does, and MTPLX enforces it.
NEED_KEY=0
[[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]] && NEED_KEY=1
(( NEED_KEY )) && ensure_api_key

# ── already running? ─────────────────────────────────────────────────────────
if pid_alive; then
  if server_healthy; then
    ok "Already running (pid $(cat "$PID_FILE")). Use ./stop.sh first to reconfigure."
    exit 0
  fi
  warn "Stale pid file - cleaning up."
  rm -f "$PID_FILE"
fi

EXISTING="$(port_pids | head -1)"
if [[ -n "$EXISTING" ]]; then
  die "Port $PORT is already in use by pid $EXISTING.
    Stop it, or change PORT in env.conf."
fi

# ── build the command ────────────────────────────────────────────────────────
ARGS=(
  --model "$MODEL_DIR"
  --profile "$PROFILE"
  --host "$HOST"
  --port "$PORT"
  --context-window "$CONTEXT_WINDOW"
  --max-tokens "$MAX_RESPONSE_TOKENS"
  --paged-kv-quantization "$KV_QUANT"
  --batching-preset "$BATCHING_PRESET"
  --max-active-requests "$MAX_CONCURRENT"
  --stream-interval "$STREAM_INTERVAL"
  --ssd-session-cache "$SSD_SESSION_CACHE"
  --warmup-tokens "$WARMUP_TOKENS"
  --model-id "$SERVED_MODEL_NAME"
  --no-stats-footer
)

if [[ "$EFFECTIVE_DEPTH" == "0" ]]; then
  ARGS+=( --no-mtp )
else
  ARGS+=( --depth "$EFFECTIVE_DEPTH" )
fi

(( NEED_KEY )) && ARGS+=( --api-key-file "$API_KEY_FILE" )
(( RATE_LIMIT > 0 )) && ARGS+=( --rate-limit "$RATE_LIMIT" )
ARGS+=( --preserve-thinking "$PRESERVE_THINKING" )
[[ "$FAN_MODE" != "default" ]] && ARGS+=( --fan-mode "$FAN_MODE" )
(( PREFILL_CHUNK_TOKENS > 0 )) && ARGS+=( --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" )

# shellcheck disable=SC2206
ARGS+=( $THINKING_ARGS )

# Bound how long the model may think. MTPLX's thinking guard only applies to
# requests that carry tools, which is exactly the agent case, and it is off by
# default - so this is what actually turns "think briefly" into a limit rather
# than a suggestion.
apply_thinking_budget_env

# Memory ceilings. MTPLX otherwise defaults to 75% of physical RAM for the
# allocator and 60% for wired pages, which is already sane; these let env.conf
# pin it instead of guessing, and matter on Macs with more than 128 GB.
MEM_BYTES=$(( MEMORY_LIMIT_GB * 1024 * 1024 * 1024 ))
export MTPLX_MEMORY_LIMIT_BYTES="$MEM_BYTES"
if (( WIRED_LIMIT_GB > 0 )); then
  export MTPLX_WIRED_LIMIT_BYTES=$(( WIRED_LIMIT_GB * 1024 * 1024 * 1024 ))
fi
# Let MTPLX do its own accounting against that same number.
export MTPLX_MEMORY_BUDGET="$MEM_BYTES"

# The session/prefix bank is the biggest allocator after the weights. MTPLX
# otherwise auto-sizes it to half the post-model surplus, which on a 64 GB Mac
# is ~16 GB - enough to push macOS into swap once a long KV cache is live.
if (( SESSION_BANK_GB > 0 )); then
  export MTPLX_SESSION_BANK_MAX_BYTES=$(( SESSION_BANK_GB * 1024 * 1024 * 1024 ))
  # Never let one session claim the whole bank.
  PER=$(( SESSION_BANK_GB / 2 )); (( PER < 2 )) && PER=2
  export MTPLX_SESSION_BANK_PER_SESSION_BYTES=$(( PER * 1024 * 1024 * 1024 ))
fi
if (( MLX_CACHE_LIMIT_GB > 0 )); then
  export MTPLX_MLX_CACHE_LIMIT=$(( MLX_CACHE_LIMIT_GB * 1024 * 1024 * 1024 ))
fi

mkdir -p "$RUN_DIR"

if (( PRINT_ONLY )); then
  log "MTPLX_MEMORY_LIMIT_BYTES=$MTPLX_MEMORY_LIMIT_BYTES \\"
  log "mtplx serve ${ARGS[*]}"
  exit 0
fi

# ── banner ───────────────────────────────────────────────────────────────────
LAN="$(lan_ip)"
log ""
log "  model        $MODEL_REPO"
log "  weights      $MODEL_DIR"
log "  context      $CONTEXT_WINDOW tokens      (KV: $KV_QUANT)"
log "  MTP depth    $EFFECTIVE_DEPTH              (profile: $PROFILE)"
if [[ "$THINKING" == "off" ]]; then
  log "  thinking     off   (history: $PRESERVE_THINKING)"
else
  log "  thinking     $THINKING   budget ${MTPLX_THINKING_BUDGET:-unlimited} tok   (history: $PRESERVE_THINKING)"
fi
log "  memory cap   ${MEMORY_LIMIT_GB} GB"
if (( SESSION_BANK_GB > 0 )); then
  log "  session bank ${SESSION_BANK_GB} GB   (prefix cache for repeat turns)"
else
  log "  session bank auto"
fi
log "  batching     $BATCHING_PRESET, max $MAX_CONCURRENT concurrent"
log ""
log "  local URL    http://127.0.0.1:${PORT}/v1"
if (( NEED_KEY )); then
  log "  LAN URL      http://${LAN}:${PORT}/v1"
  log "  API key      $(cat "$API_KEY_FILE")"
else
  log "  bound to     $HOST (not reachable from other devices)"
fi
log ""

# ── launch ───────────────────────────────────────────────────────────────────
if (( FOREGROUND )); then
  info "Starting in the foreground. Ctrl-C to stop."
  exec mtplx serve "${ARGS[@]}"
fi

# Record the effective config so ./restart.sh can report what changed.
save_config_snapshot

info "Loading the model - first token takes ~30-90s. Logs: $LOG_FILE"
# Rotate the previous log so a long-running service cannot fill the disk.
if [[ -f "$LOG_FILE" ]]; then
  SZ="$(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)"
  (( SZ > 33554432 )) && mv -f "$LOG_FILE" "${LOG_FILE}.1"
fi

# Detach fully so the server outlives this shell, this terminal, and any
# CI/agent harness waiting on our stdout. stdin from /dev/null stops the
# child from holding the terminal open; closing stdout/stderr is what lets a
# wrapper script return immediately.
nohup mtplx serve "${ARGS[@]}" </dev/null >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true
echo "$SERVER_PID" > "$PID_FILE"



info "Waiting for the server to answer /health ..."
wait_healthy 900 && RC=0 || RC=$?
case "$RC" in
  0)
    ok "Server is up."
    log ""
    log "  Chat completions:  POST http://${LAN}:${PORT}/v1/chat/completions"
    log "  Health:            GET  http://${LAN}:${PORT}/health"
    log "  Models:            GET  http://${LAN}:${PORT}/v1/models"
    log ""
    log "  Try it:  ./bench/bench.sh --quick"
    log "  Stop:    ./stop.sh"
    ;;
  2)
    rm -f "$PID_FILE"
    die "The server exited during startup. Last lines of $LOG_FILE:
$(tail -n 25 "$LOG_FILE" 2>/dev/null)"
    ;;
  1)
    die "Timed out after 900s waiting for /health. Check $LOG_FILE"
    ;;
esac
