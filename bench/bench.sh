#!/usr/bin/env bash
# Benchmark the running endpoint, or re-measure the best MTP depth.
#
#   ./bench/bench.sh --quick     one short-context sanity run (~30s)
#   ./bench/bench.sh             decode + prefill sweep at 512 / 8k / 32k / your context
#   ./bench/bench.sh --tune      re-run the AR-vs-D1/D2/D3 MTP depth measurement
#   ./bench/bench.sh --compare   sweep the config as-is, then with KV_QUANT=off
#
# The sweep uses the context window from env.conf as its upper bound, so a
# 64 GB Mac with CONTEXT_WINDOW=131072 will genuinely push 128k through prefill.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_config

MODE="sweep"
MAX_TOKENS=128
REPEATS=1
CONTEXTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)   MODE="quick" ;;
    --tune)    MODE="tune" ;;
    --compare) MODE="compare" ;;
    --max-tokens) MAX_TOKENS="$2"; shift ;;
    --repeats) REPEATS="$2"; shift ;;
    --contexts) CONTEXTS="$2"; shift ;;
    -h|--help) show_usage "$0"; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

require_bin python3 "python3 is required for the benchmark harness."

# ── re-tune MTP depth ────────────────────────────────────────────────────────
if [[ "$MODE" == "tune" ]]; then
  step "Re-measuring the fastest MTP depth on this Mac"
  [[ -d "$MODEL_DIR" ]] || die "Model not found: $MODEL_DIR"
  if pid_alive; then
    warn "Stopping the running server so the sweep has the machine to itself."
    "$REPO_DIR/stop.sh"
  fi
  log "This loads the model 4 times (AR, D1, D2, D3) and takes 10-25 minutes."
  mtplx tune --model "$MODEL_DIR" --retune --depths 1,2,3 \
      --max-tokens 256 --json > "$RUN_DIR/tune.raw.json" 2> "$RUN_DIR/tune.err" \
    || die "Tune failed. See $RUN_DIR/tune.err"
  TUNE_FILE="$(tune_file_for_current_model)"
  cp "$RUN_DIR/tune.raw.json" "$TUNE_FILE"
  # Stamp which model this was measured on; without it the result is ambiguous
  # as soon as MODEL changes.
  python3 - "$TUNE_FILE" "$MODEL_REPO" <<'PYSTAMP'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
d["model_repo"] = sys.argv[2]
json.dump(d, open(sys.argv[1], "w"), indent=2)
PYSTAMP
  python3 - "$TUNE_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ar = d.get("best_multiplier", {}).get("ar_tok_s")
print()
print(f"  {'mode':>5}  {'tok/s':>9}  {'vs AR':>7}  acceptance")
print(f"  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*30}")
for r in d.get("results", []):
    acc = r.get("acceptance_by_depth") or []
    acc_s = ", ".join(f"{a*100:.1f}%" for a in acc) or "-"
    print(f"  {r['mode']:>5}  {r['tok_s']:>9.2f}  {r['multiplier_vs_ar']:>6.2f}x  {acc_s}")
best = d.get("best", {})
print()
print(f"  Winner: MTP depth {best.get('depth')} at {best.get('tok_s', 0):.2f} tok/s "
      f"({best.get('multiplier_vs_ar', 0):.2f}x over autoregressive)")
print(f"  Saved to {sys.argv[1]} - start.sh picks it up with MTP_DEPTH=auto.")
PY
  exit 0
fi

# ── normal benchmarks need a live server ─────────────────────────────────────
if ! server_healthy; then
  die "No server answering on port $PORT. Start it with ./start.sh"
fi

KEY_ARG=()
[[ -s "$API_KEY_FILE" ]] && KEY_ARG=( --api-key-file "$API_KEY_FILE" )

run_sweep() {
  local label="$1"; shift
  local ctxs="$1"
  step "$label"
  python3 "$REPO_DIR/bench/bench.py" \
    --url "http://127.0.0.1:${PORT}" \
    --model "$SERVED_MODEL_NAME" \
    --contexts "$ctxs" \
    --max-tokens "$MAX_TOKENS" \
    --repeats "$REPEATS" \
    "${KEY_ARG[@]}"
}

if [[ "$MODE" == "quick" ]]; then
  run_sweep "Quick sanity check (short context)" "512"
  exit 0
fi

if [[ -z "$CONTEXTS" ]]; then
  # Skip a sweep rung that is at or above the configured window.
  CONTEXTS="512,8192,32768"
  if (( CONTEXT_WINDOW >= 131072 )); then
    CONTEXTS="$CONTEXTS,131072"
  elif (( CONTEXT_WINDOW > 32768 )); then
    CONTEXTS="$CONTEXTS,$CONTEXT_WINDOW"
  fi
fi

log ""
log "  Config under test:"
log "    model       $MODEL_REPO"
log "    context     $CONTEXT_WINDOW"
log "    KV quant    $KV_QUANT"
log "    MTP depth   $(effective_depth)   profile $PROFILE"
log "    thinking    $THINKING"
log ""

if [[ "$MODE" == "compare" ]]; then
  run_sweep "A. As configured (KV_QUANT=$KV_QUANT)" "$CONTEXTS"
  warn "To compare against unquantized KV, set KV_QUANT=off in env.conf, restart, and re-run."
  log "    (This script does not restart the server for you - that would hide the load cost.)"
  exit 0
fi

run_sweep "Decode / prefill sweep" "$CONTEXTS"

log ""
log "  Interpretation:"
log "    decode  = steady-state generation speed. This is the number that"
log "              determines how fast an agent produces tokens."
log "    prefill = how fast a long prompt is ingested. Matters for the FIRST"
log "              turn of a long conversation; later turns reuse the cached prefix."
log "    ttft    = time to first token, dominated by prefill."
