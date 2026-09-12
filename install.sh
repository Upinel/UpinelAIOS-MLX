#!/usr/bin/env bash
# One-time setup for the M5 Pro Qwen Agent bundle.
#
#   ./install.sh              full setup: deps + model download + speed tune
#   ./install.sh --no-tune    skip the (slow) MTP depth measurement
#   ./install.sh --deps-only  only install the MTPLX runtime
#   ./install.sh --model-only only download/verify the model

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

DO_DEPS=1; DO_MODEL=1; DO_TUNE=1
for arg in "$@"; do
  case "$arg" in
    --deps-only)  DO_MODEL=0; DO_TUNE=0 ;;
    --model-only) DO_DEPS=0;  DO_TUNE=0 ;;
    --no-tune)    DO_TUNE=0 ;;
    -h|--help)    show_usage "$0"; exit 0 ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

step "M5 Pro Qwen Agent - installer"

# ── 0. sanity ────────────────────────────────────────────────────────────────
is_apple_silicon || die "This bundle targets Apple Silicon Macs. Detected: $(uname -s)/$(uname -m)."
require_macos
RAM_GB="$(total_ram_gb)"
GPU_CORES="$(gpu_cores)"
info "Machine: Apple Silicon, ${RAM_GB} GB unified memory, ${GPU_CORES} GPU cores, macOS $(macos_version)"
info "Model:   $MODEL  ->  $MODEL_REPO"

if (( RAM_GB < 32 )); then
  warn "Only ${RAM_GB} GB of unified memory. A 4-bit 27B needs ~16 GB for weights"
  warn "plus KV cache. Consider lowering CONTEXT_WINDOW and MEMORY_LIMIT_GB in env.conf."
fi

# Headroom check: model + KV for the configured context must fit the budget.
MODEL_GB="$(model_weight_gb)"
KV_KB_PER_TOK="$(kv_kb_per_token)"
KV_GB=$(( CONTEXT_WINDOW * KV_KB_PER_TOK / 1024 / 1024 ))
NEED_GB=$(( MODEL_GB + KV_GB + 6 ))   # +6 GB for activations, vision tower, Metal scratch
info "Budget:  ${MODEL_GB} GB weights + ${KV_GB} GB KV (${KV_QUANT}) + 6 GB overhead = ${NEED_GB} GB"
info "Allowed: ${MEMORY_LIMIT_GB} GB"
if (( NEED_GB > MEMORY_LIMIT_GB )); then
  warn "Configured plan exceeds MEMORY_LIMIT_GB."
  warn "Either lower CONTEXT_WINDOW, set KV_QUANT=q4, or raise MEMORY_LIMIT_GB in env.conf."
  warn "Continuing anyway - start.sh will fail loudly if the model will not load."
else
  ok "Fits with $(( MEMORY_LIMIT_GB - NEED_GB )) GB to spare."
fi

# ── 1. dependencies ──────────────────────────────────────────────────────────
if (( DO_DEPS )); then
  step "1/3  Installing the MTPLX runtime"

  if command -v mtplx >/dev/null 2>&1; then
    ok "mtplx already installed: $(mtplx --version 2>/dev/null | head -1)"
  else
    require_bin brew "Install Homebrew from https://brew.sh, then re-run ./install.sh"
    info "brew install youssofal/mtplx/mtplx"
    brew install youssofal/mtplx/mtplx
  fi

  # First run bootstraps a private Python venv (a few hundred MB of wheels).
  info "Bootstrapping the MTPLX Python runtime (first run only, may take a few minutes)..."
  mtplx --version >/dev/null 2>&1 || true
  ok "MTPLX ready: $(mtplx --version 2>/dev/null | head -1)"

  command -v python3 >/dev/null 2>&1 || warn "python3 not found - ./bench/bench.sh needs it."
else
  step "1/3  Skipping dependency install"
fi

# ── 2. model ─────────────────────────────────────────────────────────────────
if (( DO_MODEL )); then
  step "2/3  Fetching $MODEL_REPO"
  "$REPO_DIR/lib/fetch-model.sh" "$MODEL_REPO" "$MODEL_DIR"

  info "Validating the MTPLX runtime contract..."
  if mtplx inspect "$MODEL_DIR" --require-mtp --json 2>/dev/null \
       | grep -q '"support_level": "verified-native"'; then
    ok "Model is verified-native: MTP head present and contract matched."
  else
    warn "Model did not report verified-native. Inspect it yourself with:"
    warn "    mtplx inspect '$MODEL_DIR' --json"
    warn "If it is only 'architecture-compatible', start.sh will need"
    warn "MODEL_UNVERIFIED_OK=1 in env.conf, and speed is not guaranteed."
  fi
else
  step "2/3  Skipping model download"
fi

# ── 3. speed tune ────────────────────────────────────────────────────────────
if (( DO_TUNE )); then
  step "3/3  Measuring the fastest MTP depth on THIS machine"
  log "     This loads the model 4 times and takes 10-25 minutes."
  log "     It writes the winner to run/tuning.json; MTP_DEPTH=auto uses it."
  log ""

  if [[ ! -d "$MODEL_DIR" ]]; then
    warn "Model directory missing, skipping tune: $MODEL_DIR"
  elif mtplx tune --model "$MODEL_DIR" --retune --depths 1,2,3 \
        --max-tokens 256 --json > "$RUN_DIR/tune.raw.json" 2> "$RUN_DIR/tune.err"; then
    cp "$RUN_DIR/tune.raw.json" "$TUNE_FILE"
    DEPTH="$(tuned_depth)"
    ok "Fastest MTP depth on this Mac: ${DEPTH:-unknown}"
  else
    warn "Auto-tune did not complete. See $RUN_DIR/tune.err"
    warn "The server will fall back to MTP depth 3, which is a good default."
    tail -n 5 "$RUN_DIR/tune.err" 2>/dev/null || true
  fi
else
  step "3/3  Skipping speed tune (MTP depth will default to 3)"
fi

# ── 4. wired memory ceiling (optional) ───────────────────────────────────────
if (( WIRED_LIMIT_GB > 0 )); then
  step "Optional: raising the macOS GPU wired-memory ceiling"
  CURRENT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
  TARGET_MB=$(( WIRED_LIMIT_GB * 1024 ))
  if (( CURRENT_MB >= TARGET_MB )); then
    ok "Already at ${CURRENT_MB} MB (target ${TARGET_MB} MB)."
  elif (( TARGET_MB > RAM_GB * 1024 * 90 / 100 )); then
    warn "WIRED_LIMIT_GB=${WIRED_LIMIT_GB} is over 90% of RAM. Refusing to apply."
    warn "That configuration causes jetsam kills and can leak wired pages on a hard kill."
  else
    warn "This needs sudo once and does NOT survive a reboot."
    warn "Command: sudo sysctl iogpu.wired_limit_mb=${TARGET_MB}"
    sudo sysctl "iogpu.wired_limit_mb=${TARGET_MB}" \
      && ok "Wired limit set to ${TARGET_MB} MB for this boot." \
      || warn "Could not set the wired limit; macOS default stands."
  fi
fi

# ── done ─────────────────────────────────────────────────────────────────────
step "Install complete"
cat <<EOF

  Start the server:     ./start.sh
  Check it:             ./status.sh
  Measure tok/s:        ./bench/bench.sh
  Re-tune MTP depth:    ./bench/bench.sh --tune
  Stop it:              ./stop.sh

  Config lives in:      env.conf

EOF
