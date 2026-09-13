#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  dev@upinel.com  ·  upinel.com
#  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.
#
#  Upinel Personal Free License: free for personal use, and free for creators
#  (YouTubers, KOLs) to make content with - just email dev@upinel.com to say so.
#  Other commercial use needs written permission. Derivatives must credit the
#  author. Covers this project's own code only. See LICENSE.
#
#  "Make it work, make it right, make it fast - then measure it, because
#   the third one is only a claim until the numbers agree."
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup for the UpinelAIOS-MLX bundle.
#
#   ./install.sh                 scan hardware, suggest settings, install
#   ./install.sh --yes           accept the suggested settings without asking
#   ./install.sh --no-tune       skip the (slow) MTP depth measurement
#   ./install.sh --deps-only     only install the MTPLX runtime
#   ./install.sh --model-only    only download and verify the model
#   ./install.sh --scan-only     print the hardware scan and suggestions, stop
#   ./install.sh --model REPO    use this model instead of env.conf's
#
# Re-running is safe: downloads resume and finished steps are skipped.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config
# shellcheck source=lib/preflight.sh
source "$REPO_DIR/lib/preflight.sh"

DO_DEPS=1; DO_MODEL=1; DO_TUNE=1; DO_SCAN=1
ASSUME_YES=0
MODEL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)     ASSUME_YES=1 ;;
    --no-scan)    DO_SCAN=0 ;;
    --scan-only)  DO_SCAN=1; DO_DEPS=0; DO_MODEL=0; DO_TUNE=0 ;;
    --deps-only)  DO_MODEL=0; DO_TUNE=0 ;;
    --model-only) DO_DEPS=0; DO_TUNE=0; DO_SCAN=0 ;;
    --no-tune)    DO_TUNE=0 ;;
    --model)      MODEL_OVERRIDE="$2"; shift ;;
    -h|--help)    show_usage "$0"; exit 0 ;;
    *)            die "Unknown argument: $1  (try --help)" ;;
  esac
  shift
done

step "UpinelAIOS-MLX - Upinel's One-Click AI Agent Server OS for Mac (MLX)"

is_apple_silicon || die "This bundle targets Apple Silicon Macs. Detected: $(uname -s)/$(uname -m)."
require_macos

# ── 1. hardware scan and configuration ───────────────────────────────────────
if (( DO_SCAN )); then
  run_preflight "$ASSUME_YES" || die "Setup stopped. Fix the issue above and re-run ./install.sh"
  # load_config already re-ran inside apply_config; refresh derived values.
  MODEL_REPO="$(model_repo_for "$MODEL")"
  MODEL_DIR="$MODELS_DIR/${MODEL_REPO//\//--}"
else
  step "Skipping the hardware scan (--model-only)"
fi

# A --model flag beats both env.conf and the scan, but is not written back:
# it is a one-off choice for this run.
if [[ -n "$MODEL_OVERRIDE" ]]; then
  MODEL_REPO="$(model_repo_for "$MODEL_OVERRIDE")"
  MODEL_DIR="$MODELS_DIR/${MODEL_REPO//\//--}"
  info "Overriding the model for this run: $MODEL_REPO"
  info "To make it permanent, set MODEL in env.conf."
fi

if (( DO_MODEL || DO_TUNE )); then
  log ""
  log "  Serving:  $MODEL_REPO"
  log "  Context:  $CONTEXT_WINDOW tokens   KV: $KV_QUANT   profile: $PROFILE"
fi

# ── 2. dependencies ──────────────────────────────────────────────────────────
if (( DO_DEPS )); then
  step "Installing the MTPLX runtime"

  if command -v mtplx >/dev/null 2>&1; then
    ok "mtplx already installed: $(mtplx --version 2>/dev/null | head -1)"
  else
    require_bin brew "Install Homebrew from https://brew.sh, then re-run ./install.sh"
    info "brew install youssofal/mtplx/mtplx"
    brew install youssofal/mtplx/mtplx
  fi

  info "Bootstrapping the MTPLX Python runtime (first run only, a few minutes)..."
  mtplx --version >/dev/null 2>&1 || true
  ok "MTPLX ready: $(mtplx --version 2>/dev/null | head -1)"

  command -v python3 >/dev/null 2>&1 || warn "python3 not found - ./bench/bench.sh needs it."

  # Every Python file must actually parse with the python3 we are about to use.
  #
  # This runs at install time because a syntax error can be version-dependent:
  # an f-string with a backslash in its replacement field (f"{'\u2591' * n}")
  # parses on Python 3.12+ and is a hard SyntaxError on anything older. A file
  # can therefore work perfectly on the machine it was written on and fail on a
  # user's the moment they run ./status.sh. Checking all of them here turns
  # that into a clear message at install time instead of a stack trace later.
  if command -v python3 >/dev/null 2>&1; then
    BADPY="$(python3 - "$REPO_DIR" <<'PYCHECK'
import os, sys
root = sys.argv[1]
bad = []
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "models", "run")]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(base, f)
        try:
            compile(open(p, encoding="utf-8").read(), p, "exec")
        except SyntaxError as e:
            bad.append(f"{os.path.relpath(p, root)}:{e.lineno}: {e.msg}")
print("\n".join(bad))
PYCHECK
)"
    if [[ -n "$BADPY" ]]; then
      warn "These Python files do not parse with $(python3 -V 2>&1):"
      printf '%s\n' "$BADPY" | while IFS= read -r l; do log "    $l"; done
      die "Fix the syntax or use a newer python3; the dashboard and bench tools will not run."
    fi
    ok "Python files parse with $(python3 -V 2>&1 | cut -d' ' -f2)"
  fi

else
  step "Skipping dependency install"
fi

# ── 3. model ─────────────────────────────────────────────────────────────────
if (( DO_MODEL )); then
  step "Fetching the model"
  "$REPO_DIR/lib/fetch-model.sh" "$MODEL_REPO" "$MODEL_DIR"

  info "Validating the MTPLX runtime contract..."
  if mtplx inspect "$MODEL_DIR" --require-mtp --json 2>/dev/null \
       | grep -q '"support_level": "verified-native"'; then
    ok "Verified-native: MTP head present and the runtime contract matches."
  else
    warn "This model did not report verified-native. Check it yourself:"
    warn "    mtplx inspect '$MODEL_DIR' --json"
    warn "If it only reports 'architecture-compatible', MTP speed is not guaranteed."
    warn "Repos published through 'mtplx forge' are the ones with a verified contract."
  fi
else
  step "Skipping model download"
fi

# ── 4. speed tune ────────────────────────────────────────────────────────────
if (( DO_TUNE )); then
  step "Measuring the fastest MTP depth on THIS machine"
  log "     Loads the model 4 times; 10-25 minutes."
  log "     Writes the winner to run/tuning.json, which MTP_DEPTH=auto uses."
  log ""

  if [[ ! -d "$MODEL_DIR" ]]; then
    warn "Model directory missing, skipping tune: $MODEL_DIR"
  elif mtplx tune --model "$MODEL_DIR" --retune --depths 1,2,3 \
        --max-tokens 256 --json > "$RUN_DIR/tune.raw.json" 2> "$RUN_DIR/tune.err"; then
    TUNE_FILE="$(tune_file_for_current_model)"
    cp "$RUN_DIR/tune.raw.json" "$TUNE_FILE"
    # Stamp which model this was measured on. Without it a result is
    # ambiguous the moment someone switches MODEL.
    python3 - "$TUNE_FILE" "$MODEL_REPO" <<'PYSTAMP'
import json, sys
path, repo = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception:
    raise SystemExit
d["model_repo"] = repo
json.dump(d, open(path, "w"), indent=2)
PYSTAMP
    DEPTH="$(tuned_depth)"
    ok "Fastest MTP depth on this Mac: ${DEPTH:-unknown}"
    python3 - "$TUNE_FILE" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
print()
print("     {:>5}  {:>9}  {:>7}".format("mode", "tok/s", "vs AR"))
for r in d.get("results", []):
    print("     {:>5}  {:>9.2f}  {:>6.2f}x".format(
        r["mode"], r["tok_s"], r["multiplier_vs_ar"]))
PY
  else
    warn "Auto-tune did not complete. See $RUN_DIR/tune.err"
    warn "Falling back to MTP depth 3, which is a good default."
    tail -n 5 "$RUN_DIR/tune.err" 2>/dev/null || true
  fi
else
  step "Skipping the speed tune (MTP depth will use MTPLX's default of 3)"
fi

# ── 5. optional wired-memory ceiling ─────────────────────────────────────────
if (( WIRED_LIMIT_GB > 0 )); then
  step "Raising the macOS GPU wired-memory ceiling"
  CURRENT_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
  TARGET_MB=$(( WIRED_LIMIT_GB * 1024 ))
  if (( CURRENT_MB >= TARGET_MB )); then
    ok "Already at ${CURRENT_MB} MB (target ${TARGET_MB} MB)."
  elif (( TARGET_MB > HW_RAM_GB * 1024 * 90 / 100 )); then
    warn "WIRED_LIMIT_GB=${WIRED_LIMIT_GB} is over 90% of RAM. Refusing to apply."
    warn "That setting causes jetsam kills and can leak wired pages on a hard kill."
  else
    warn "Needs sudo once, and does NOT survive a reboot."
    sudo sysctl "iogpu.wired_limit_mb=${TARGET_MB}" \
      && ok "Wired limit set to ${TARGET_MB} MB for this boot." \
      || warn "Could not set it; the macOS default stands."
  fi
fi

# ── done ─────────────────────────────────────────────────────────────────────
if (( DO_DEPS == 0 && DO_MODEL == 0 && DO_TUNE == 0 )); then
  step "Scan complete"
  log "  Review env.conf, then run:  ./install.sh"
  exit 0
fi

step "Install complete"
cat <<EOF

  Start the server:      ./start.sh
  Watch it live:         ./status.sh
  Verify tool calling:   ./bench/verify-tools.sh
  Measure tok/s:         ./bench/bench.sh
  Re-tune MTP depth:     ./bench/bench.sh --tune
  Apply env.conf changes: ./restart.sh
  Stop it:               ./stop.sh

  Config lives in:       env.conf

EOF
