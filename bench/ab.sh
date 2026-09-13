#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Nova Upinel Chow, MSc, LLM, BBA, MENSA  ·  upinel@me.com  ·  upinel.com
#  Copyright (c) 2026 Nova Upinel Chow. All rights reserved.
#
#  Upinel Personal Free License: free for personal use, commercial use by
#  written permission, and anything built from this must credit the author.
#  See LICENSE.
#
#  "Make it work, make it right, make it fast - then measure it, because
#   the third one is only a claim until the numbers agree."
# ─────────────────────────────────────────────────────────────────────────────
# A/B a single env.conf setting and print a side-by-side table.
#
#   bench/ab.sh KV_QUANT off q8 --contexts 8192,32768
#   bench/ab.sh PROFILE sustained turbo --contexts 32768
#
# It edits env.conf in place, restarts the server for each arm, and restores
# the original line on exit - even if you Ctrl-C. Long-context arms are slow:
# a 131k prefill takes several minutes on a 64 GB Mac, per arm.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_config

KEY="${1:-}"
[[ -n "$KEY" ]] || { show_usage "$0"; exit 1; }
shift

# Every bare word before the first --flag is an arm value.
VALS=""
CONTEXTS="8192,32768"
MAX_TOKENS=192
while [[ $# -gt 0 ]]; do
  case "$1" in
    --contexts)   CONTEXTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --*)          die "Unknown argument: $1" ;;
    *)            VALS="$VALS $1"; shift ;;
  esac
done
VALS="${VALS# }"
[[ -n "$VALS" ]] || die "No arm values given. Example: bench/ab.sh KV_QUANT off q8"

ORIG_LINE="$(grep -E "^${KEY}=" "$ENV_FILE" || true)"
[[ -n "$ORIG_LINE" ]] || die "$KEY is not set in $ENV_FILE"
ORIG_VALUE="$(printf '%s' "$ORIG_LINE" | sed 's/^[^=]*=//; s/^"//; s/"$//')"

set_env_value() {
  python3 - "$ENV_FILE" "$1" "$2" <<'PY'
import re, sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
s, n = re.subn(rf'^{re.escape(key)}=.*$', f'{key}="{val}"', s, count=1, flags=re.M)
if n != 1:
    raise SystemExit(f"could not set {key} in {path}")
open(path, 'w').write(s)
PY
}

restore_env() {
  set_env_value "$KEY" "$ORIG_VALUE" >/dev/null 2>&1 || true
  warn "Restored $KEY=\"$ORIG_VALUE\" in env.conf"
}
trap restore_env EXIT

RESULT_DIR="$RUN_DIR/ab-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULT_DIR"

step "A/B: $KEY  ($VALS)   contexts=$CONTEXTS   max_tokens=$MAX_TOKENS"

for V in $VALS; do
  step "Arm: $KEY=$V"
  set_env_value "$KEY" "$V"

  "$REPO_DIR/stop.sh" >/dev/null 2>&1 || true
  "$REPO_DIR/start.sh" >/dev/null 2>&1 || true

  if ! server_healthy; then
    warn "Server did not come up with $KEY=$V - skipping this arm."
    tail -n 8 "$LOG_FILE" 2>/dev/null || true
    continue
  fi

  python3 "$REPO_DIR/bench/bench.py" \
    --url "http://127.0.0.1:${PORT}" \
    --model "$SERVED_MODEL_NAME" \
    --api-key-file "$API_KEY_FILE" \
    --contexts "$CONTEXTS" \
    --max-tokens "$MAX_TOKENS" \
    --json > "$RESULT_DIR/${KEY}=${V}.json" 2>&1

  ok "Recorded $RESULT_DIR/${KEY}=${V}.json"
done

step "Results"
python3 "$REPO_DIR/bench/ab_report.py" "$RESULT_DIR" "$KEY" $VALS
info "Raw JSON in $RESULT_DIR"
