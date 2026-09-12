#!/usr/bin/env bash
# Live dashboard for the agent endpoint: CPU, GPU, ANE, memory, and the
# server's concurrent activity, refreshed in real time.
#
#   ./status.sh                live dashboard (Ctrl-C to exit)
#   ./status.sh --once         one-shot summary, for scripts and logs
#   ./status.sh --json         machine-readable snapshot
#   ./status.sh --key          print only the API key, for scripting
#   ./status.sh --thinking off|minimal|low|medium|high   change thinking LIVE
#   ./status.sh --no-keys      display only; disable the t/m key toggles
#
# While it runs, these keys work:
#   t   cycle the thinking level        m   cycle the downloaded models
#   Enter apply now   Esc cancel        q   quit
#
# A toggle arms a 2-second countdown and applies when it expires, so pressing
# the key again moves to the next option without committing to the last one.
#   ./status.sh --thinking     show the current thinking setting
#   ./status.sh --interval 2   slower refresh
#   ./status.sh --power        add real ANE/GPU power (needs passwordless sudo)
#
# ANE note: Apple exposes the Neural Engine only through powermetrics, which
# needs root. MLX is GPU-only anyway, so the ANE is genuinely idle here and is
# shown as "n/a" rather than faked. --power reads the real figure when sudo is
# already passwordless.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

case "${1:-}" in
  -h|--help)
    show_usage "$0"
    exit 0 ;;
  --key)
    # Just the key, so it can be captured: export KEY=$(./status.sh --key)
    ensure_api_key
    printf '%s\n' "$API_KEY"
    exit 0 ;;
  --thinking)
    # Change thinking on the fly. The model stays loaded; only the decode
    # policy changes, so this is instant and does not disturb in-flight work.
    require_bin curl "curl is required."
    if ! server_healthy; then
      die "No server answering on port $PORT. Start it with ./start.sh"
    fi
    if [[ -z "${2:-}" ]]; then
      CUR="$(live_settings_get | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('unknown'); raise SystemExit
mode = d.get('reasoning','?')
effort = d.get('reasoning_effort') or ''
# effort only means anything while reasoning is on
print(mode if mode == 'off' else (mode + ' ' + effort).strip())
" 2>/dev/null)"
      log "  live thinking: ${C_BOLD}${CUR}${C_RESET}"
      log "  levels: off | minimal | low | medium | high"
      log "  ${C_DIM}restart-persistent default is THINKING=\"$THINKING\" in env.conf${C_RESET}"
      exit 0
    fi
    LEVEL="$2"
    thinking_level_ok "$LEVEL" || die "THINKING level must be off | minimal | low | medium | high"
    RESP="$(live_settings_set "$(live_thinking_payload "$LEVEL")")"
    if [[ -z "$RESP" ]]; then
      die "The server rejected the change. Check run/server.log"
    fi
    NOW="$(printf '%s' "$RESP" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('?'); raise SystemExit
mode = d.get('reasoning','?'); effort = d.get('reasoning_effort') or ''
print(mode if mode == 'off' else (mode + ' ' + effort).strip())
" 2>/dev/null)"
    ok "Thinking is now: ${C_BOLD}$NOW${C_RESET}"
    log ""
    case "$LEVEL" in
      off)     log "  ${C_DIM}No thinking at all. Fastest, and the right default for tool loops.${C_RESET}" ;;
      minimal) log "  ${C_DIM}Thinking is bounded to $(thinking_budget_for minimal) tokens with an early stop.${C_RESET}" ;;
      low)     log "  ${C_DIM}Thinking is bounded to $(thinking_budget_for low) tokens.${C_RESET}" ;;
      medium)  log "  ${C_DIM}Thinking is bounded to $(thinking_budget_for medium) tokens.${C_RESET}" ;;
      high)    log "  ${C_YELLOW:-}Thinking is unbounded. Expect the slowest, most deliberative output.${C_RESET}" ;;
    esac
    log ""
    log "  ${C_DIM}This is live only. To make it survive a restart, set THINKING in env.conf.${C_RESET}"
    exit 0 ;;
  *)
    ;;
esac

require_bin python3 "python3 is required for the dashboard."

# Hand the configuration to the dashboard as JSON so we keep one source of
# truth (env.conf) instead of duplicating parsing in Python.
CONFIG_JSON="$(python3 - <<PY
import json, os
print(json.dumps({
    "base": "http://127.0.0.1:${PORT}/v1",
    "lan_url": "http://$(lan_ip):${PORT}/v1",
    "api_key": open("${API_KEY_FILE}").read().strip() if os.path.exists("${API_KEY_FILE}") else "",
    "api_key_file": "${API_KEY_FILE}",
    "pid_file": "${PID_FILE}",
    "log_file": "${LOG_FILE}",
    "error_log": "$RUN_DIR/dashboard.err",
    "models_dir": "${MODELS_DIR}",
    "repo_dir": "${REPO_DIR}",
    "env_file": "${ENV_FILE}",
    "model_dir": "${MODEL_DIR}",
    "model": "${MODEL}",
    "served_name": "${SERVED_MODEL_NAME}",
    "model_repo": "${MODEL_REPO}",
    "context": "${CONTEXT_WINDOW}",
    "kv": "${KV_QUANT}",
    "profile": "${PROFILE}",
    "thinking": "${THINKING}",
    "preserve_thinking": "${PRESERVE_THINKING}",
    "depth": "$(effective_depth)",
    "memory_limit": "${MEMORY_LIMIT_GB}",
    "batching": "${BATCHING_PRESET}",
    "host": "${HOST}",
    "port": "${PORT}",
    "chip": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')",
    "macos": "$(macos_version)",
}))
PY
)"
export MTPLX_DASH_CFG="$CONFIG_JSON"

exec python3 "$REPO_DIR/lib/dashboard.py" "$@"
