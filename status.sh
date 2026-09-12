#!/usr/bin/env bash
# Live dashboard for the agent endpoint: CPU, GPU, ANE, memory, and the
# server's concurrent activity, refreshed in real time.
#
#   ./status.sh                live dashboard (Ctrl-C to exit)
#   ./status.sh --once         one-shot summary, for scripts and logs
#   ./status.sh --json         machine-readable snapshot
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
  --once|--json|--power|--interval|*)
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
    "api_key_short": (open("${API_KEY_FILE}").read().strip()[:12] + "...") if os.path.exists("${API_KEY_FILE}") else "none",
    "pid_file": "${PID_FILE}",
    "model_dir": "${MODEL_DIR}",
    "model": "${MODEL}",
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
