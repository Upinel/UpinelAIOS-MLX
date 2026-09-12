#!/usr/bin/env bash
# Show what the endpoint is doing: config, process, live decode stats, memory.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

case "${1:-}" in
  -h|--help) show_usage "$0"; exit 0 ;;
  "")        ;;
  *)         die "Unknown argument: $1" ;;
esac

step "Qwen3.8-27B agent endpoint - status"

# ── configuration ────────────────────────────────────────────────────────────
log ""
log "${C_BOLD}Configuration${C_RESET}  (env.conf)"
# MODEL may be an alias or the repo id itself; do not print it twice.
if [[ "$MODEL" == "$MODEL_REPO" ]]; then
  printf '  %-18s %s\n' "model" "$MODEL_REPO"
else
  printf '  %-18s %s\n' "model" "$MODEL  ($MODEL_REPO)"
fi
printf '  %-18s %s\n' "context window" "$CONTEXT_WINDOW tokens"
printf '  %-18s %s\n' "KV quant"       "$KV_QUANT"
printf '  %-18s %s\n' "MTP depth"      "$(effective_depth)  (MTP_DEPTH=$MTP_DEPTH)"
printf '  %-18s %s\n' "profile"        "$PROFILE"
printf '  %-18s %s\n' "thinking"       "$THINKING"
printf '  %-18s %s\n' "memory cap"     "${MEMORY_LIMIT_GB} GB"
printf '  %-18s %s\n' "session bank"   "$( (( SESSION_BANK_GB > 0 )) && echo "${SESSION_BANK_GB} GB" || echo auto )"
printf '  %-18s %s\n' "batching"       "$BATCHING_PRESET / max $MAX_CONCURRENT"

# Rough resident-memory estimate so users can see their headroom.
KV_KB_PER_TOK="$(kv_kb_per_token)"
KV_GB=$(( CONTEXT_WINDOW * KV_KB_PER_TOK / 1024 / 1024 ))
MODEL_GB="$(model_weight_gb)"
printf '  %-18s %s\n' "est. footprint" "~$(( MODEL_GB + KV_GB )) GB (weights + KV) of ${MEMORY_LIMIT_GB} GB allowed"

# ── hardware ─────────────────────────────────────────────────────────────────
log ""
log "${C_BOLD}Hardware${C_RESET}"
printf '  %-18s %s\n' "chip"    "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
printf '  %-18s %s\n' "memory"  "$(total_ram_gb) GB unified"
printf '  %-18s %s\n' "gpu cores" "$(gpu_cores)"
printf '  %-18s %s\n' "macOS"   "$(macos_version)"
WIRED="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
if [[ "$WIRED" == "0" ]]; then
  printf '  %-18s %s\n' "wired limit" "macOS default (~75% of RAM)"
else
  printf '  %-18s %s\n' "wired limit" "${WIRED} MB"
fi

# ── process ──────────────────────────────────────────────────────────────────
log ""
log "${C_BOLD}Process${C_RESET}"
if pid_alive; then
  PID="$(cat "$PID_FILE")"
  RSS_MB="$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%d", $1/1024}')"
  CPU="$(ps -o %cpu= -p "$PID" 2>/dev/null | tr -d ' ')"
  printf '  %-18s %s\n' "state" "running (pid $PID)"
  printf '  %-18s %s\n' "resident" "${RSS_MB} MB"
  printf '  %-18s %s\n' "cpu" "${CPU}%"
elif server_healthy; then
  log "  running, but not started by this bundle (no pid file)."
else
  log "  not running. Start it with ./start.sh"
fi

# ── live health ──────────────────────────────────────────────────────────────
if server_healthy; then
  log ""
  log "${C_BOLD}Server${C_RESET}"
  HEALTH="$(health_json)"
  python3 - "$HEALTH" <<'PY' 2>/dev/null || log "  (could not parse /health)"
import json, sys
try:
    h = json.loads(sys.argv[1])
except Exception:
    raise SystemExit(1)

def show(label, value, suffix=""):
    if value is None:
        return
    print(f"  {label:<18} {value}{suffix}")

for key, label, suffix in [
    ("status", "status", ""),
    ("model", "loaded model", ""),
    ("model_id", "served as", ""),
    ("context_window", "context window", ""),
    ("profile", "profile", ""),
    ("generation_mode", "generation", ""),
    ("depth", "mtp depth", ""),
    ("mtp_depth", "mtp depth", ""),
    ("reasoning_effort", "thinking effort", ""),
    ("scheduler_mode", "scheduler", ""),
    ("max_active_requests", "max concurrent", ""),
    ("paged_kv_quantization", "kv quant", ""),
    ("kv_quantization", "kv quant", ""),
]:
    if key not in h:
        continue
    value = h[key]
    # Some fields are deep configuration objects (the profile reports its whole
    # env block). Show the name, not the wall of text.
    if isinstance(value, dict):
        value = value.get("name") or value.get("runtime_profile") or "(configured)"
    if isinstance(value, str) and len(value) > 80:
        value = value[:77] + "..."
    show(label, value, suffix)

for section in ("memory", "memory_plan", "cache", "session_bank"):
    sub = h.get(section)
    if isinstance(sub, dict):
        print(f"  -- {section} --")
        for k, v in list(sub.items())[:8]:
            if isinstance(v, (int, float, str, bool)) or v is None:
                show(k, v)
PY
fi

# ── endpoint ─────────────────────────────────────────────────────────────────
log ""
log "${C_BOLD}Endpoints${C_RESET}"
LAN="$(lan_ip)"
log "  LAN base URL     http://${LAN}:${PORT}/v1      (OpenAI-compatible)"
log "  Health           http://${LAN}:${PORT}/health"
log "  Metrics          http://${LAN}:${PORT}/metrics"
if [[ -s "$API_KEY_FILE" ]]; then
  log "  API key          $(cat "$API_KEY_FILE")"
else
  log "  API key          none (loopback-only bind)"
fi

log ""
log "  Quick test:"
log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
log "      -H 'Content-Type: application/json' \\"
if [[ -s "$API_KEY_FILE" ]]; then
  log "      -H 'Authorization: Bearer \$(cat $API_KEY_FILE)' \\"
fi
log '      -d '"'"'{"model":"'"$SERVED_MODEL_NAME"'","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'"'"''
log ""
