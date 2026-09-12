#!/usr/bin/env bash
# Stop the Qwen3.8-27B agent endpoint.
#
#   ./stop.sh            graceful shutdown (recommended)
#   ./stop.sh --force    SIGKILL - only if graceful stop hangs
#
# Why graceful matters: a process holding a large wired MLX allocation that
# is SIGKILLed can leak those wired pages at the kernel level until reboot.
# Always prefer the graceful path.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

FORCE=0
case "${1:-}" in
  --force)      FORCE=1 ;;
  -h|--help)    show_usage "$0"; exit 0 ;;
  "")           ;;
  *)            die "Unknown argument: $1" ;;
esac

STOPPED=0

# 1. The server we started ourselves.
if pid_alive; then
  PID="$(cat "$PID_FILE")"
  if (( FORCE )); then
    warn "Sending SIGKILL to pid $PID"
    kill -9 "$PID" 2>/dev/null || true
  else
    info "Stopping pid $PID (SIGTERM)..."
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      warn "Still alive after 30s; sending SIGKILL."
      kill -9 "$PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
  ok "Stopped our server."
  STOPPED=1
fi

# 2. Anything else holding the port - e.g. an MTPLX daemon started by hand
#    or by the menu-bar app.
for P in $(port_pids); do
  warn "Port $PORT is also held by pid $P (started outside this bundle)."
  if (( FORCE )); then
    kill -9 "$P" 2>/dev/null || true
  else
    kill -TERM "$P" 2>/dev/null || true
  fi
  ok "Signalled pid $P."
  STOPPED=1
done

if (( STOPPED == 0 )); then
  info "Nothing was running on port $PORT."
else
  # Give macOS a moment to release the wired GPU allocation.
  sleep 2
  if server_healthy; then
    warn "Something is still answering /health. Try ./stop.sh --force"
  else
    ok "Port $PORT is free."
  fi
fi
