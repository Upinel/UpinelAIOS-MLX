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
# Restart the UpinelAIOS-MLX server, reloading env.conf.
#
#   ./restart.sh              graceful stop, then start with the current env.conf
#   ./restart.sh --force      SIGKILL on stop if the graceful path hangs
#   ./restart.sh --wait 5     extra seconds to let the port and GPU settle
#   ./restart.sh --print      show what would start, and whether config changed
#
# Editing env.conf has no effect until the server is restarted, because the
# settings are passed to MTPLX as command-line arguments at launch. This script
# is the supported way to apply them, and it tells you which settings actually
# changed rather than leaving you to diff the file by hand.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

FORCE=0
SETTLE=3
PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)  FORCE=1 ;;
    --wait)   SETTLE="$2"; shift ;;
    --print)  PRINT_ONLY=1 ;;
    -h|--help) show_usage "$0"; exit 0 ;;
    *) die "Unknown argument: $1  (try --help)" ;;
  esac
  shift
done

step "UpinelAIOS-MLX restart"

# ── what is about to change ──────────────────────────────────────────────────
if [[ -f "$CONFIG_SNAPSHOT_FILE" ]] && ! diff -q <(config_fingerprint) \
        "$CONFIG_SNAPSHOT_FILE" >/dev/null 2>&1; then
  log ""
  log "  ${C_BOLD}env.conf changed since the server last started:${C_RESET}"
  # Pair the two sides by key so each setting reads as old -> new rather than
  # as two separated lists the reader has to reconcile. Both sides go in as
  # files: a heredoc and a process substitution cannot share stdin.
  config_fingerprint > "$RUN_DIR/config.pending"
  python3 "$REPO_DIR/lib/diff_config.py" "$CONFIG_SNAPSHOT_FILE" "$RUN_DIR/config.pending"
  rm -f "$RUN_DIR/config.pending"
  log ""
  log "  ${C_DIM}left of the arrow is what is running now; right is what will start.${C_RESET}"
else
  info "env.conf is unchanged since the last start."
fi

EFFECTIVE_DEPTH="$(effective_depth)"

# Ask before printing the banner, not after: otherwise the banner would name
# one model and the restart would load another. start.sh runs the same picker,
# so the choice is handed over as --model to stop it asking twice.
PICKED=0
if (( ! PRINT_ONLY )); then
  if choose_model_on_disk; then
    PICKED=1
    info "Serving $MODEL_REPO for this run. Set MODEL in env.conf to make it permanent."
  fi
fi

log ""
log "  ${C_BOLD}Starting with:${C_RESET}"
log "    model      $MODEL_REPO"
log "    served as  $SERVED_MODEL_NAME"
log "    context    $CONTEXT_WINDOW   KV $KV_QUANT   MTP depth $EFFECTIVE_DEPTH"
log "    profile    $PROFILE   thinking $THINKING   history $PRESERVE_THINKING"
log "    memory     ${MEMORY_LIMIT_GB} GB cap   session bank ${SESSION_BANK_GB} GB"
log "    network    $HOST:$PORT"
log ""

if (( PRINT_ONLY )); then
  info "--print given; not restarting."
  exit 0
fi

# ── stop ─────────────────────────────────────────────────────────────────────
WAS_RUNNING=0
pid_alive && WAS_RUNNING=1
[[ -n "$(port_pids)" ]] && WAS_RUNNING=1

if (( WAS_RUNNING )); then
  if (( FORCE )); then
    "$REPO_DIR/stop.sh" --force
  else
    "$REPO_DIR/stop.sh"
  fi
else
  info "Server was not running; starting it."
fi

# ── settle ───────────────────────────────────────────────────────────────────
# Wait for the port to actually be free, then a few more seconds. macOS needs a
# moment to release the wired GPU allocation, and starting into a port that is
# still closing is the usual cause of a failed restart.
info "Waiting for the port and GPU memory to settle..."
for _ in $(seq 1 30); do
  [[ -z "$(port_pids)" ]] && break
  sleep 1
done
if [[ -n "$(port_pids)" ]]; then
  die "Port $PORT is still held after 30s. Try ./stop.sh --force, then ./start.sh"
fi
sleep "$SETTLE"

# ── start ────────────────────────────────────────────────────────────────────
# start.sh records the new config snapshot itself, so a failed start still
# leaves an accurate "last attempted" record for the next diff.
# Two branches rather than an array: expanding an empty array is an unbound
# variable error in bash 3.2 under `set -u`.
if (( PICKED )); then
  "$REPO_DIR/start.sh" --model "$MODEL_REPO"
else
  "$REPO_DIR/start.sh"
fi
RC=$?

if (( RC != 0 )); then
  warn "Restart failed. The previous state is in $LOG_FILE"
  exit $RC
fi

ok "Restart complete."
