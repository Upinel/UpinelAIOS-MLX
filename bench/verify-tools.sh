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
# Verify that the running endpoint calls tools correctly, in every format an
# agent uses. Run this first if an agent "cannot write files".
#
#   ./bench/verify-tools.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
load_config

case "${1:-}" in
  -h|--help) show_usage "$0"; exit 0 ;;
  "")        ;;
  *)         die "Unknown argument: $1" ;;
esac

if ! server_healthy; then
  die "No server answering on port $PORT. Start it with ./start.sh"
fi

KEY_ARG=()
[[ -s "$API_KEY_FILE" ]] && KEY_ARG=( --api-key-file "$API_KEY_FILE" )

exec python3 "$REPO_DIR/bench/verify-tools.py" \
  --url "http://127.0.0.1:${PORT}" \
  --model "$SERVED_MODEL_NAME" \
  "${KEY_ARG[@]}"
