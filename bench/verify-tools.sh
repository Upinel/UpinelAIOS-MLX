#!/usr/bin/env bash
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
