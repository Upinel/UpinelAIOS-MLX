#!/usr/bin/env bash
# Interactive chat against the running UpinelAIOS-MLX endpoint.
#
#   ./chat.sh                      start chatting
#   ./chat.sh --system "..."       set the system prompt
#   ./chat.sh --thinking on        start with thinking on
#   ./chat.sh --temp 0.2           sampling temperature
#   ./chat.sh --no-stream          wait for whole replies instead of streaming
#   ./chat.sh --max-tokens 4096    reply ceiling
#
# The chat client is lib/chat.py, shared with the sister GGUF project. This
# wrapper resolves the running server's address, model id and API key from
# env.conf so the client does not need to know the project layout - the same
# split as ./status.sh.
#
# Requires the server to be up: ./start.sh, and check ./status.sh.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

SYSTEM=""
THINKING_START="$THINKING"
TEMP=""
STREAM=1
MAXTOK="$MAX_RESPONSE_TOKENS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)     SYSTEM="${2:-}"; shift 2 ;;
    --thinking)   THINKING_START="${2:-}"; shift 2 ;;
    --temp)       TEMP="${2:-}"; shift 2 ;;
    --max-tokens) MAXTOK="${2:-}"; shift 2 ;;
    --no-stream)  STREAM=0; shift ;;
    -h|--help)    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            die "Unknown argument: $1  (try --help)" ;;
  esac
done

case "$THINKING_START" in
  off)     THINKING_BOOL=0 ;;
  on)      THINKING_BOOL=1 ;;
  "")      THINKING_BOOL=0 ;;
  *)       THINKING_BOOL=1 ;;   # minimal/low/medium/high all mean "think"
esac

# HOST is often a bind address like 0.0.0.0, which is not something to connect
# to. Chat always goes over loopback.
case "$HOST" in
  0.0.0.0|"") CONNECT_HOST="127.0.0.1" ;;
  *)          CONNECT_HOST="$HOST" ;;
esac

KEY=""
[[ -f "$API_KEY_FILE" ]] && KEY="$(cat "$API_KEY_FILE")"

# Capabilities.
#   stream  MTPLX streams over the OpenAI-compatible API.
# No "thinking" capability here: MTPLX takes the reasoning mode as a live
# server setting rather than a per-request chat_template_kwargs, so /thinking
# reports that and points at ./status.sh --thinking instead of pretending to
# change anything.
if (( STREAM )); then CAPS="stream"; else CAPS=""; fi

export CHAT_BASE="http://${CONNECT_HOST}:${PORT}/v1"
export CHAT_KEY="$KEY"
export CHAT_MODEL="$SERVED_MODEL_NAME"
export CHAT_MAX_TOKENS="$MAXTOK"
export CHAT_THINKING="$([[ $THINKING_BOOL -eq 1 ]] && echo on || echo off)"
export CHAT_CAPS="$CAPS"
export CHAT_SYSTEM="$SYSTEM"
export CHAT_TEMP="${TEMP:-}"

# Arguments are consumed here; chat.py reads the environment.
exec python3 "$REPO_DIR/lib/chat.py"
