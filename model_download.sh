#!/usr/bin/env bash
# Download uncensored models and switch between them.
#
#   ./model_download.sh                 list known models and what is on disk
#   ./model_download.sh 9b              download one
#   ./model_download.sh moe             (any alias, or an owner/name repo id)
#   ./model_download.sh --all           download every known model
#   ./model_download.sh --switch 9b     download if needed, set it as the
#                                       default in env.conf, and restart
#
# Every known model is an uncensored fine-tune. UpinelAIOS does not ship or
# suggest aligned models.
#
# Downloads resume, so re-running after an interruption is safe and cheap.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
load_config

case "${1:-}" in
  -h|--help) show_usage "$0"; exit 0 ;;
esac

size_hint_for() {
  # Order matters: bash case takes the FIRST match, so the specific
  # "35B-A3B" pattern has to come before the generic "-MTPLX-4bit" one or the
  # MoE is reported at the dense model's size.
  case "$1" in
    *35B-A3B*|*26B-A4B*|*A10B*|*122B*) echo "22 GB+" ;;
    *HauhauCS-Aggressive-MTPLX-6bit*)   echo "23 GB" ;;
    *HauhauCS-Aggressive-MTPLX-4bit*)   echo "15 GB" ;;
    *MTPLX-3bit*)                       echo "14 GB" ;;
    *MTPLX-6bit*)                       echo "23 GB" ;;
    *MTPLX-4bit*)                       echo "17 GB" ;;
    *9B*|*9b*)                          echo "6 GB" ;;
    *)                                  echo "unknown" ;;
  esac
}

model_on_disk() {
  local dir="$MODELS_DIR/${1//\//--}"
  [[ -d "$dir" ]] || return 1
  find "$dir" -name '*.safetensors' -type f 2>/dev/null | grep -q . || return 1
  return 0
}

disk_usage() {
  du -sh "$MODELS_DIR/${1//\//--}" 2>/dev/null | awk '{print $1}'
}

list_models() {
  step "Known uncensored models"
  printf '  %-10s %-8s %-9s %s\n' "ALIAS" "SIZE" "ON DISK" "REPO"
  printf '  %-10s %-8s %-9s %s\n' "----------" "--------" "---------" "----------------------------------------"
  for alias in $MODEL_ALIASES; do
    repo="$(model_repo_for "$alias")"
    if model_on_disk "$repo"; then
      state="${C_GREEN}yes${C_RESET} $(disk_usage "$repo")"
    else
      state="${C_DIM}no${C_RESET}"
    fi
    marker=""
    [[ "$repo" == "$MODEL_REPO" ]] && marker="  ${C_CYAN}<- current${C_RESET}"
    printf "  %-10s %-8s %-9b %s%b\n" "$alias" "$(size_hint_for "$repo")" "$state" "$repo" "$marker"
  done
  log ""
  log "  ${C_DIM}current selection: $MODEL_REPO${C_RESET}"
  log "  ${C_DIM}models live in:    $MODELS_DIR${C_RESET}"
  log ""
  log "  Download one:   ./model_download.sh 9b"
  log "  Switch to one:  ./model_download.sh --switch 9b"
  log "  ${C_DIM}Any owner/name Hugging Face repo also works, if it is an MTPLX pack.${C_RESET}"
}

download_one() {
  local want="$1"
  local repo; repo="$(model_repo_for "$want")" || return 1
  local dir="$MODELS_DIR/${repo//\//--}"

  if model_on_disk "$repo"; then
    ok "$want is already on disk ($(disk_usage "$repo"))"
    return 0
  fi

  step "Downloading $want"
  log "  repo: $repo"
  log "  size: about $(size_hint_for "$repo")"
  log ""
  "$REPO_DIR/lib/fetch-model.sh" "$repo" "$dir" || return 1

  info "Checking the MTPLX runtime contract..."
  if mtplx inspect "$dir" --require-mtp --json 2>/dev/null \
       | grep -q '"support_level": "verified-native"'; then
    ok "$want is verified-native: MTP head present and contract matched."
  else
    warn "$want did not report verified-native - MTP speed is not guaranteed."
    warn "  mtplx inspect '$dir' --json"
  fi
}

switch_to() {
  local want="$1"
  local repo; repo="$(model_repo_for "$want")" || return 1

  if ! model_on_disk "$repo"; then
    download_one "$want" || die "Download failed; not switching."
  fi

  MODEL_REPO="$repo"
  local tune; tune="$(tune_file_for_current_model)"
  log ""
  info "Set MODEL in env.conf to:"
  log "    $repo"
  python3 - "$ENV_FILE" "$repo" <<'PY'
import re, sys
path, repo = sys.argv[1], sys.argv[2]
s = open(path).read()
s, n = re.subn(r'^MODEL=.*$', f'MODEL="{repo}"', s, count=1, flags=re.M)
if n != 1:
    raise SystemExit(f"could not set MODEL in {path}")
open(path, 'w').write(s)
PY
  ok "env.conf updated."

  if [[ -f "$tune" ]]; then
    ok "A tuned MTP depth exists for this model (depth $(tuned_depth || echo '?'))."
  else
    warn "No MTP depth has been measured for this model on this Mac."
    warn "MTP_DEPTH=auto will use the runtime default of 3."
    warn "Measure it properly with: ./bench/bench.sh --tune"
  fi

  log ""
  if pid_alive || [[ -n "$(port_pids)" ]]; then
    info "Restarting to pick it up..."
    "$REPO_DIR/restart.sh"
  else
    info "Nothing running. Start it with ./start.sh"
  fi
}

# ── dispatch ─────────────────────────────────────────────────────────────────
case "${1:-}" in
  "")
    list_models
    ;;
  --all)
    step "Downloading every known model"
    failed=0
    for alias in $MODEL_ALIASES; do
      download_one "$alias" || { warn "failed: $alias"; failed=1; }
    done
    log ""
    (( failed )) && die "Some downloads failed. Re-run to resume."
    ok "All known models are on disk."
    ;;
  --switch)
    [[ -n "${2:-}" ]] || die "--switch needs a model, e.g. ./model_download.sh --switch 9b"
    switch_to "$2"
    ;;
  -*)
    die "Unknown argument: $1  (try --help)"
    ;;
  *)
    download_one "$1"
    ;;
esac
