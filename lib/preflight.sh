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
# Hardware scan, configuration recommendation, and env.conf rewriting.
#
# Sourced by install.sh. Bash 3.2 compatible (no associative arrays, no
# ${var,,}) because that is what ships on macOS.

# ── hardware scan ────────────────────────────────────────────────────────────

# All of these are cheap; the whole scan takes well under a second except the
# system_profiler call, which we keep to a single invocation.
scan_hardware() {
  HW_CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  HW_RAM_GB="$(total_ram_gb)"
  HW_GPU_CORES="$(gpu_cores)"
  HW_CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
  HW_MACOS="$(macos_version)"
  HW_MACOS_MAJOR="${HW_MACOS%%.*}"
  HW_MODEL_ID="$(sysctl -n hw.model 2>/dev/null || echo unknown)"

  # Free space on the volume that will hold the weights.
  local probe="$MODELS_DIR"
  [[ -d "$probe" ]] || probe="$(dirname "$probe")"
  [[ -d "$probe" ]] || probe="$REPO_DIR"
  HW_FREE_GB="$(df -g "$probe" 2>/dev/null | awk 'NR==2 {print $4}')"
  [[ -n "${HW_FREE_GB:-}" ]] || HW_FREE_GB=0

  # Is the model already on disk?
  HW_MODEL_PRESENT="no"
  if [[ -d "$MODEL_DIR" ]] && [[ -n "$(find "$MODEL_DIR" -name '*.safetensors' -type f 2>/dev/null | head -1)" ]]; then
    HW_MODEL_PRESENT="yes"
  fi
  HW_WEIGHTS_GB="$(model_weight_gb)"

  # Is MTPLX installed, and is MLX reachable through it?
  HW_MTPLX="not installed"
  if command -v mtplx >/dev/null 2>&1; then
    HW_MTPLX="$(mtplx --version 2>/dev/null | head -1)"
    [[ -n "$HW_MTPLX" ]] || HW_MTPLX="installed"
  fi
  HW_BREW="no"; command -v brew >/dev/null 2>&1 && HW_BREW="yes"
}

print_hardware() {
  local chip_family
  # "Apple M5 Pro" -> "M5"
  chip_family="$(printf '%s' "$HW_CHIP" | awk '{print $2}')"

  log ""
  printf '  %-16s %s\n' "chip"        "$HW_CHIP  (${HW_GPU_CORES} GPU cores, ${HW_CPU_CORES} CPU cores)"
  printf '  %-16s %s\n' "unified memory" "${HW_RAM_GB} GB"
  printf '  %-16s %s\n' "macOS"       "$HW_MACOS  ($HW_MODEL_ID)"
  printf '  %-16s %s\n' "free disk"   "${HW_FREE_GB} GB"
  printf '  %-16s %s\n' "MTPLX"       "$HW_MTPLX"
  printf '  %-16s %s\n' "model on disk" "$HW_MODEL_PRESENT${HW_MODEL_PRESENT:+ }$([[ "$HW_MODEL_PRESENT" == yes ]] && echo "(${HW_WEIGHTS_GB} GB measured)")"

  # The single most useful hardware signal for this workload is memory
  # bandwidth class, which tracks the chip generation more than core count.
  case "$chip_family" in
    M1|M2) log "  $C_DIM note: $chip_family-generation memory bandwidth is roughly half an M5's,$C_RESET"
           log "  $C_DIM expect decode around half the numbers in the README.$C_RESET" ;;
    M3|M4) log "  $C_DIM note: $chip_family is close to the reference M5 Pro for decode rate.$C_RESET" ;;
    *)     ;;
  esac
}

# ── recommendation ───────────────────────────────────────────────────────────
# Sets REC_* variables. The rules encode what was actually measured:
#   * decode needs the weights resident, so MEMORY_LIMIT_GB must clear
#     weights + KV + ~6 GB of activations
#   * KV costs 64/34/18 KB per token for f16/q8/q4 (only 16 of 64 layers cache)
#   * the session bank is the biggest allocator after the weights, and MTPLX
#     auto-sizes it far too generously on a shared desktop
recommend_config() {
  local ram="$HW_RAM_GB"

  # moe is the default, and it is not a close call: the 35B-A3B activates only
  # ~3B parameters per token, so it decodes at ~83 tok/s against 42-51 for the
  # dense 27B, while costing about the same memory. The dense builds are the
  # fidelity picks, not the speed picks.
  #
  # This used to recommend the dense 4bit 27B on every Mac under 96 GB, which
  # meant install.sh would quietly downgrade anyone already on the default.
  if (( ram >= 48 )); then
    REC_MODEL="$(model_repo_for moe)"
    REC_WEIGHTS_GB=22
    REC_REASON_MODEL="35B MoE with ~3B active - the fastest option here (~83 tok/s), and it fits ${ram} GB"
  elif (( ram >= 32 )); then
    REC_MODEL="$(model_repo_for 4bit)"
    REC_WEIGHTS_GB=15
    REC_REASON_MODEL="dense 4-bit 27B: the MoE wants ~30 GB resident, which is most of a ${ram} GB Mac"
  else
    REC_MODEL="$(model_repo_for 9b)"
    REC_WEIGHTS_GB=5
    REC_REASON_MODEL="${ram} GB is below what a 27B wants; the 9B is the only entry here that fits"
  fi

  # Context and KV quant together have to fit the memory budget.
  if (( ram >= 128 )); then
    REC_CONTEXT=262144; REC_KV="off"
    REC_REASON_CTX="maximum context with an unquantized KV cache"
  elif (( ram >= 96 )); then
    REC_CONTEXT=262144; REC_KV="q8"
    REC_REASON_CTX="maximum context; q8 KV keeps it inside the budget"
  elif (( ram >= 56 )); then
    REC_CONTEXT=131072; REC_KV="q8"
    REC_REASON_CTX="128K, with headroom left for your desktop apps"
  elif (( ram >= 40 )); then
    REC_CONTEXT=131072; REC_KV="q8"
    REC_REASON_CTX="128K; expect to close other heavy apps for very long prompts"
  elif (( ram >= 30 )) ; then
    REC_CONTEXT=65536; REC_KV="q8"
    REC_REASON_CTX="64K is the reliable ceiling at ${ram} GB"
  else
    REC_CONTEXT=32768; REC_KV="q4"
    REC_REASON_CTX="${ram} GB is below what a 27B wants; 32K with q4 KV is the safe floor"
  fi

  # Leave roughly a quarter of RAM to macOS and everything else. On a 64 GB
  # Mac that is the 48 GB ceiling the reference machine was tuned at.
  REC_MEMORY_LIMIT_GB=$(( ram * 3 / 4 ))
  (( REC_MEMORY_LIMIT_GB > 8 )) || REC_MEMORY_LIMIT_GB=8

  # Session bank: a fraction of RAM, capped. MTPLX's own auto-sizing picked
  # 16.6 GB on a 64 GB Mac, which pushed wired memory to ~50 GB.
  REC_SESSION_BANK_GB=$(( ram / 8 ))
  (( REC_SESSION_BANK_GB < 2 )) && REC_SESSION_BANK_GB=2
  (( REC_SESSION_BANK_GB > 24 )) && REC_SESSION_BANK_GB=24

  # Prefill chunking: keeps each Metal allocation small. This is what prevents
  # the Metal command-buffer OOM at long context.
  REC_PREFILL_CHUNK=2048

  # Profile: sustained is memory-safe at long context; turbo is faster but only
  # if you stay short.
  if (( REC_CONTEXT > 32768 )); then
    REC_PROFILE="sustained"
    REC_REASON_PROFILE="long-context path; turbo's prefill triggers a Metal OOM past ~32K"
  else
    REC_PROFILE="turbo"
    REC_REASON_PROFILE="short-context path; fastest decode, and safe at ${REC_CONTEXT} tokens"
  fi

  # Model + KV + activation headroom, for the fit check.
  #
  # KV now comes from the model's own architecture rather than one hardcoded
  # curve: the recommended model here is the 35B MoE, not the 27B, and the two
  # do not cache KV on the same number of layers.
  local kv_kb
  kv_kb="$(kv_kb_for "$REC_MODEL" "$REC_KV")"
  REC_KV_GB=$(( REC_CONTEXT * kv_kb / 1024 / 1024 ))
  REC_NEED_GB=$(( REC_WEIGHTS_GB + REC_KV_GB + 6 ))
}

# KV KB/token for one repo id at one quant, from the architecture table.
kv_kb_for() {
  local repo="$1" quant="$2" kb
  kb="$(kv_kb_per_token_f16 "$repo")"
  case "$quant" in
    q4|q4_0) kb=$(( kb / 4 )) ;;
    q8|q8_0) kb=$(( kb / 2 )) ;;
  esac
  echo "$kb"
}

# Does the *current* env.conf fit this machine?
current_config_fits() {
  local cur_kb cur_kv_gb cur_weights cur_need
  cur_kb="$(kv_kb_for "${MODEL_REPO:-$(model_repo_for "$MODEL")}" "$KV_QUANT")"
  cur_kv_gb=$(( CONTEXT_WINDOW * cur_kb / 1024 / 1024 ))
  cur_weights="$(model_weight_gb)"
  cur_need=$(( cur_weights + cur_kv_gb + 6 ))

  CUR_KV_GB="$cur_kv_gb"
  CUR_NEED_GB="$cur_need"

  # It fits if the plan is inside the configured cap AND inside physical RAM.
  (( cur_need <= MEMORY_LIMIT_GB )) && (( cur_need <= HW_RAM_GB ))
}

# ── per-model fit, for the picker ────────────────────────────────────────────
# Download size in GB for each known repo. The table lives in common.sh so the
# picker and model_weight_gb() cannot drift apart.
model_size_gb() {
  model_download_gb "$1"
}

# One-line note about a model, shown beside its verdict.
model_note() {
  case "$1" in
    moe)       echo "35B MoE, ~3B active - the fastest here (~83 tok/s), and the default" ;;
    4bit)      echo "dense 27B, ~50 tok/s - the fidelity pick at well under half the speed" ;;
    6bit)      echo "dense 27B, closest to the original weights; the slowest 27B" ;;
    27b-3bit)  echo "dense 27B squeezed to 3-bit: smallest 27B, some quality loss" ;;
    27b-4bit)  echo "another 27B 4-bit from a different publisher: 1 GB bigger, same shape" ;;
    9b)        echo "dense 9B, ~40 tok/s - the only entry that fits a small Mac" ;;
    *)         echo "" ;;
  esac
}

# What this model would cost on THIS machine, and whether it fits.
# Prints "<need> <verdict>".
model_fit() {
  # Named _mf rather than "alias": alias is a bash builtin, and shadowing it in
  # a function that other code may call is asking for trouble.
  local _mf="$1" repo size_gb kv_gb need_gb
  repo="$(model_repo_for "$_mf" 2>/dev/null)"
  size_gb="$(model_size_gb "$repo")"
  (( size_gb > 0 )) || { echo "? unknown"; return; }
  # KV for this model at the recommended context, scaled by the quant.
  local per_tok; per_tok="$(kv_kb_for "$repo" "$REC_KV")"
  kv_gb=$(( REC_CONTEXT * per_tok / 1024 / 1024 ))
  need_gb=$(( size_gb + kv_gb + 6 ))

  local verdict
  if [[ -n "$REC_ALIAS" && "$_mf" == "$REC_ALIAS" ]]; then
    verdict="RECOMMENDED"
  elif (( need_gb + 6 <= HW_RAM_GB )); then
    verdict="fits comfortably"
  elif (( need_gb <= HW_RAM_GB )); then
    verdict="tight - expect paging"
  else
    verdict="will not fit"
  fi
  echo "${need_gb} ${verdict}"
}

# alias_for_repo() lives in lib/common.sh: the start/restart picker needs it
# too, and it does not source this file.

# Numbered picker. Everything it needs is in globals; it does not read input.
print_model_menu() {
  log ""
  log "  ${C_BOLD}Pick a model${C_RESET}   ${C_DIM}this Mac has ${HW_RAM_GB} GB of unified memory${C_RESET}"
  log ""
  printf '  %3s  %-10s %-6s %-20s %s\n' "#" "ALIAS" "SIZE" "VERDICT" "NOTE"
  printf '  %3s  %-10s %-6s %-20s %s\n' "---" "----------" "------" "--------------------" "----------------------------------------"

  local i=1 alias fit need verdict note colour
  MODEL_MENU_ALIASES=""
  for alias in $MODEL_ALIASES; do
    fit="$(model_fit "$alias")"
    need="${fit%% *}"; verdict="${fit#* }"
    note="$(model_note "$alias")"
    case "$verdict" in
      RECOMMENDED)       colour="$C_GREEN"  ;;
      fits\ comfortably) colour=""           ;;
      tight*)            colour="$C_YELLOW" ;;
      will\ not\ fit)    colour="$C_RED"    ;;
      *)                 colour="$C_DIM"    ;;
    esac
    printf '  %3d  %-10s %-6s %s%-20s%s %s%s%s\n' \
      "$i" "$alias" \
      "$(model_size_gb "$(model_repo_for "$alias" 2>/dev/null)") GB" \
      "$colour" "$verdict" "$C_RESET" "$C_DIM" "$note" "$C_RESET"
    MODEL_MENU_ALIASES="$MODEL_MENU_ALIASES $alias"
    i=$(( i + 1 ))
  done

  log ""
  log "  ${C_DIM}Enter a number, or press Enter to keep your current model.${C_RESET}"
  log "  ${C_DIM}A model that will not fit can still be chosen - it will just be slow,${C_RESET}"
  log "  ${C_DIM}or fail to load. ./model_download.sh fetches it afterwards.${C_RESET}"
  log ""
  printf '  Model number: '
}

# ── display ──────────────────────────────────────────────────────────────────
print_recommendation() {
  log ""
  printf '  %-24s %-34s %s\n' "SETTING" "CURRENT (env.conf)" "SUGGESTED"
  printf '  %-24s %-34s %s\n' "------------------------" "----------------------------------" "----------------------------------"

  # Repo ids are far longer than the column; shorten the head and keep the
  # meaningful tail (the quant tag) so the table stays readable.
  short() {
    local v="$1"
    if (( ${#v} <= 32 )); then printf '%s' "$v"; return; fi
    local owner="${v%%/*}" tail="${v##*/}"
    if (( ${#tail} > 24 )); then tail="...${tail: -21}"; fi
    printf '%s/%s' "$owner" "$tail"
  }

  row() {   # name, current, suggested
    local name="$1" cur="$2" sug="$3"
    local mark=""
    [[ "$cur" == "$sug" ]] && mark="  ${C_DIM}(unchanged)${C_RESET}"
    printf '  %-24s %-34s %s%s\n' "$name" "$(short "$cur")" "$(short "$sug")" "$mark"
  }

  row "MODEL"              "$MODEL"              "$REC_MODEL"
  row "CONTEXT_WINDOW"     "$CONTEXT_WINDOW"     "$REC_CONTEXT"
  row "KV_QUANT"           "$KV_QUANT"           "$REC_KV"
  row "MEMORY_LIMIT_GB"    "$MEMORY_LIMIT_GB"    "$REC_MEMORY_LIMIT_GB"
  row "SESSION_BANK_GB"    "$SESSION_BANK_GB"    "$REC_SESSION_BANK_GB"
  row "PREFILL_CHUNK_TOKENS" "$PREFILL_CHUNK_TOKENS" "$REC_PREFILL_CHUNK"
  row "PROFILE"            "$PROFILE"            "$REC_PROFILE"

  if [[ "$MODEL" != "$REC_MODEL" ]]; then
    log ""
    log "  Full model ids:"
    log "    now       $MODEL"
    log "    suggested $REC_MODEL"
  fi

  log ""
  log "  ${C_BOLD}Why these values:${C_RESET}"
  log "    model    $REC_REASON_MODEL"
  log "    context  $REC_REASON_CTX"
  log "    profile  $REC_REASON_PROFILE"
  log ""
  log "  Projected footprint: ${REC_WEIGHTS_GB} GB weights + ${REC_KV_GB} GB KV ($REC_KV)"
  log "                       = ${REC_NEED_GB} GB of a ${REC_MEMORY_LIMIT_GB} GB cap on ${HW_RAM_GB} GB of RAM"
}

# ── apply ────────────────────────────────────────────────────────────────────
# Rewrite only the keys we recommend, leaving every comment intact.
apply_config() {
  python3 - "$ENV_FILE" \
    "MODEL=$REC_MODEL" \
    "CONTEXT_WINDOW=$REC_CONTEXT" \
    "KV_QUANT=$REC_KV" \
    "MEMORY_LIMIT_GB=$REC_MEMORY_LIMIT_GB" \
    "SESSION_BANK_GB=$REC_SESSION_BANK_GB" \
    "PREFILL_CHUNK_TOKENS=$REC_PREFILL_CHUNK" \
    "PROFILE=$REC_PROFILE" <<'PY'
import re, sys

path = sys.argv[1]
changes = [a.split("=", 1) for a in sys.argv[2:]]
src = open(path).read()

# Keys whose value is a plain string needing quotes in env.conf.
QUOTED = {"MODEL", "KV_QUANT", "PROFILE"}
applied, missing = [], []

for key, raw in changes:
    value = f'"{raw}"' if key in QUOTED else raw
    new_src, n = re.subn(rf'^{re.escape(key)}=.*$', f'{key}={value}',
                         src, count=1, flags=re.M)
    if n == 1:
        src = new_src
        applied.append(f"{key}={raw}")
    else:
        missing.append(key)

open(path, 'w').write(src)
print("  applied: " + ", ".join(applied))
if missing:
    print("  NOT FOUND in env.conf (add manually): " + ", ".join(missing))
PY
  # Re-read so the rest of install.sh sees the new values.
  load_config
}

# ── the interactive step ─────────────────────────────────────────────────────
# Returns 0 if config is ready to use, 1 if the user aborted.
run_preflight() {
  local assume_yes="$1"

  step "Hardware scan"
  scan_hardware
  print_hardware

  recommend_config
  REC_ALIAS="$(alias_for_repo "$REC_MODEL")"
  print_recommendation

  # Disk space for the download, only if it is not already there.
  if [[ "$HW_MODEL_PRESENT" == "no" ]]; then
    local need=$(( REC_WEIGHTS_GB + 3 ))
    if (( HW_FREE_GB < need )); then
      warn "Only ${HW_FREE_GB} GB free where the weights go; the download needs about ${need} GB."
      warn "Free up space and re-run, or set MODELS_DIR in env.conf to a bigger volume."
      return 1
    fi
    ok "Disk space is fine for a ~${need} GB download."
  fi

  # Hard blockers.
  if (( HW_RAM_GB < 24 )); then
    warn "${HW_RAM_GB} GB of unified memory is below what a 27B model needs."
    warn "It may load with a tiny context and q4 KV, but expect it to be slow and tight."
  fi
  if (( HW_MACOS_MAJOR < 14 )); then
    die "macOS ${HW_MACOS} is too old; MLX inference needs macOS 14 or newer."
  fi

  if current_config_fits; then
    ok "Your current env.conf fits this machine (${CUR_NEED_GB} GB plan, ${MEMORY_LIMIT_GB} GB cap)."
    if (( REC_NEED_GB < CUR_NEED_GB )) || [[ "$REC_MODEL" != "$MODEL" ]]; then
      log ""
      log "  The suggestions above would still be an improvement on this hardware."
    fi
  else
    warn "Your current env.conf needs about ${CUR_NEED_GB} GB but the cap is ${MEMORY_LIMIT_GB} GB."
    warn "The suggested values above fix that."
  fi

  log ""
  if (( assume_yes )); then
    info "--yes given: applying the suggested values."
    apply_config
    return 0
  fi

  # Only prompt when a real terminal is attached. Under a pipe or in CI the
  # open of /dev/tty fails loudly on macOS, so test for it first.
  if [[ ! -c /dev/tty ]] || [[ ! -t 0 && ! -t 1 ]]; then
    warn "No terminal attached; keeping your current env.conf."
    log "    Re-run ./install.sh in a terminal, or use --yes to accept these values."
    return 0
  fi

  printf '  Apply the suggested settings to env.conf? [Y/n] '
  local reply=""
  read -r reply < /dev/tty || reply=""
  case "$reply" in
    n|N|no|NO)
      # Declining the whole suggestion usually means "not that model", so offer
      # the list with a per-machine verdict rather than just giving up.
      print_model_menu
      local pick=""
      read -r pick < /dev/tty || pick=""
      pick="${pick//[!0-9]/}"
      if [[ -z "$pick" ]]; then
        info "Keeping your current env.conf."
        return 0
      fi
      local idx=1 chosen="" a
      for a in $MODEL_MENU_ALIASES; do
        if (( idx == pick )); then chosen="$a"; break; fi
        idx=$(( idx + 1 ))
      done
      if [[ -z "$chosen" ]]; then
        warn "No model number $pick; keeping your current env.conf."
        return 0
      fi
      local fit verdict size_gb
      fit="$(model_fit "$chosen")"; verdict="${fit#* }"
      REC_MODEL="$(model_repo_for "$chosen")"
      REC_ALIAS="$chosen"
      size_gb="$(model_size_gb "$REC_MODEL")"
      log ""
      if [[ "$verdict" == "will not fit" ]]; then
        warn "$chosen needs about ${fit%% *} GB and this Mac has ${HW_RAM_GB} GB."
        warn "It will be slow at best and may fail to load. Choosing it anyway."
      elif [[ "$verdict" == tight* ]]; then
        warn "$chosen needs about ${fit%% *} GB on a ${HW_RAM_GB} GB Mac - expect paging."
      fi
      # The disk check above ran against the SUGGESTED model's size, so a larger
      # pick has to be re-checked or the download dies half way through.
      if (( size_gb > 0 )) && (( HW_FREE_GB < size_gb + 3 )); then
        warn "$chosen downloads about ${size_gb} GB and only ${HW_FREE_GB} GB is free."
        warn "Free up space, or set MODELS_DIR in env.conf to a bigger volume."
        info "Keeping your current env.conf."
        return 0
      fi
      info "Using $chosen. The other suggested settings still apply."
      (( size_gb > 0 )) && REC_WEIGHTS_GB="$size_gb"
      apply_config
      ;;
    *)         apply_config ;;
  esac
  return 0
}
