#!/usr/bin/env bash
# Update this checkout to the latest version.
#
#   ./update.sh              pull, keep your env.conf, offer a restart
#   ./update.sh --check      report what is available, change nothing
#   ./update.sh --yes        do not ask anything
#   ./update.sh --no-restart do not offer to restart the server
#
# Why this is a script and not three lines of git in the README
# ------------------------------------------------------------
# `env.conf` is tracked, and editing it is the expected thing to do - it is the
# one file the project asks you to change. So a plain `git pull` very often
# stops with "Your local changes to the following files would be overwritten by
# merge: env.conf", which is a poor first experience for a project that sells
# itself as one command. This keeps your settings, takes the upstream code, and
# tells you exactly what happened.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

CHECK_ONLY=0
ASSUME_YES=0
OFFER_RESTART=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)      CHECK_ONLY=1; shift ;;
    --yes|-y)     ASSUME_YES=1; shift ;;
    --no-restart) OFFER_RESTART=0; shift ;;
    -h|--help)    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "Unknown argument: $1  (try --help)" >&2; exit 2 ;;
  esac
done

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
[[ -t 1 ]] || { BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""; }

say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s%s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '  %s%s%s\n' "$YELLOW" "$*" "$RESET"; }
die()  { printf '  %s%s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

[[ -d .git ]] || die "Not a git checkout: $REPO_DIR"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" == "HEAD" ]]; then
  die "Detached HEAD - check out a branch first: git checkout main"
fi

say ""
say "  ${BOLD}Updating $(basename "$REPO_DIR")${RESET}   ${DIM}branch $BRANCH${RESET}"
say ""

# ── what is available ────────────────────────────────────────────────────────
BEFORE="$(git rev-parse --short HEAD)"
BASE_REMOTE="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
if [[ -z "$BASE_REMOTE" ]]; then
  warn "No upstream branch configured; using origin/$BRANCH."
  BASE_REMOTE="origin/$BRANCH"
fi

if ! git fetch --quiet origin "$BRANCH" 2>/dev/null; then
  die "Could not reach the remote. Check your network, then try again."
fi
ok "fetched origin/$BRANCH"

BEHIND="$(git rev-list --count "HEAD..$BASE_REMOTE" 2>/dev/null || echo 0)"
AHEAD="$(git rev-list --count "$BASE_REMOTE..HEAD" 2>/dev/null || echo 0)"

if (( BEHIND == 0 )); then
  say ""
  ok "Already up to date at $BEFORE."
  (( AHEAD > 0 )) && warn "You have $AHEAD local commit(s) not on the remote."
  exit 0
fi

say ""
say "  $BEHIND new commit(s):"
git --no-pager log --oneline --no-decorate "HEAD..$BASE_REMOTE" | sed 's/^/      /' | head -25
(( BEHIND > 25 )) && say "      ... and $((BEHIND - 25)) more"

if (( CHECK_ONLY )); then
  say ""
  say "  Run ${BOLD}./update.sh${RESET} to apply."
  exit 0
fi

say ""

# ── local edits ──────────────────────────────────────────────────────────────
# Only these are expected to carry local edits. Everything else is ours to
# overwrite, so a dirty tree anywhere else is worth stopping for.
LOCAL_FILES="env.conf"
STASHED=0

DIRTY_TRACKED="$(git diff --name-only)"
if [[ -n "$DIRTY_TRACKED" ]]; then
  UNEXPECTED="$(printf '%s\n' "$DIRTY_TRACKED" | grep -vxF "$LOCAL_FILES" || true)"
  if [[ -n "$UNEXPECTED" ]]; then
    warn "These tracked files have local edits, and they are not ones this"
    warn "script knows how to merge:"
    printf '%s\n' "$UNEXPECTED" | sed 's/^/      /'
    say ""
    die "Commit or discard them first (git stash / git checkout -- <file>)."
  fi
  say "  ${DIM}Your env.conf edits will be kept.${RESET}"
  git stash push --quiet -m "update.sh: local env.conf" -- $LOCAL_FILES
  STASHED=1
fi

restore_env() {
  (( STASHED )) || return 0
  if git stash pop --quiet 2>/dev/null; then
    ok "Restored your env.conf."
    STASHED=0
    return 0
  fi

  # The pop conflicted, which means git left env.conf full of <<<<<<< markers.
  # That file is what load_config sources, so leaving it like this would break
  # every script in the project - a worse outcome than losing the merge. Put
  # the file back to the upstream version, which is known good, and keep the
  # stash so the user's settings are recoverable rather than gone.
  git reset --quiet -- env.conf 2>/dev/null || true
  git checkout --quiet HEAD -- env.conf 2>/dev/null || true
  warn "Upstream changed the same lines you did, so your env.conf could not be"
  warn "merged. Your version is safe in the stash; env.conf now holds upstream's."
  say  ""
  say  "      ${BOLD}See your version:${RESET}    git stash show -p stash@{0}"
  say  "      ${BOLD}Use your version:${RESET}    git checkout stash@{0} -- env.conf"
  say  "      ${BOLD}Drop it and move on:${RESET} git stash drop"
  say  "      ${DIM}Diffing the two is usually fastest: copy what you changed across.${RESET}"
  STASHED=0
  return 1
}

# ── pull ─────────────────────────────────────────────────────────────────────
if (( AHEAD > 0 )); then
  say ""
  warn "You have $AHEAD local commit(s). Rebasing your work onto the update."
  if ! git pull --rebase --quiet origin "$BRANCH"; then
    restore_env || true
    die "Rebase hit a conflict. Resolve it, then: git rebase --continue"
  fi
else
  if ! git pull --ff-only --quiet origin "$BRANCH"; then
    restore_env || true
    die "Fast-forward failed. Nothing was changed beyond the stash; resolve manually."
  fi
fi

AFTER="$(git rev-parse --short HEAD)"
ok "$BEFORE -> $AFTER"

restore_env || true

# ── what changed ─────────────────────────────────────────────────────────────
say ""
say "  ${BOLD}What changed${RESET}"
CHANGED="$(git diff --name-only "$BEFORE" "$AFTER" 2>/dev/null || true)"
printf '%s\n' "$CHANGED" | sed 's/^/      /' | head -20
(( $(printf '%s\n' "$CHANGED" | wc -l) > 20 )) && say "      ..."

# Dependencies or the launcher changing means a plain restart is not enough.
NEED_INSTALL=0
if printf '%s\n' "$CHANGED" | grep -qE '^(install\.sh|lib/|start\.sh|env\.conf)$'; then
  NEED_INSTALL=1
fi

say ""
if (( NEED_INSTALL )); then
  say "  ${DIM}lib/, start.sh or env.conf changed, so a restart is enough.${RESET}"
  say "  ${DIM}Run ./install.sh only if a requirement changed (the output above shows it).${RESET}"
else
  say "  ${DIM}Only scripts or docs changed.${RESET}"
fi

# ── restart ──────────────────────────────────────────────────────────────────
if (( OFFER_RESTART )); then
  port="${PORT:-8000}"
  if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    say ""
    if (( ASSUME_YES )); then REPLY=y; else
      printf '  Restart the server now to run the new code? [Y/n] '
      read -r REPLY || REPLY=n
    fi
    case "$REPLY" in
      ""|y|Y|yes|YES) exec "$REPO_DIR/restart.sh" ;;
      *) say "  ${DIM}Not restarting. ./restart.sh when you are ready.${RESET}" ;;
    esac
  else
    say ""
    say "  ${DIM}Server is not running. ./start.sh when you are ready.${RESET}"
  fi
fi

say ""
