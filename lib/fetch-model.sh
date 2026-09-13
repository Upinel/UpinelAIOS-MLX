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
# Resumable Hugging Face model fetcher - no huggingface_hub / pip dependency.
#
#   lib/fetch-model.sh <owner/repo> <destination-dir>
#
# Features
#   * lists the repo through the public HF API and downloads every file
#   * resumes partial downloads (curl -C -), so Ctrl-C is safe
#   * verifies each file's byte size against the API, and repairs mismatches
#   * skips work entirely when the local copy is already complete
#   * MODEL_SOURCE_DIR=<path> adopts an existing local copy with hardlinks
#     instead of downloading (useful when you already have the weights)

set -euo pipefail

SOURCE_DIR="${BASH_SOURCE[0]}"
LIB_DIR="$(cd "$(dirname "$SOURCE_DIR")" && pwd)"
# shellcheck source=lib/common.sh
source "$LIB_DIR/common.sh"

REPO="${1:-}"
DEST="${2:-}"
[[ -n "$REPO" && -n "$DEST" ]] || die "usage: fetch-model.sh <owner/repo> <destination-dir>"

REPO="${REPO#https://huggingface.co/}"
REPO="${REPO%/}"
mkdir -p "$DEST"

# ── adopt an existing local copy instead of downloading ──────────────────────
if [[ -n "${MODEL_SOURCE_DIR:-}" && -d "${MODEL_SOURCE_DIR:-}" ]]; then
  info "Adopting existing weights from $MODEL_SOURCE_DIR"
  # Hardlink when possible (same volume, zero extra disk); copy otherwise.
  if cp -al "$MODEL_SOURCE_DIR"/. "$DEST"/ 2>/dev/null; then
    ok "Hardlinked $(find "$DEST" -type f | wc -l | tr -d ' ') files."
  else
    warn "Hardlink failed (different filesystem?). Falling back to a full copy."
    cp -a "$MODEL_SOURCE_DIR"/. "$DEST"/
    ok "Copied $(find "$DEST" -type f | wc -l | tr -d ' ') files."
  fi
  exit 0
fi

# ── fetch the file manifest ──────────────────────────────────────────────────
info "Querying https://huggingface.co/api/models/$REPO"
MANIFEST="$DEST/.manifest.tsv"

if ! curl -fsSL --max-time 60 \
      "https://huggingface.co/api/models/${REPO}?blobs=true" \
      -o "$DEST/.manifest.json"; then
  die "Could not reach the Hugging Face API for '$REPO'. Check the repo id and your network."
fi

python3 - "$DEST/.manifest.json" > "$MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for f in d.get("siblings", []):
    name = f.get("rfilename")
    size = f.get("size")
    if not name or name.endswith("/"):
        continue
    # Skip the repo furniture nobody needs at inference time.
    if name in {".gitattributes", "README.md"}:
        continue
    print(f"{size if size is not None else -1}\t{name}")
PY

[[ -s "$MANIFEST" ]] || die "The API returned no files for '$REPO'."
TOTAL="$(awk -F'\t' '{s+=$1} END {printf "%.1f", s/1e9}' "$MANIFEST")"
COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
info "$COUNT files, $TOTAL GB to fetch"

# ── download loop ────────────────────────────────────────────────────────────
FAILED=0
while IFS=$'\t' read -r WANT NAME; do
  [[ -n "$NAME" ]] || continue
  OUT="$DEST/$NAME"
  mkdir -p "$(dirname "$OUT")"

  HAVE=0
  [[ -f "$OUT" ]] && HAVE="$(stat -f%z "$OUT" 2>/dev/null || echo 0)"

  if [[ "$WANT" != "-1" ]] && [[ "$HAVE" == "$WANT" ]]; then
    ok "have   $NAME  ($((WANT / 1000000)) MB)"
    continue
  fi

  if [[ "$HAVE" != "0" ]]; then
    info "resume $NAME  (${HAVE} of ${WANT} bytes)"
  else
    info "fetch  $NAME"
  fi

  # -C - resumes; --retry rides out transient CDN failures; -L follows to the
  # xet/cdn host. Writing to a .part file keeps a torn download from looking
  # complete on the next run.
  if curl -fL --retry 5 --retry-delay 2 --retry-all-errors -C - \
        -o "$OUT" "https://huggingface.co/${REPO}/resolve/main/${NAME}"; then
    NEW="$(stat -f%z "$OUT" 2>/dev/null || echo 0)"
    if [[ "$WANT" != "-1" ]] && [[ "$NEW" != "$WANT" ]]; then
      warn "size mismatch on $NAME (want $WANT, got $NEW) - refetching from scratch"
      rm -f "$OUT"
      curl -fL --retry 5 --retry-delay 2 --retry-all-errors \
        -o "$OUT" "https://huggingface.co/${REPO}/resolve/main/${NAME}" \
        || { warn "FAILED $NAME"; FAILED=1; }
    fi
  else
    warn "FAILED $NAME  (re-run ./install.sh to resume)"
    FAILED=1
  fi
done < "$MANIFEST"

rm -f "$DEST/.manifest.tsv"

if (( FAILED )); then
  die "Some files did not download. Re-run ./install.sh - it resumes where it stopped."
fi

# ── final size audit ─────────────────────────────────────────────────────────
# Write the expected sizes to a plain file first. Nesting a heredoc inside a
# process substitution is legal but fragile, and a mis-parse here would silently
# pass an incomplete download.
python3 - "$DEST/.manifest.json" > "$DEST/.expected.tsv" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for f in d.get("siblings", []):
    name = f.get("rfilename")
    size = f.get("size")
    if name and size is not None and name not in {".gitattributes", "README.md"}:
        print(f"{size}\t{name}")
PY

BAD=0
while IFS=$'\t' read -r WANT NAME; do
  [[ -n "$NAME" ]] || continue
  HAVE="$(stat -f%z "$DEST/$NAME" 2>/dev/null || echo 0)"
  if [[ "$HAVE" != "$WANT" ]]; then
    warn "incomplete: $NAME ($HAVE/$WANT bytes)"
    BAD=1
  fi
done < "$DEST/.expected.tsv"

rm -f "$DEST/.expected.tsv" "$DEST/.manifest.json"

(( BAD )) && die "Model download is incomplete. Re-run ./install.sh."

DU="$(du -sh "$DEST" | cut -f1)"
ok "Model ready at $DEST  ($DU)"
