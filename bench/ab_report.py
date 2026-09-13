#!/usr/bin/env python3
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
"""Render the side-by-side table produced by bench/ab.sh."""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
key = sys.argv[2]
vals = sys.argv[3:]

data = {}
for v in vals:
    f = root / f"{key}={v}.json"
    if not f.exists():
        continue
    try:
        d = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    rows = {}
    for r in d.get("results", []):
        if r.get("ok") and r.get("decode_tps"):
            rows[r["prompt_tokens"]] = r
    if rows:
        data[v] = rows

if not data:
    print("  No usable results.")
    raise SystemExit(1)

prompts = sorted({p for rows in data.values() for p in rows})

width = max(14, max(len(v) for v in data) + len(key) + 2)
print()
print(f"  {'prompt tok':>10}  " + "  ".join(f"{key}={v:>{width - len(key) - 1}}" for v in data))
print(f"  {'-' * 10}  " + "  ".join("-" * width for _ in data))
for p in prompts:
    cells = []
    for v in data:
        r = data[v].get(p)
        cells.append(f"{r['decode_tps']:.2f} t/s" if r else "n/a")
    print(f"  {p:>10}  " + "  ".join(f"{c:>{width}}" for c in cells))

print()
print(f"  {'prompt tok':>10}  " + "  ".join(f"{key}={v:>{width - len(key) - 1}}" for v in data))
print(f"  {'-' * 10}  " + "  ".join("-" * width for _ in data))
for p in prompts:
    cells = []
    for v in data:
        r = data[v].get(p)
        cells.append(f"{r['prefill_tps']:.0f} t/s" if r and r.get("prefill_tps") else "n/a")
    print(f"  {p:>10}  " + "  ".join(f"{c:>{width}}" for c in cells))
print("  (rows above: decode / prefill)")

print()
for v in data:
    best = max(r["decode_tps"] for r in data[v].values())
    print(f"  {key}={v:<14} best decode {best:.2f} tok/s")
