#!/usr/bin/env python3
"""
Measure real decode throughput against a running OpenAI-compatible endpoint.

Client-side timing, so the numbers are what a client actually experiences:
  TTFT   = time to first streamed token (dominated by prefill)
  decode = (completion_tokens - 1) / (last_chunk_time - first_chunk_time)

Usage:
  bench.py --url http://127.0.0.1:8000 --model Upinel-AIOS \
           --contexts 512,8192,32768,131072 --max-tokens 128
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Sensors recorded a steady rise in ambient temperature across the valley. "
    "Engineers reviewed the telemetry and adjusted the cooling schedule again. "
)


def _prompt_for_tokens(n_tokens: int) -> str:
    """Build a prompt of roughly n_tokens tokens.

    The instruction deliberately asks for a long, unbroken continuation: a
    prompt that invites a one-word answer makes the model stop after a couple
    of tokens, and a decode rate measured over 5 tokens is noise.
    """
    header = (
        "Continue the following narrative in a single unbroken run of prose. "
        "Do not stop early, do not summarise, and do not ask questions. "
        "Keep writing until you are told to stop.\n\n"
    )
    # This prose tokenises at ~5.5 characters/token, not the usual 4, so a
    # 4-chars-per-token estimate undershoots the requested context by ~30%.
    body_chars = max(0, int(n_tokens * 5.5) - len(header))
    reps = max(1, body_chars // len(FILLER))
    return header + FILLER * reps


def run_case(url, model, context, max_tokens, api_key, timeout, quiet=False):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": _prompt_for_tokens(context)},
        ],
        "max_tokens": max_tokens,
        "temperature": 1.0,      # the model's own recommended sampling
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    target = url.rstrip("/") + "/v1/chat/completions"
    if not target.startswith("http"):
        target = "http://" + target
    # Tolerate a base URL given with or without the /v1 suffix.
    if not target.endswith("/chat/completions"):
        target = target.rstrip("/") + "/v1/chat/completions"

    req = urllib.request.Request(
        target,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )

    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content")
                             or delta.get("reasoning_content")
                             or delta.get("reasoning"))
                    if piece:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        return {"ok": False, "context": context,
                "error": f"HTTP {e.code}: {body}"}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "context": context, "error": f"{type(e).__name__}: {e}"}

    end = time.perf_counter()
    ttft = (first - t0) if first else None
    window = (last - first) if (first and last and last > first) else None
    completion_tokens = usage.get("completion_tokens") or chunks
    prompt_tokens = usage.get("prompt_tokens") or 0

    decode_tps = (completion_tokens - 1) / window if window and completion_tokens > 1 else None
    prefill_tps = prompt_tokens / ttft if (ttft and prompt_tokens) else None

    short = completion_tokens < 32
    if not quiet:
        print(f"  context~{context:>7}  "
              f"prompt={prompt_tokens:>7} tok  "
              f"ttft={('%.2fs' % ttft) if ttft else 'n/a':>8}  "
              f"prefill={('%.0f t/s' % prefill_tps) if prefill_tps else 'n/a':>10}  "
              f"gen={completion_tokens:>4} tok  "
              f"decode={('%.2f t/s' % decode_tps) if decode_tps else 'n/a':>10}  "
              f"wall={end - t0:.1f}s" + ("   <-- short sample, treat as noisy" if short else ""))

    return {
        "ok": True,
        "context": context,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_window_s": window,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "wall_s": end - t0,
        "short_sample": short,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Upinel-AIOS")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--contexts", default="512,8192,32768")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each context N times and report the median")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    key = args.api_key
    if not key and args.api_key_file:
        try:
            key = open(args.api_key_file).read().strip()
        except OSError:
            pass

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    if not args.json:
        print(f"\nBenchmarking {args.url}  model={args.model}  max_tokens={args.max_tokens}")
        print(f"{'':>2}{'context':>10}  {'prompt':>13}  {'ttft':>8}  {'prefill':>10}  "
              f"{'gen':>8}  {'decode':>10}")

    results = []
    for ctx in contexts:
        runs = []
        for i in range(args.repeats):
            r = run_case(args.url, args.model, ctx, args.max_tokens, key,
                         args.timeout, quiet=args.json)
            if not r["ok"]:
                if not args.json:
                    print(f"  context~{ctx:>7}  FAILED: {r['error']}")
                results.append(r)
                break
            runs.append(r)
        if not runs:
            continue
        if len(runs) > 1:
            med = {
                "ok": True, "context": ctx,
                "prompt_tokens": runs[0]["prompt_tokens"],
                "completion_tokens": runs[0]["completion_tokens"],
                "ttft_s": statistics.median(r["ttft_s"] for r in runs),
                "decode_tps": statistics.median(r["decode_tps"] for r in runs),
                "prefill_tps": statistics.median(
                    r["prefill_tps"] for r in runs if r["prefill_tps"]),
                "wall_s": statistics.median(r["wall_s"] for r in runs),
            }
            results.append(med)
        else:
            results.append(runs[0])

    good = [r for r in results if r.get("ok") and r.get("decode_tps")]
    if args.json:
        print(json.dumps({"results": results,
                          "best_decode_tps": max((r["decode_tps"] for r in good), default=None)},
                         indent=2))
        return 0

    print()
    if good:
        best = max(good, key=lambda r: r["decode_tps"])
        print(f"  Peak decode: {best['decode_tps']:.2f} tok/s at ~{best['context']} context")
    else:
        print("  No successful measurements.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
