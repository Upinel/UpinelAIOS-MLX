#!/usr/bin/env python3
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
"""
Verify that the endpoint's tool-calling works, in every shape an agent uses.

This is a client-side diagnostic. It answers the question "can the model call
tools?" separately from "is my agent harness wired up correctly?" — which are
different failures with the same symptom.

Usage:
  bench/verify-tools.py --url http://127.0.0.1:8000 --model Upinel-AIOS-MLX
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function",
     "function": {"name": "write_file",
                  "description": "Write text to a file, creating it if needed",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"},
                                                "content": {"type": "string"}},
                                 "required": ["path", "content"]}}},
    {"type": "function",
     "function": {"name": "read_file",
                  "description": "Read a file and return its contents",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"}},
                                 "required": ["path"]}}},
    {"type": "function",
     "function": {"name": "run_bash",
                  "description": "Run a shell command and return its output",
                  "parameters": {"type": "object",
                                 "properties": {"command": {"type": "string"}},
                                 "required": ["command"]}}},
    {"type": "function",
     "function": {"name": "list_dir",
                  "description": "List the entries in a directory",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"}},
                                 "required": ["path"]}}},
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""


def post(url, payload, headers, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_post(url, payload, headers, timeout):
    """Return (tool_name, raw_arguments, finish_reason) from an SSE stream."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream", **headers})
    name = None
    args = ""
    finish = None
    saw_index = False
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
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                for tc in delta.get("tool_calls") or []:
                    if "index" in tc:
                        saw_index = True
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        name = fn["name"]
                    if fn.get("arguments"):
                        args += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return name, args, finish, saw_index


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.transport = 0      # could not reach the server at all

    def check(self, label, ok, detail="", transport=False):
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {label}")
        if detail:
            print(f"         {DIM}{detail}{RESET}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            if transport:
                self.transport += 1
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Upinel-AIOS-MLX")
    ap.add_argument("--api-key")
    ap.add_argument("--api-key-file")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    key = args.api_key
    if not key and args.api_key_file:
        try:
            key = open(args.api_key_file).read().strip()
        except OSError:
            pass

    base = args.url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    report = Report()

    print(f"\nTool-calling diagnostic against {base}\n")
    print("  1. Server reachability")

    try:
        req = urllib.request.Request(base + "/models", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            models = json.loads(r.read())
        ids = [m.get("id") for m in models.get("data", [])]
        report.check("GET /v1/models", bool(ids), f"serving: {', '.join(str(i) for i in ids)}")
    except Exception as e:                                     # noqa: BLE001
        report.check("GET /v1/models", False, f"{type(e).__name__}: {e}")
        print("\nThe server is not answering. Start it with ./start.sh\n")
        return 1

    prompt = "Create a file called /tmp/demo.txt containing the word hi. Use the tool."

    # ── 2. non-streaming tool call ───────────────────────────────────────────
    print("\n  2. OpenAI non-streaming tool call")
    try:
        body = post(base + "/chat/completions",
                    {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                     "tools": TOOLS, "tool_choice": "auto", "max_tokens": 512},
                    headers, args.timeout)
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        report.check("response contains tool_calls", bool(calls),
                     f"content={msg.get('content')!r}" if not calls else "")
        if calls:
            fn = calls[0].get("function") or {}
            report.check("picked the right tool", fn.get("name") == "write_file",
                         f"chose {fn.get('name')!r}")
            try:
                parsed = json.loads(fn.get("arguments") or "{}")
                ok = parsed.get("path") and parsed.get("content")
                report.check("arguments are valid JSON", bool(ok),
                             json.dumps(parsed)[:160])
            except json.JSONDecodeError as e:
                report.check("arguments are valid JSON", False, str(e))
            report.check("finish_reason is 'tool_calls'",
                         (body.get("choices") or [{}])[0].get("finish_reason") == "tool_calls",
                         f"got {(body.get('choices') or [{}])[0].get('finish_reason')!r}")
    except Exception as e:                                     # noqa: BLE001
        report.check("non-streaming tool call", False,
                     f"{type(e).__name__}: {e}",
                     transport=isinstance(e, (urllib.error.URLError, OSError,
                                              TimeoutError)))

    # ── 3. streaming tool call (what agents actually use) ────────────────────
    print("\n  3. OpenAI streaming tool call  (the path real agents use)")
    try:
        name, raw, finish, saw_index = stream_post(
            base + "/chat/completions",
            {"model": args.model, "messages": [{"role": "user", "content": prompt}],
             "tools": TOOLS, "stream": True, "max_tokens": 512},
            headers, args.timeout)
        report.check("stream emitted tool_calls deltas", name is not None,
                     f"tool={name!r}")
        if name:
            report.check("streamed name matches", name == "write_file", f"got {name!r}")
        try:
            json.loads(raw or "{}")
            report.check("assembled arguments are valid JSON", bool(raw),
                         (raw or "")[:160])
        except json.JSONDecodeError as e:
            report.check("assembled arguments are valid JSON", False,
                         f"{e} -- raw: {raw[:120]!r}")
        report.check("finish_reason is 'tool_calls'", finish == "tool_calls",
                     f"got {finish!r}")
        report.check("deltas carry an index (required by OpenAI clients)", saw_index)
    except Exception as e:                                     # noqa: BLE001
        report.check("streaming tool call", False, f"{type(e).__name__}: {e}",
                     transport=isinstance(e, (urllib.error.URLError, OSError,
                                              TimeoutError)))

    # ── 4. multi-turn: feed the result back ──────────────────────────────────
    print("\n  4. Multi-turn tool result round-trip")
    try:
        body = post(base + "/chat/completions",
                    {"model": args.model, "max_tokens": 256, "tools": TOOLS,
                     "messages": [
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": None,
                          "tool_calls": [{"id": "call_1", "type": "function",
                                          "function": {"name": "write_file",
                                                       "arguments": json.dumps(
                                                           {"path": "/tmp/demo.txt",
                                                            "content": "hi"})}}]},
                         {"role": "tool", "tool_call_id": "call_1",
                          "content": "File written successfully."},
                     ]},
                    headers, args.timeout)
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("content") or "")
        report.check("model accepts a tool result and answers",
                     bool(text.strip()), text.strip()[:160])
    except Exception as e:                                     # noqa: BLE001
        report.check("tool result round-trip", False, f"{type(e).__name__}: {e}",
                     transport=isinstance(e, (urllib.error.URLError, OSError,
                                              TimeoutError)))

    # ── 5. Anthropic shape, used by Claude Code ──────────────────────────────
    print("\n  5. Anthropic /v1/messages tool_use  (Claude Code path)")
    try:
        a_headers = dict(headers)
        if key:
            a_headers["x-api-key"] = key
        a_headers["anthropic-version"] = "2023-06-01"
        body = post(base + "/messages",
                    {"model": args.model, "max_tokens": 512,
                     "messages": [{"role": "user", "content": prompt}],
                     "tools": [{"name": "write_file",
                                "description": "Write text to a file",
                                "input_schema": {"type": "object",
                                                 "properties": {"path": {"type": "string"},
                                                                "content": {"type": "string"}},
                                                 "required": ["path", "content"]}}]},
                    a_headers, args.timeout)
        blocks = body.get("content") or []
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        report.check("response contains a tool_use block", bool(uses),
                     f"stop_reason={body.get('stop_reason')!r}")
        if uses:
            report.check("tool_use input is populated",
                         bool(uses[0].get("input")), json.dumps(uses[0].get("input"))[:160])
    except urllib.error.HTTPError as e:
        report.check("Anthropic /v1/messages", False, f"HTTP {e.code}")
    except Exception as e:                                     # noqa: BLE001
        report.check("Anthropic /v1/messages", False, f"{type(e).__name__}: {e}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print()
    if report.failed == 0:
        print(f"  {GREEN}All {report.passed} checks passed.{RESET}")
        print("\n  The endpoint calls tools correctly in every format an agent uses.")
        print("  If your agent still cannot write files, the problem is on the")
        print("  client side. Check, in order:")
        print("    1. Did you give the agent a file-write tool in its config?")
        print("    2. Is the agent pointed at this base URL with the right API key?")
        print("    3. Does the agent accept tool calls from a local/OpenAI-compatible")
        print("       provider, or is it a hosted-only tool that ignores them?")
        print("    4. Ask the agent to show the raw tool call it received.")
        print()
        return 0

    print(f"  {RED}{report.failed} check(s) failed{RESET} ({report.passed} passed).")

    if report.transport >= report.failed and report.transport > 0:
        # Nothing structural: the requests never landed.
        print(f"\n  {YELLOW}These look like connection failures, not model failures.{RESET}")
        print("  The endpoint answered /v1/models but then stopped responding,")
        print("  which usually means it is still starting up or was restarting")
        print("  mid-run. A freshly started server spends 30-90s loading weights")
        print("  and is not ready until /health returns ok.")
        print("\n    ./status.sh --once     # check the server state")
        print("    tail -f run/server.log # watch it come up")
        print("\n  Re-run this diagnostic once it reports 'serving'.\n")
        return 1

    print(f"\n  {YELLOW}If the failing checks are about JSON or finish_reason:{RESET}")
    print("    the reply is probably being truncated. Raise max_tokens, or set")
    print("    THINKING=\"off\" in env.conf — thinking tokens count against the")
    print("    same budget and can eat it before the tool call is emitted.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
