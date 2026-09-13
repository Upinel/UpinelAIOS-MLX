#!/usr/bin/env python3
"""Interactive chat against a running UpinelAIOS endpoint.

Started by ./chat.sh, which passes the connection details in the environment:

  CHAT_BASE          http://host:port/v1
  CHAT_KEY           bearer token (may be empty on a loopback-only server)
  CHAT_MODEL         the model id the server answers to
  CHAT_MAX_TOKENS    reply ceiling
  CHAT_THINKING      starting thinking mode: "on" or "off"
  CHAT_CAPS          comma-separated capabilities, see below
  CHAT_SYSTEM        starting system prompt

Capabilities, so one client serves both projects:

  stream     the server streams. Without it replies arrive whole.
  thinking   thinking can be toggled per request via chat_template_kwargs,
             rather than needing a restart. llama.cpp supports this.
  timings    the server reports prefill/decode rates; otherwise they are
             measured from wall time.

Nothing here is specific to a model family. Tool calling is deliberately not
implemented: this is a chat client, and a client that pretends to be an agent
without executing anything would be more confusing than useful.
"""

import json
import os
import queue
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

# ── terminal ─────────────────────────────────────────────────────────────────
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


BOLD = "1"
DIM = "2"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
CYAN = "36"


def width():
    return shutil.get_terminal_size((90, 24)).columns


def rule(char="─"):
    return c(DIM, char * min(width(), 72))


HELP = f"""
{c(BOLD, 'Commands')}
  {c(CYAN, '/help')}              this text
  {c(CYAN, '/reset')}             forget the conversation, keep the system prompt
  {c(CYAN, '/system <text>')}     replace the system prompt (resets history)
  {c(CYAN, '/thinking [on|off]')} show or set thinking for the next turns
  {c(CYAN, '/temp <0..1.5>')}     set sampling temperature for the next turns
  {c(CYAN, '/stats')}             tokens and rates for this session
  {c(CYAN, '/save <file>')}       write the transcript out as markdown
  {c(CYAN, '/clear')}             clear the screen, keep the conversation
  {c(CYAN, '/exit')}              quit

{c(BOLD, 'Keys')}
  Enter sends. Ctrl-C stops the current reply without quitting.
  Ctrl-D or /exit leaves. Up/Down walks your input history.
"""


class Client:
    def __init__(self):
        self.base = os.environ.get("CHAT_BASE", "http://127.0.0.1:8000/v1")
        self.key = os.environ.get("CHAT_KEY", "")
        self.model = os.environ.get("CHAT_MODEL", "Upinel-AIOS")
        self.max_tokens = int(os.environ.get("CHAT_MAX_TOKENS", "32768"))
        self.caps = set(filter(None, os.environ.get("CHAT_CAPS", "").split(",")))
        self.thinking = os.environ.get("CHAT_THINKING", "off") == "on"
        self.system = os.environ.get("CHAT_SYSTEM", "")
        # CHAT_TEMP is optional; empty means "use the default".
        try:
            self.temperature = float(os.environ.get("CHAT_TEMP") or 0.7)
        except ValueError:
            self.temperature = 0.7
        self.history = []
        self.session = {"turns": 0, "prompt": 0, "completion": 0,
                        "decode_s": [], "ttft_s": []}

    # ── request ──────────────────────────────────────────────────────────────
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def payload(self, stream):
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.extend(self.history)
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if stream:
            body["stream"] = True
            # Ask for usage on the final chunk where the server supports it.
            body["stream_options"] = {"include_usage": True}
        if "thinking" in self.caps:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        return body

    def send(self, text, on_delta, on_wait=None):
        """Stream one reply. Returns (content, reasoning, timings).

        `on_wait` is called roughly once a second until the first token lands,
        with (elapsed_seconds, server_progress). A server that is prefilling a
        long context, or throttled by memory pressure, can sit silent for a
        minute or more - and with no output at all that is indistinguishable
        from a hang, which is exactly how it reads to a user.
        """
        self.history.append({"role": "user", "content": text})
        stream = "stream" in self.caps
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(self.payload(stream)).encode(),
            headers=self._headers())

        content, reasoning, timings = [], [], {}
        t0 = time.time()
        ttft = None

        with urllib.request.urlopen(req, timeout=3600) as r:
            if not stream:
                d = json.loads(r.read())
                msg = (d.get("choices") or [{}])[0].get("message") or {}
                content = [msg.get("content") or ""]
                reasoning = [msg.get("reasoning_content") or ""]
                timings = d.get("timings") or {}
                usage = d.get("usage") or {}
                ttft = time.time() - t0
                for piece in content + reasoning:
                    if piece:
                        on_delta(piece, "content")
            else:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("timings"):
                        timings = chunk["timings"]
                    # MTPLX emits progress heartbeats while it works. They carry
                    # no content, but they are the difference between "busy" and
                    # "stuck", so pass them to the waiting indicator rather than
                    # discarding them.
                    if chunk.get("mtplx_progress") and on_wait:
                        on_wait(time.time() - t0, chunk["mtplx_progress"])
                    usage = chunk.get("usage")
                    if usage:
                        timings.setdefault("_usage", usage)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            if ttft is None:
                                ttft = time.time() - t0
                            reasoning.append(delta["reasoning_content"])
                            on_delta(delta["reasoning_content"], "reasoning")
                        if delta.get("content"):
                            if ttft is None:
                                ttft = time.time() - t0
                            content.append(delta["content"])
                            on_delta(delta["content"], "content")

        full = "".join(content).strip()
        self.history.append({"role": "assistant", "content": full})

        # Book-keeping for /stats.
        wall = time.time() - t0
        ct = (timings.get("_usage") or {}).get("completion_tokens") \
            or timings.get("predicted_n") or 0
        pt = (timings.get("_usage") or {}).get("prompt_tokens") \
            or timings.get("prompt_n") or 0
        self.session["turns"] += 1
        self.session["completion"] += ct
        self.session["prompt"] += pt
        if timings.get("predicted_per_second"):
            self.session["decode_s"].append(timings["predicted_per_second"])
        elif ct and wall > 0:
            self.session["decode_s"].append(ct / wall)
        if ttft:
            self.session["ttft_s"].append(ttft)
        return full, "".join(reasoning), timings

    def transcript(self):
        out = ["# UpinelAIOS transcript", "",
               f"- model: `{self.model}`",
               f"- endpoint: `{self.base}`",
               f"- thinking: `{'on' if self.thinking else 'off'}`",
               f"- temperature: `{self.temperature}`", ""]
        if self.system:
            out += ["## System", "", self.system, ""]
        for m in self.history:
            who = "You" if m["role"] == "user" else "Assistant"
            out += [f"## {who}", "", m["content"], ""]
        return "\n".join(out)


# ── rendering ────────────────────────────────────────────────────────────────
def render_reasoning(piece):
    sys.stdout.write(c(DIM, piece))
    sys.stdout.flush()


def render_content(piece):
    sys.stdout.write(piece)
    sys.stdout.flush()


def main():
    cl = Client()

    # readline gives arrow-key history and line editing; optional.
    try:
        import readline
        readline.set_history_length(500)
    except ImportError:
        readline = None

    print()
    print(c(BOLD, "  UpinelAIOS chat"))
    print(c(DIM, f"  {cl.base}   model {cl.model}"))
    caps = ", ".join(sorted(cl.caps)) or "none"
    print(c(DIM, f"  streaming: {'yes' if 'stream' in cl.caps else 'no'}"
                 f"   thinking: {'on' if cl.thinking else 'off'}"
                 f"   caps: {caps}"))
    print(c(DIM, "  /help for commands, /exit to leave"))
    print()

    in_reasoning = False
    while True:
        try:
            line = input(c(BOLD + ";" + GREEN, "you ▸ ") if _COLOR else "you > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = line.strip()
        if not text:
            continue

        # ── commands ─────────────────────────────────────────────────────────
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            cmd = cmd.lower()
            arg = arg.strip()

            if cmd in ("exit", "quit", "q"):
                break
            if cmd == "help":
                print(HELP)
            elif cmd == "reset":
                cl.history.clear()
                print(c(DIM, "  history cleared"))
            elif cmd == "system":
                if not arg:
                    print(c(DIM, f"  system prompt: {cl.system or '(none)'}"))
                else:
                    cl.system = arg
                    cl.history.clear()
                    print(c(DIM, "  system prompt set, history cleared"))
            elif cmd == "thinking":
                if arg in ("on", "off"):
                    if "thinking" in cl.caps:
                        cl.thinking = arg == "on"
                        print(c(DIM, f"  thinking {arg}"))
                    else:
                        print(c(YELLOW, "  this server fixes thinking at launch."))
                        print(c(DIM, "  change it with ./status.sh --thinking on|off,"
                                     " or THINKING in env.conf plus ./restart.sh"))
                else:
                    print(c(DIM, f"  thinking is {'on' if cl.thinking else 'off'}"))
            elif cmd == "temp":
                try:
                    v = float(arg)
                    if 0 <= v <= 2:
                        cl.temperature = v
                        print(c(DIM, f"  temperature {v}"))
                    else:
                        print(c(YELLOW, "  temperature must be between 0 and 2"))
                except ValueError:
                    print(c(DIM, f"  temperature is {cl.temperature}"))
            elif cmd == "stats":
                s = cl.session
                avg = sum(s["decode_s"]) / len(s["decode_s"]) if s["decode_s"] else 0
                ttft = sum(s["ttft_s"]) / len(s["ttft_s"]) if s["ttft_s"] else 0
                print(c(DIM, f"  turns {s['turns']}   prompt {s['prompt']} tok   "
                             f"completion {s['completion']} tok"))
                print(c(DIM, f"  decode {avg:.1f} t/s   first token {ttft:.2f} s"))
            elif cmd == "save":
                if not arg:
                    print(c(YELLOW, "  usage: /save <file>"))
                else:
                    try:
                        with open(arg, "w") as fh:
                            fh.write(cl.transcript())
                        print(c(DIM, f"  wrote {arg}"))
                    except OSError as e:
                        print(c(RED, f"  could not write {arg}: {e}"))
            elif cmd == "clear":
                print("\033[2J\033[H", end="")
            else:
                print(c(YELLOW, f"  unknown command /{cmd} - try /help"))
            continue

        # ── one turn ─────────────────────────────────────────────────────────
        sys.stdout.write(c(BOLD + ";" + BLUE, "ai  ▸ ") if _COLOR else "ai  > ")
        sys.stdout.flush()
        # Starts False: only a reasoning delta should open the thinking block,
        # otherwise the separator prints on every reply with nothing before it.
        in_reasoning = [False]
        shown_thinking_header = [False]

        def on_delta(piece, kind):
            if kind == "reasoning":
                if not shown_thinking_header[0]:
                    sys.stdout.write(c(DIM, "(thinking) "))
                    shown_thinking_header[0] = True
                in_reasoning[0] = True
                render_reasoning(piece)
            else:
                if in_reasoning[0]:
                    sys.stdout.write(c(DIM, "\n" + " " * 6))
                    in_reasoning[0] = False
                render_content(piece)

        # The request runs on a worker so this thread can keep drawing while
        # the server is silent. A cold model can take a minute to its first
        # token; printing nothing for that long reads as a hang.
        events = queue.Queue()
        outcome = {}

        def worker():
            try:
                outcome["r"] = cl.send(
                    text,
                    lambda p, k: events.put(("delta", (p, k))),
                    lambda el, pg: events.put(("wait", (el, pg))))
            except BaseException as e:                          # noqa: BLE001
                outcome["e"] = e
            finally:
                events.put(("done", None))

        t0 = time.time()
        started = False
        progress = None
        th = threading.Thread(target=worker, daemon=True)
        th.start()

        while True:
            try:
                kind, payload = events.get(timeout=0.5)
            except queue.Empty:
                if not started:
                    el = time.time() - t0
                    if progress:
                        note = (f"{progress.get('phase', 'working')}  "
                                f"{progress.get('completion_tokens', 0)} tok")
                    else:
                        note = "waiting for the server"
                    sys.stdout.write(
                        c(DIM, f"\r  ⏳ {el:4.1f}s   {note}   ") if _COLOR
                        else f"\r  ... {el:4.1f}s   {note}   ")
                    sys.stdout.flush()
                    if el > 25 and int(el * 2) % 20 == 0:
                        sys.stdout.write(c(YELLOW, "\n  (still nothing - "
                                                  "./status.sh shows whether the "
                                                  "server is throttled)"))
                        sys.stdout.flush()
                continue

            if kind == "wait":
                progress = payload[1]
                continue
            if kind == "done":
                break
            piece, dkind = payload
            if not started:
                # First real output: wipe the waiting line.
                sys.stdout.write("\r" + " " * max(0, width() - 1) + "\r")
                sys.stdout.flush()
                started = True
            on_delta(piece, dkind)

        err = outcome.get("e")
        if err is not None:
            sys.stdout.write("\r" + " " * max(0, width() - 1) + "\r")
            if isinstance(err, urllib.error.HTTPError):
                detail = err.read().decode("utf-8", "replace")[:300]
                print(c(RED, f"  HTTP {err.code}: {detail}"))
            elif isinstance(err, urllib.error.URLError):
                print(c(RED, f"  cannot reach {cl.base}: {err.reason}"))
                print(c(DIM, "  is the server running? try ./status.sh or ./start.sh"))
            elif isinstance(err, KeyboardInterrupt):
                print(c(DIM, "  (stopped)"))
            else:
                print(c(RED, f"  {type(err).__name__}: {err}"))
            if cl.history and cl.history[-1]["role"] == "user":
                cl.history.pop()
            continue

        content, reasoning, timings = outcome.get("r", ("", "", {}))

        wall = time.time() - t0
        if not content and not reasoning:
            print(c(YELLOW, "  (empty reply)"))

        # One compact line of telemetry, so speed is always visible.
        bits = []
        if timings.get("predicted_per_second"):
            bits.append(f"{timings['predicted_per_second']:.1f} t/s")
        elif wall > 0.05:
            ct = timings.get("_usage", {}).get("completion_tokens")
            if ct:
                bits.append(f"{ct / wall:.1f} t/s")
        if timings.get("prompt_per_second"):
            bits.append(f"prefill {timings['prompt_per_second']:.0f} t/s")
        bits.append(f"{wall:.1f}s")
        print()
        print(c(DIM, "  " + "   ".join(bits)))
        print()

    print(c(DIM, "  bye"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
