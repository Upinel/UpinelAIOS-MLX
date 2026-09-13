# Troubleshooting

## First move, always

```bash
./status.sh                    # live dashboard: CPU, GPU, memory, activity
./status.sh --once             # same thing as a plain summary, for logs
./restart.sh --print           # what an env.conf change would do
tail -n 60 run/server.log      # what the server actually said
```

Two panels are the fastest way to spot trouble:

- **ENDPOINT PROCESS** — the inference process's own allocation. Watch `total`
  against `host free`: when `total` climbs past ~30 GB on a 64 GB Mac and
  `host free` drops under 3 GB, you are in the swap zone described below.
- **TOKEN RATE** — a chart of live decode rate. A collapsing throughput shows
  up immediately as a descending line, and the min/avg/max figures tell you
  whether it is a blip or a trend.

The **CLIENTS** panel shows who is connected, by peer address and process name.
If something you do not recognise is using your endpoint, that is where you will
see it. On a short terminal the panel is dropped and its contents move into the
footer line — resize the window taller to get the full table back.

`run/server.log` is the whole story. Every failure below was diagnosed from it.

---

## "internal server error" on a long prompt

Symptom: a request with a very large prompt returns HTTP 500 after several
minutes with the message

```json
{"error":{"message":"internal server error; see the MTPLX server log","type":"server_error"}}
```

Find the request id in the response, then grep the log for it. If you see:

```
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

this is a **Metal command-buffer allocation failure, not a system RAM
shortage.** macOS caps how much a single Metal command buffer may allocate, and
a very long prefill can exceed it. Free system memory will not help; the
allocator ceiling is the constraint.

Levers, in order:

1. **Lower `PREFILL_CHUNK_TOKENS`** (try `2048`) so prefill runs in smaller
   Metal allocations instead of one large one. This is the direct fix.
2. **Raise `MEMORY_LIMIT_GB`.** Counterintuitive, but if MLX's own ceiling sits
   below what the prefilled batch needs, raising it gives Metal room. Do not
   push it past ~85% of physical RAM.
3. **Raise the macOS wired ceiling** with `WIRED_LIMIT_GB` — and read the warning
   in `env.conf` first. A hard-killed process can leak wired pages until reboot.

Observed on the reference machine (M5 Pro, 64 GB, desktop session running):
prompts up to ~93k tokens prefill successfully, a single ~131k-token prefill
fails with this error. **Incremental context growth works** — an agent that
builds a 128k conversation turn by turn is reusing cached prefixes, not
prefilling 131k in one go.

## Swap storm / the machine becomes unresponsive

`vm_stat` shows `Pages wired down` approaching physical RAM and swap fills.
MTPLX's session bank is usually the cause: it auto-sizes to *half the
post-model RAM surplus* (16.6 GB on a 64 GB Mac with a 48 GB cap).

```conf
SESSION_BANK_GB=8      # was auto; 8 is a good balance
MLX_CACHE_LIMIT_GB=4   # tighten further if still swapping
MEMORY_LIMIT_GB=44     # or lower the overall ceiling
```

Restart with `./stop.sh && ./start.sh`. **Always stop gracefully** — `SIGTERM`,
not `kill -9`. A process holding a large wired MLX allocation that is
hard-killed may leak those pages at the kernel level until the machine reboots.
`./stop.sh --force` exists, but it is a last resort.

## The dashboard shows "display error"

The live dashboard reports a rendering fault instead of exiting, and writes the
traceback to `run/dashboard.err`.

This is deliberate. A fault in one panel used to kill the process and dump a
traceback over the frame, which is a bad way to learn about a bug in a tool you
are using to watch a server. Now the frame degrades: the header, the fault
count, and the system numbers that were already collected stay on screen, and
the rest is logged.

The usual cause is telemetry that MTPLX has not measured yet. It publishes
`null` for those fields — most often `decode_tok_s` while a long prompt is still
prefilling — and a `null` reaching a format specifier raises `TypeError`. The
known cases are fixed; anything new lands in `run/dashboard.err`.

```bash
cat run/dashboard.err        # the traceback, with the frame number
```

Include that file if you report it. Restarting `./status.sh` clears the fault;
no other part of UpinelAIOS-MLX is affected, because the dashboard is read-only and
never talks to the model directly.

## The server will not start

```bash
./start.sh --print        # exactly what it would run
tail -n 40 run/server.log
```

Common causes:

| Symptom in the log | Cause |
|---|---|
| `Port ... already in use` | Another MTPLX daemon (maybe the menu-bar app). `./stop.sh` finds and stops it. |
| `Model not found at ...` | Model not downloaded. Run `./install.sh`. |
| `missing or invalid API key` on `/health` | You are bound to `0.0.0.0`, which always requires the key. `./status.sh` prints it. |
| A `mtplx` command not found | `brew install youssofal/mtplx/mtplx`, or re-run `./install.sh`. |

## "The model cannot create or write files"

**Start here:** run the diagnostic.

```bash
./bench/verify-tools.sh
```

It checks all 13 things an agent needs — non-streaming tool calls, streaming
tool-call deltas with indexes, argument JSON assembly, the multi-turn tool
result round-trip, and the Anthropic `/v1/messages` `tool_use` shape. If it
passes, the endpoint is fine and the problem is on the client side.

**Understand what the model can and cannot do.** A language model has no
filesystem. It cannot write a file; it can only *ask* the client to write one
by emitting a tool call. Something else — your agent harness — has to receive
that call and perform it. If you ask the model directly in a chat box to "create
a file", the correct behaviour is for it to describe the file, not to create it.
That is not a bug.

So when file writing does not work, the failure is almost always one of these:

| Check | How to confirm |
|---|---|
| The agent was never given a write tool | Look at the tools list in the agent's request. No `write_file`-style tool means nothing to call. |
| The agent is pointed at the wrong endpoint | It should be the base URL from `./status.sh`, ending in `/v1`, with the API key. |
| The agent only supports hosted providers | Some tools ignore tool calls from a local OpenAI-compatible endpoint. Check its provider docs. |
| The reply is being truncated | If the tool call is cut off, arguments arrive as invalid JSON. See below. |
| A middle layer strips tool calls | Proxies, gateways, and "OpenAI-compatible" shims sometimes drop `tool_calls`. |

### The truncation trap, which is the one that bites

Tool calls are emitted as tokens like any other output, and they are emitted
**after** any thinking block. With `THINKING="low"` a model can spend several
hundred tokens reasoning before it starts writing the call, and if the client's
`max_tokens` is small the call is cut off mid-JSON. The client then sees an
unparseable tool call and reports that the model "failed".

```conf
THINKING="off"
```

is the fix, and it is the right setting for a tool-calling loop anyway: it is
faster, and it removes the truncation risk. If you want to keep thinking on,
give the agent a generous `max_tokens` (2048+) instead.

The diagnostic prints this hint automatically when it sees a malformed
`arguments` string or a `finish_reason` other than `tool_calls`.

### Verifying by hand

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat run/api-key)" \
  -d '{
    "model": "Upinel-AIOS-MLX",
    "messages": [{"role":"user","content":"Create /tmp/x.txt containing hi. Use the tool."}],
    "tools": [{"type":"function","function":{
      "name":"write_file",
      "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
    "max_tokens": 512
  }'
```

A working endpoint answers with `tool_calls` and `"finish_reason":"tool_calls"`.

## Output loops, or the model will not stop

This is a known pathology of reasoning models on very repetitive input, and this
fine-tune's own converter disables thinking by default specifically to avoid it.

```conf
THINKING="off"
```

That is the fastest fix and the fastest setting. If you need thinking, MTPLX
also ships a loop guard (`MTPLX_LOOP_GUARD=1`) and supports
`presence_penalty` — but note that any non-zero penalty breaks MTPLX's
exactness guarantee, so treat it as a last resort rather than a default.

## Everything is slower than the numbers in the README

Check, in order:

1. **Are you actually using MTP?** `./status.sh` shows the depth. If it reads 0,
   you are at ~15 tok/s by design. `./bench/bench.sh --tune` to re-measure.
2. **Is macOS under memory pressure?** `vm_stat | grep -i wired` and
   `sysctl vm.nswusage`. See the swap section above.
3. **How long is your context?** Decode falls from ~52 tok/s at 400 tokens to
   ~22 tok/s at 32k. This is expected, not a misconfiguration.
4. **Is it throttling?** A laptop under a long agent run will thermally throttle.
   Try `FAN_MODE="smart"`.
5. **Are you re-prefilling every turn?** If your client rewrites the system
   prompt or injects a timestamp each request, the prefix cache misses and you
   pay full prefill cost every time. This is the most common self-inflicted
   slowdown in agent setups.

## Reporting a problem upstream

`mtplx doctor --bundle` produces a redacted support bundle. If you report the
Metal OOM case, include the `[METAL] Command buffer execution failed` line and
your `PREFILL_CHUNK_TOKENS` / `MEMORY_LIMIT_GB` values — those are the two
inputs that changed the outcome here.
