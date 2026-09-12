# Troubleshooting

## First move, always

```bash
./status.sh                    # config, process, live /health, API key
tail -n 60 run/server.log      # what the server actually said
```

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
