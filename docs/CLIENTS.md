# Connecting your agents

Point any OpenAI-compatible client at UpinelAIOS. It speaks Chat Completions,
an Anthropic-compatible `/v1/messages`, and streaming tool calls in both shapes.

```
Base URL:  http://<mac-lan-ip>:8000/v1
Model:     Upinel-AIOS             (or whatever SERVED_MODEL_NAME says)
API key:   the contents of run/api-key
```

`./status.sh` prints all three. From another machine on the same network,
verify first:

```bash
curl http://192.168.1.20:8000/v1/models -H "Authorization: Bearer $KEY"
```

> **Changed `SERVED_MODEL_NAME`?** Clients asking for the old id will start
> failing. Update the model field in each client, or set the name back.
> `./status.sh` always shows the id the server is currently advertising.

---

## Tool calling — read this before wiring up an agent

The endpoint calls tools correctly in every shape an agent uses. Verify it in
one command:

```bash
./bench/verify-tools.sh        # 13 checks, all formats
```

Output on a healthy server:

```
  2. OpenAI non-streaming tool call    [PASS] chose 'write_file'
  3. OpenAI streaming tool call        [PASS] deltas carry an index
  4. Multi-turn tool result            [PASS] model answers after the result
  5. Anthropic /v1/messages tool_use   [PASS] stop_reason='tool_use'
```

**The model cannot write files by itself.** It has no filesystem. It emits a
tool call; your harness executes it. If an agent reports that it cannot write
files, run the diagnostic above first — if it passes, the endpoint is not the
problem.

Two settings matter for agent use:

```conf
THINKING="off"          # recommended for tool loops
MAX_RESPONSE_TOKENS=32768
```

Thinking tokens are generated **before** the tool call and count against the
same `max_tokens` budget. With thinking on and a small client-side budget, the
tool call gets truncated mid-JSON and the client sees an unparseable call. Turn
thinking off, or give the agent 2048+ tokens of headroom. Thinking off is also
simply faster, and an agent that only needs to call a tool rarely benefits from
it.

## Output length

The server enforces `MAX_RESPONSE_TOKENS` (default 32768) as a *ceiling*. Client
software usually sends its own lower `max_tokens`. For agent work, note that
thinking tokens count against this budget: at `THINKING="high"` a model can burn
most of a small budget before writing any answer. Either give agents generous
`max_tokens`, or set `THINKING="off"` for tool-calling loops.

## Thinking

```conf
THINKING="off" | "low" | "medium" | "high"
```

Set in `env.conf`; it becomes the server default. A client can override per
request if it knows how to pass template kwargs:

```json
{
  "model": "Upinel-AIOS",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"}
}
```

The model's template accepts `low`, `medium`, and `xhigh` for
`reasoning_effort`; the server maps `THINKING="high"` to `xhigh`.

> **Note on this fine-tune:** the HauhauCS Aggressive fine-tune ships with
> thinking **disabled by default** in its chat template, which is deliberate —
> the converter notes it "avoids greedy-decoding degeneration loops." The bundle
> overrides that with `THINKING="low"` because that is what was asked for. If you
> ever see the model loop or refuse to stop, try `THINKING="off"` first: it is
> the single most effective fix, and it is also the fastest setting.

---

## Open WebUI

Settings → Connections → OpenAI API:

| Field | Value |
|---|---|
| Base URL | `http://<mac-ip>:8000/v1` |
| API Key | your key |

If Open WebUI runs in Docker on the same Mac, use `host.docker.internal:8000`
instead of `127.0.0.1`.

## Claude Code / Anthropic clients

```bash
export ANTHROPIC_BASE_URL="http://<mac-ip>:8000"
export ANTHROPIC_AUTH_TOKEN="$(cat run/api-key)"
```

The server exposes `/v1/messages` with streaming and tool calls.

## Cline / Continue / Roo (VS Code)

Choose the OpenAI-compatible provider and set:

```
Base URL:  http://<mac-ip>:8000/v1
Model:     Upinel-AIOS
API Key:   <your key>
```

## Aider

```bash
export OPENAI_API_BASE="http://<mac-ip>:8000/v1"
export OPENAI_API_KEY="$(cat run/api-key)"
aider --model openai/Upinel-AIOS
```

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.20:8000/v1", api_key="YOUR_KEY")

stream = client.chat.completions.create(
    model="Upinel-AIOS",
    messages=[{"role": "user", "content": "Write a haiku about memory bandwidth."}],
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

## Tool calling

The server advertises `"tool-prompt-mode": "hybrid"`, meaning it uses the model's
native tool-call format and falls back to a prompt-based convention. Tool calls
arrive as standard OpenAI `tool_calls` deltas. Test with a real agent before
trusting it: agent frameworks differ in how strictly they parse arguments.

---

## Sizing the client, not the server

Two client-side habits matter more than any server flag:

1. **Keep the conversation prefix stable.** The server caches prefixes. If your
   agent rewrites the system prompt, reorders tool definitions, or injects a
   timestamp at the top of every turn, it invalidates the cache on every request
   and re-prefills the whole context. This is the difference between a 2-second
   first token and a 3-minute one.
2. **Don't send the whole file tree every turn.** The context window is large
   enough that agents will happily fill it. Prefill cost is linear in prompt
   tokens; use retrieval instead of dumping.

## Health and metrics

```bash
curl -H "Authorization: Bearer $KEY" http://<mac-ip>:8000/health    | python3 -m json.tool
curl -H "Authorization: Bearer $KEY" http://<mac-ip>:8000/metrics
```

`/health` reports the loaded model, generation mode, MTP depth, KV quantization,
and the memory plan. Check it after a config change instead of guessing.
