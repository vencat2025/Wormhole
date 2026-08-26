# Custom Harness & AI Tool Integration Guide

This guide explains how to connect your existing AI coding tools, agents, and custom harnesses (**Claude Code**, **OpenAI Codex**, **Cursor IDE**, **Aider CLI**, **Continue.dev**) directly to **WormHole** so your development LLM traffic is routed to cheaper models automatically.

Measured over 521 requests logged by the gateway during development
(`python scripts/measure_savings.py` reproduces this from your own `wormhole.db`):
124,563 input and 176,688 output tokens cost **$0.13** on the models actually
selected, against **$2.08** for the same measured token counts priced at GPT-4o
rates, a **93.6%** reduction. The token counts and model choices are real; the
GPT-4o baseline is a counterfactual computed from the rates in `config.py`, not
an observed bill. This sample is the author's own development traffic and is not
a claim about any other workload.

---

## ⚡ Overview

WormHole exposes a standard, 100% OpenAI-compatible REST proxy endpoint:
- **Base URL**: `http://127.0.0.1:8000/v1`
- **Default API Key**: `wh_live_demo123456789`
- **Routing Keyword**: `wormhole-auto` (or any target model ID)

---

## 1. 🤖 Claude Code CLI Integration (`claude`)

To route all **Claude Code CLI** tool requests through WormHole:

```bash
# Set base URL to point to local WormHole proxy
# Claude Code appends /v1/messages itself, so the base URL must NOT end in /v1
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="any-non-empty-value"  # only checked when ENABLE_AUTH=true

# Launch Claude Code as usual
claude
```

> **Status.** The gateway implements the Anthropic Messages API including tool
> translation (`input_schema` -> function parameters), `tool_use` content blocks
> and streaming `input_json_delta` events, verified against the live endpoint.
>
> Running Claude Code end to end needs headroom this project's default free
> tiers do not have. Claude Code sends roughly 40k tokens per turn (a ~7k system
> prompt plus ~30k of tool schemas across 29 tools), which exceeds Groq's
> on-demand ceiling of 8,000 tokens per minute even with every tool stripped.
> Requests that large are routed to a local Ollama model automatically, and
> `qwen2.5-coder:7b` is not reliable at native tool calling with that many tools
> — it tends to emit JSON as text instead.
>
> To use Claude Code through WormHole you need one of: a Groq Dev Tier account,
> a larger local tool-capable model, or a working OpenAI/Anthropic key. Codex CLI
> works on the free tier because it sends a far smaller tool payload.


> WormHole intercepts every code edit and terminal tool invocation, selectively enhancing prompts and routing requests to the cheapest capable model (`gemini-2.5-flash` or `gpt-4o-mini`), while automatically preserving `claude-3-5-sonnet` quality for complex architectural tasks.

---

## 2. 💻 Cursor IDE / VS Code Extensions

To connect **Cursor IDE** or VS Code AI plugins (like Continue or OpenRouter extensions):

1. Open **Cursor Settings** $\rightarrow$ **Models** / **OpenAI API Key**.
2. Override the **OpenAI Base URL**:
   ```text
   http://127.0.0.1:8000/v1
   ```
3. Enter your WormHole API key:
   ```text
   wh_live_demo123456789
   ```
4. Set model name to `wormhole-auto`.

---

## 3. 🐍 OpenAI Codex & Python / Node.js SDKs

If you have custom scripts or internal agent harnesses using the official OpenAI Python or Node SDK:

### Python SDK (`openai`):
```python
from openai import OpenAI

# Simply override base_url and api_key
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="wh_live_demo123456789"
)

response = client.chat.completions.create(
    model="wormhole-auto", # Automatic sub-2ms SLM routing
    messages=[
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
    ]
)

print(response.choices[0].message.content)
```

---

## 4. 🛠️ Aider CLI Integration (`aider`)

To route **Aider** terminal coding sessions through WormHole:

```bash
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="wh_live_demo123456789"

# Run Aider with WormHole routing
aider --model openai/wormhole-auto
```

---

## 5. 🧰 Continue.dev Integration (`config.json`)

Inside your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "WormHole Auto Router",
      "provider": "openai",
      "model": "wormhole-auto",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "wh_live_demo123456789"
    }
  ]
}
```

---

## 📊 Live Verification Dashboard

Whenever any harness (Claude Code, Cursor, Aider, custom scripts) executes requests through WormHole, open your dashboard at **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to watch real-time cost savings, prompt comparisons, and judge scores!
