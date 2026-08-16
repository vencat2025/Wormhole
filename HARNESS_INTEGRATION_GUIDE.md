# Custom Harness & AI Tool Integration Guide

This guide explains how to connect your existing AI coding tools, agents, and custom harnesses (**Claude Code**, **OpenAI Codex**, **Cursor IDE**, **Aider CLI**, **Continue.dev**) directly to **WormHole** so 100% of your development LLM traffic is automatically optimized, saving **90%+ in API costs**.

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
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000/v1"
export ANTHROPIC_API_KEY="wh_live_demo123456789"

# Launch Claude Code as usual
claude
```

> WormHole intercepts every code edit and terminal tool invocation, selectively enhancing prompts and routing requests to the cheapest capable model (`gemini-1.5-flash` or `gpt-4o-mini`), while automatically preserving `claude-3-5-sonnet` quality for complex architectural tasks.

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
