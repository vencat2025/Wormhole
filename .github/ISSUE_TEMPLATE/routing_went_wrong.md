---
name: Routing went wrong
about: The router picked a model that was too weak, or too expensive, for a prompt
labels: routing
---

**The prompt** (redact anything private):

**What it routed to**, from the dashboard or `/api/logs`:

**What you expected**, and why:

**Your fleet** — the output of the startup log line beginning "Routing over",
plus `MIN_ROUTING_TIER` and `ROUTING_MODELS` if you set them:

---

This is the most useful kind of issue for this project. The classifier's known
weakness is short imperative prompts, and real examples of it getting them
wrong are what improve it.
