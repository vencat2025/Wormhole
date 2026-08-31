# Security and maintenance

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's [private vulnerability reporting](https://github.com/vencat2025/Wormhole/security/advisories/new),
or email the maintainer.

Please include what you found, how to reproduce it, and what an attacker could
do with it. You will get an acknowledgement when I next pick up the project —
see the note on maintenance below.

## Running this safely

WormHole is a proxy that holds provider API keys and sees every prompt your
coding agent sends. A few things follow from that:

- **Bind it to localhost.** The default is `127.0.0.1`. Exposing it on a network
  interface hands anyone who can reach it your provider credentials.
- **Turn on auth if it is not on your own machine.** Set `ENABLE_AUTH=true` and
  supply `WORMHOLE_API_KEYS`. With auth on and no keys configured, the gateway
  refuses every request rather than defaulting open.
- **Keep `.env` out of git.** It is gitignored. Anything committed to a git
  repository should be considered public, even in a private repo, and even
  after deletion — history keeps it.
- **`wormhole.db` records prompts and completions.** That is how the routing
  feedback loop works, and it means the file contains whatever you asked your
  agent. It is gitignored. Treat it as you would your shell history.
- **Cross-origin access is off unless you ask for it.** Any page in your
  browser can send requests to `127.0.0.1`, so a gateway that answers them
  cross-origin lets any site you visit read your prompt history. Earlier
  versions did exactly that: `allow_origins=["*"]` with credentials allowed
  meant a request carrying `Origin: https://evil.example.com` got that origin
  reflected back and a readable body of prompts. Fixed by defaulting to no
  cross-origin access at all. `CORS_ORIGINS` can name specific origins if you
  really are fronting this from elsewhere; do not put `*` there.
- **The endpoints that expose prompts require auth.** `/api/logs`,
  `/api/routing/decisions`, `/api/dataset/export` and `/api/router/retrain` sit
  behind `verify_api_key`, so turning `ENABLE_AUTH` on actually protects the
  prompt history and not only the inference routes. The dashboard is served
  from the same origin and is unaffected.

## The benchmark runs model-written code on your machine

`scripts/evaluate_routing_quality.py` asks models to write Python, then
executes it to see whether it passes MBPP's assertions. Executing it is the
whole point -- that is what makes the result a measurement rather than an
opinion -- but it means **code you did not write, from a model, runs with your
user's permissions.**

The MBPP tasks themselves are benign, and normal model output for them is a
short pure function. The risk is not that MBPP is dangerous; it is that you are
running unreviewed generated code, and a compromised provider response or a
prompt-injected model could put anything in it. There is no sandbox in that
script.

If that matters for your environment, run it inside a container or a VM:

```bash
docker run --rm -it -v "$PWD":/w -w /w python:3.12 \
  sh -c "pip install -q -r requirements.txt && python scripts/evaluate_routing_quality.py --n 24"
```

Nothing else in this project executes model output. The gateway itself only
forwards tool calls back to your harness, which applies its own approval rules.

## If a key has ever been committed

Purging it from history is necessary and not sufficient. **Rotate the key.**

`git filter-repo` rewrites your local history, but a key that was ever pushed
should be treated as disclosed:

- GitHub keeps unreferenced objects reachable by commit SHA after a force-push.
  Anyone who saw the old SHA can still fetch the blob until GitHub garbage
  collects it, and that is not automatic — you have to ask GitHub Support to
  run it for the repository.
- Anyone who cloned or forked before the rewrite still has the original
  objects, and nothing you do to your copy reaches theirs.
- Automated scrapers watch public pushes for credential patterns. For a key
  pushed to a public repository, assume it was collected within minutes.

So the order that actually works is: **rotate the key at the provider first**,
then purge history, then ask GitHub Support to garbage collect, and only then
consider making the repository public. Rotating last leaves a live credential
exposed for the whole window.

Rotate at:
[Groq](https://console.groq.com/keys) ·
[OpenAI](https://platform.openai.com/api-keys) ·
[Google AI Studio](https://aistudio.google.com/apikey) ·
[Anthropic](https://console.anthropic.com/settings/keys)

## Maintenance expectations

This is a personal project, maintained in spare time.

- Issues and pull requests are welcome, and I read them.
- I may be slow, and I may decline changes that widen the scope.
- There is no support commitment, response time, or release schedule.
- Per the Apache 2.0 licence, the software is provided "as is", without
  warranty of any kind.

If you depend on this for something that matters, fork it. That is what the
licence is for, and a fork you control is better than waiting on me.
