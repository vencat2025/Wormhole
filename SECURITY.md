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

## Maintenance expectations

This is a personal project, maintained in spare time.

- Issues and pull requests are welcome, and I read them.
- I may be slow, and I may decline changes that widen the scope.
- There is no support commitment, response time, or release schedule.
- Per the Apache 2.0 licence, the software is provided "as is", without
  warranty of any kind.

If you depend on this for something that matters, fork it. That is what the
licence is for, and a fork you control is better than waiting on me.
