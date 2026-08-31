# Contributing

Thanks for looking. This is a side project maintained in spare time, so please
read the [maintenance expectations](SECURITY.md#maintenance-expectations)
before opening anything — they are deliberately modest.

## Pull requests are limited to collaborators right now

Please read this before writing a patch, so you do not find out at the end.

Pull request creation is restricted while the routing work settles. Cloning,
forking and running it are all open; only sending a PR back is not.

**Please open an issue instead, and the most valuable one is a routing case we
got wrong.** The classifier's known weakness is short imperative prompts, and a
real example of it sending the wrong tier is worth more to this project than
almost any patch. There is a
[template](.github/ISSUE_TEMPLATE/routing_went_wrong.md) for exactly that.

If you have a fix you would rather send as code, say so in the issue and we
will work out how to get it in.

## When PRs open up, before you open one

Run the tests:

```bash
pytest tests/ -q
```

Most of them need no credentials. The few that need a model they can actually
reach skip themselves when `.env` has no provider key, so a clean clone gives
16 passed / 3 skipped, and CI runs that way too. Add one key and you get the
full 18.

CI deliberately has no secrets. A pull request from a fork can run workflow
code, so any key available to CI is a key handed to anyone who opens a PR.

If you changed routing, run the quality benchmark too, and **paste the output
in the PR**. It executes MBPP's own assertions, so it is a measurement rather
than an opinion:

```bash
python scripts/evaluate_routing_quality.py --n 24 --baseline gpt-4o
```

## What gets merged quickly

- A bug with a reproduction
- A provider added to `CANDIDATE_MODELS` with **verified** pricing
  (`pricing_verified=True` means you checked the published price list, not that
  you estimated it)
- Documentation that corrects something wrong

## What will get questions

**Numbers without a run behind them.** Every figure in the README traces to a
logged request or an executed benchmark. If a change alters a published number,
the PR needs the run that produced the new one.

Watch the denominator in particular: when calls fail, that arm attempted fewer
questions, and a pass rate over the smaller number flatters it. Compare tasks
solved out of tasks given. This repository has made that exact mistake and
reported an 8-point improvement that was not there.

## Style

Match the surrounding code. Comments explain *why*, especially where the
behaviour looks wrong until you know the constraint — most comments here exist
because something failed in a non-obvious way.
