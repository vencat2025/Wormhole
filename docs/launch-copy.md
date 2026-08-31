# Launch copy

Two pieces, both built around the same observation: choosing a model well means
knowing how hard a prompt is, and that is not knowable before it is answered.

Every number below came from the gateway's own log during a real Codex session.
Regenerate them with `scripts/evaluate_routing_quality.py` or by reading
`/api/logs`.

---

## LinkedIn post

Arul and I built a small gateway that picks the right model for each prompt, so
nobody has to guess.

Here is the problem we kept running into ourselves.

AI spend adds up quickly, and the sensible response is to use the expensive
model only where it earns its price. We have both given that advice. We have
both tried to follow it.

It is genuinely hard to follow, and that is not a discipline problem. You
cannot tell how hard a prompt is until it has been answered. Guess low and you
might get something subtly wrong and lose the turn. Guess high and it works.
So we both did what everyone does: pinned the strongest model and stopped
thinking about it.

That is a reasonable decision. It is just an expensive one, and it is being
asked at the moment when the least is known.

So we moved the decision. Your Codex or Claude Code stays pinned to whatever it
is pinned to, and nothing about how you work changes. A classifier on your own
laptop reads each prompt, in under a millisecond with no network call, and
sends it to the tier that can actually do the job.

Real turns from one of our sessions, pinned to the top model:

rename a variable → ran on the cheapest tier, 19.5x less
add a docstring → ran on the cheapest tier, 19.5x less
make it safe under concurrent mutation, prove no double counting, add tests →
climbed to a stronger model on its own

Nobody chose any of that. The session cost $0.11 instead of $0.44.

It runs on your machine, so prompts never leave it to be routed. Every decision
is logged with the model that was asked for beside the model that ran, so
nothing is hidden from the developer and there is a per request record if
somebody needs one.

It is open source, and the README is honest about what it does not do well yet.
If you try it, what we would most like to hear is where the routing got it
wrong.

---

## Article

### Choosing a model means knowing something you cannot know yet

Arul and I built a local routing gateway in our own time. This is the problem
that led us there, and it started with our own habits rather than anyone
else's.

Most teams paying for AI arrive at the same sensible position: use the
expensive model where it earns its price, and something cheaper where it does
not. It is good advice. We have given it and we have tried to take it.

We want to be precise about why it is hard to act on, because the obvious
explanations are wrong. It is not carelessness with money, and it is not that
the guidance needed to be communicated more clearly.

It is that the question is difficult to answer at the moment it gets asked.

You are about to send a prompt. Is it hard? Often you genuinely do not know.
Some one line requests need real reasoning. Some long ones are mechanical. How
hard a task is turns out to be a property of the answer, and the answer is what
you do not have yet.

Then consider what each guess costs. Guess low and the cheaper model might
produce something subtly wrong, and you lose the turn and a little confidence
in the tool. Guess high and it works, every time.

Given that, pinning the strongest model is not laziness. It is the right call
with the information available. Both of us do it. That is precisely why asking
people to choose differently rarely changes the number.

### Moving the decision

If the decision is hard to make in advance, it should be made by something that
sees the prompt and can decide in the moment. That is what we built.

WormHole is a small gateway that runs on your own machine, between your coding
harness and the model providers. Your harness stays pinned to whatever it is
pinned to. A classifier on the laptop reads each prompt and decides which tier
it needs, in under a millisecond, with no network call. The prompt then goes to
the cheapest model that clears that bar.

We found this shows up in two places, and the same fix covers both.

The first is anything automated calling `codex exec`. A script, a CI job, a
pre-commit hook, another agent. Whoever wrote it chose a model on the day they
wrote it, usually the best one available, and there has been no reason to
revisit that line since. Every invocation since has paid top rate.

The second is a person in an interactive session, who set a strong model in
their config once and moved on, for exactly the reasons above.

The gateway cannot tell the two apart, and does not need to.

### What it actually did

Here is a real Codex session with the model pinned to `gpt-5.6-sol`, the
strongest tier. These rows come from the gateway's log, not from a projection.

A variable rename ran on `gpt-5.6-luna`, the cheapest reasoning tier, at
roughly a twentieth of the cost. Adding a docstring did the same. Then this
prompt: make the file safe under concurrent mutation of the list, prove the
total cannot double count, and add tests that demonstrate the race is gone.

That one climbed to `gpt-5.6-terra` on its own. It snapshotted the list, wrote
threaded regression tests, ran them, and they passed.

Nobody chose any of it. The session cost $0.11 rather than $0.44.

The escalation matters to us as much as the saving. A router that only ever
routes down is a cost cut wearing a technical disguise, and the first time it
sends a migration to a small model it has cost more than it saved.

### What it does not do

The classifier is honest about being a classifier. It gets the easy end right
and it still under rates short hard instructions, because every hard example in
its training data is a long bug report and nothing in it resembles a terse
imperative sentence. Your own judged traffic is what closes that, and the
learning loop exists for exactly that reason.

We also removed the headline number this project used to advertise. Re
measuring it with the providers' real token counts, rather than an estimate,
moved one figure from 96 percent to 45 percent. That is the kind of number that
should not sit in a README, so it does not. There is a command that measures it
on your own fleet, and that answer is worth more than ours.

Some cheap models cannot drive an agent loop at all, which is a different
question from whether they support tool calling. Three of them failed in our
testing, in ways their capability tier did not predict. They are marked in the
config with a note recording what we saw.

### Where it runs

On your laptop. Routing is a local classifier, so prompts do not leave the
machine to be routed. The log is a local SQLite file. Every decision records
the model that was asked for beside the model that ran, so a developer can see
what happened and there is a per request record if it is ever needed.

It is open source, Apache 2.0, and built in our own time. If you try it, the
thing we would most like to hear about is where the routing was wrong.
