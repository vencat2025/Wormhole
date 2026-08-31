# Launch copy

Two pieces, both built around the same argument: the decision about which model
to use is being asked of the one person who cannot make it in advance.

Every number below came from the gateway's own log during a real Codex session.
Regenerate them with `scripts/evaluate_routing_quality.py` or by reading
`/api/logs`.

---

## LinkedIn post

Your engineers picked the best model. Of course they did.

Every company trying to control AI spend eventually sends the same message.
Use the cheaper model when you can. Save the expensive one for hard problems.
It goes in a wiki, or a quarterly reminder, or a dashboard nobody opens.

It never works. Not because people are careless.

You cannot tell how hard a prompt is until it has been answered. Guess low and
you risk a wrong answer and a wasted turn. Guess high and it always works. So
every engineer pins the strongest model and never thinks about it again. That
is the rational choice. It is also the expensive one.

I spent my free time building a small gateway that makes the choice per
request instead.

Your Codex or Claude Code stays pinned to whatever it is pinned to. Nothing
about how anyone works changes. A classifier on your own laptop reads each
prompt, in under a millisecond, with no network call, and sends it to the tier
that can actually do the job.

Here are real turns from one session, pinned to the top model:

rename a variable → ran on the cheapest tier, 19.5x less
add a docstring → ran on the cheapest tier, 19.5x less
make it safe under concurrent mutation, prove no double counting, add tests →
climbed to a stronger model on its own

Nobody was asked to choose. That session cost $0.11 instead of $0.44.

It runs on your machine. Prompts never leave it to be routed. Every decision is
logged with the model that was asked for next to the model that ran, so nothing
is hidden from the developer and finance gets a per request record.

It is open source and it is honest about what it cannot do yet. Link below.

---

## Article

### The model budget nobody can follow

Every engineering organisation that pays for AI eventually arrives at the same
policy. Use the cheaper model when you can. Save the expensive one for the hard
work.

It is a reasonable thing to ask and it does not hold. I want to be precise
about why, because the usual explanation is wrong. It is not that engineers are
careless with money, or that the guidance was not communicated clearly enough,
or that the dashboard needed better charts.

It is that the question is unanswerable at the moment it is asked.

You are about to send a prompt. Is it hard? You genuinely do not know. Some
one line requests need real reasoning. Some long ones are mechanical. The
difficulty of a task is a property of the answer, and you do not have the
answer yet. That is the whole problem.

Now consider the incentives. Guess low and the cheap model might produce
something subtly wrong, and you lose the turn and some trust. Guess high and it
works, every time, and the cost lands on a budget line you never see.

Given that, pinning the strongest model is not laziness. It is correct. Every
sensible engineer arrives there, which is exactly why the policy fails.

### Moving the decision

The decision has to move to something that can make it. That is what I built.

WormHole is a small gateway that runs on your own machine, between your coding
harness and the model providers. Your harness stays pinned to whatever it is
pinned to. A classifier on the laptop reads each prompt and decides which tier
it needs, in under a millisecond, with no network call. The prompt then goes to
the cheapest model that clears that bar.

This shows up in two places, and the fix is the same for both.

The first is anything automated calling `codex exec`. A script, a CI job, a
pre-commit hook, another agent. Whoever wrote it pinned a model on the day they
wrote it, usually the best one available, and nobody has revisited that line
since. Every invocation pays top rate forever.

The second is a person in an interactive session. They set the strongest model
in their config once and moved on, for exactly the reasons above.

The gateway cannot tell the two apart, and does not need to.

### What it actually did

Here is a real Codex session with the model pinned to `gpt-5.6-sol`, the
strongest tier. These rows are from the gateway's log, not a projection.

A variable rename ran on `gpt-5.6-luna`, the cheapest reasoning tier, at
roughly a twentieth of the cost. Adding a docstring did the same. Then this
prompt: make the file safe under concurrent mutation of the list, prove the
total cannot double count, and add tests that demonstrate the race is gone.

That one climbed to `gpt-5.6-terra` on its own. It snapshotted the list, wrote
threaded regression tests, ran them, and they passed.

Nobody was asked to choose. The session cost $0.11 rather than $0.44.

The escalation matters as much as the saving. A router that only ever routes
down is a cost cut wearing a technical disguise, and the first time it sends a
migration to a small model it has cost you more than it saved.

### What it does not do

The classifier is honest about being a classifier. It gets the easy end right
and it still under rates short hard instructions, because every hard example in
its training data is a long bug report and nothing in it resembles a terse
imperative sentence. Your own judged traffic is what closes that, and the
learning loop exists for exactly that reason.

I also removed the headline number this project used to advertise. Re measuring
it with the providers' real token counts, rather than an estimate, moved one
figure from 96 percent to 45 percent. That is the kind of number that should
not be in a README, so it is not. There is a command that measures it on your
own fleet, and that answer is worth more than mine.

Some cheap models cannot drive an agent loop at all, which is a different
problem from whether they support tool calling. Three of them failed in
testing, in ways their capability tier did not predict. They are marked in the
config with a note saying what was observed.

### Where it runs

On your laptop. Routing is a local classifier, so prompts do not leave the
machine to be routed. The log is a local SQLite file. Every decision records
the model that was asked for beside the model that ran, so a developer can see
what happened and finance can audit it per request.

It is open source, Apache 2.0, and built in my own time. If you try it, the
thing I want to hear about is where the routing was wrong.
