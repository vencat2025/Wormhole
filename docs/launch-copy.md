# Launch copy

Both pieces make one point: choosing a model well means knowing how hard a
prompt is, and nobody knows that before it is answered.

Figures come from the gateway's own log during a real Codex session.

---

## LinkedIn post

Arul and I built a small gateway that picks the model for each prompt, so
nobody has to guess.

The problem is one we kept hitting ourselves. AI spend adds up, and the
sensible advice is to use the expensive model only where it earns its price.
We have both given that advice and both failed to follow it.

It is hard to follow for a real reason. You cannot tell how hard a prompt is
until it has been answered. Guess low and you might lose the turn to a subtly
wrong answer. Guess high and it always works. So everyone pins the strongest
model and stops thinking about it, which is the right call with the information
available.

So we moved the decision. Your Codex or Claude Code stays pinned to whatever it
is pinned to, and nothing about how you work changes. A classifier on your own
laptop reads each prompt in under a millisecond, with no network call, and
sends it to the tier that can do the job.

One of our sessions, pinned to the top model:

· rename a variable → dropped a tier
· add a docstring → dropped a tier
· make it thread-safe, prove no double counting, add tests → kept the top model

Nobody chose any of that. The escalation matters as much as the saving: a
router that only ever routes down is a cost cut in disguise.

It runs on your machine, so prompts never leave it to be routed. Every decision
is logged with the model that was asked for beside the model that ran.

Open source, Apache 2.0. If you try it, tell us where the routing got it wrong.

---

## Article

### Choosing a model means knowing something you cannot know yet

Arul and I built a local routing gateway in our own time. Here is the problem
that led us there.

Most teams paying for AI reach the same sensible position: use the expensive
model where it earns its price, something cheaper where it does not. Good
advice. We have given it and tried to take it.

It is hard to act on, and not because anyone is careless. The question is
unanswerable at the moment it is asked.

You are about to send a prompt. Is it hard? Often you do not know. Some
one-line requests need real reasoning; some long ones are mechanical. How hard
a task is turns out to be a property of the answer, and the answer is what you
do not have yet.

Then weigh the guesses. Guess low and the cheaper model might produce something
subtly wrong, costing you the turn. Guess high and it works, every time.

Pinning the strongest model is not laziness. It is correct with the information
available, which is exactly why asking people to choose differently rarely
moves the number.

### Moving the decision

If the choice is hard to make in advance, something that sees the prompt should
make it in the moment.

WormHole runs on your own machine, between your coding harness and the model
providers. Your harness stays pinned to whatever it is pinned to. A classifier
on the laptop reads each prompt, decides which tier it needs in under a
millisecond with no network call, and the request goes to the cheapest model
that clears that bar.

This shows up in two places and the same fix covers both. The first is anything
automated calling `codex exec` — a script, a CI job, a pre-commit hook, another
agent — where whoever wrote it pinned a model on the day they wrote it and
nobody has revisited that line since. The second is a person who set a strong
model in their config once and moved on.

The gateway cannot tell the two apart, and does not need to.

### What it did

A real Codex session, pinned to `gpt-5.6-sol`, from the gateway's log.

A variable rename ran a tier down. Adding a docstring did the same. Then: make
the file safe under concurrent mutation, prove the total cannot double count,
and add tests that demonstrate the race is gone.

That one kept `gpt-5.6-sol`. It snapshotted the list, wrote threaded regression
tests, ran them, and they passed.

Nobody chose any of it.

### What it does not do yet

The classifier gets the easy end right and still misjudges some short, hard
instructions — every example in its training data is a bug report or a coding
exercise, and none of it is the terse imperative sentence people actually type.
`MIN_ROUTING_TIER` bounds how far wrong that can go, and the scores from your
own traffic are what improve it.

Some cheap models cannot sustain an agent loop at all, which is a different
question from whether they support tool calling. The ones we saw fail are
marked in the config with a note saying what we observed.

### Where it runs

On your laptop. Routing is a local classifier, so prompts do not leave the
machine to be routed. The log is a local SQLite file, and every decision records
the model that was asked for beside the model that ran.

Open source, Apache 2.0, built in our own time. If you try it, what we would
most like to hear is where the routing was wrong.
