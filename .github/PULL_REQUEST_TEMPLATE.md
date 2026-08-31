<!--
Pull request creation is currently limited to collaborators. If you are
reading this as an outside contributor, please open an issue instead:
a routing case we got wrong is the most useful thing you can send us.
See CONTRIBUTING.md.
-->

## What this changes

<!-- One or two sentences. -->

## How you know it works

<!--
Please paste output rather than describing it.

`pytest tests/ -q` is the minimum. The tests that need a provider key skip
without one, so a clean run on a fork is 16 passed / 3 skipped.

If you changed routing, also paste a before and after: which model each prompt
went to. Routing changes are easy to make and hard to evaluate, and a diff
alone does not show whether the decisions got better.
-->

## If this changes a published number

Every figure in the README traces to a run. If your change moves one, include
the run that produced the new value.

Watch the denominator in particular. When calls fail, that arm attempted fewer
questions, and a pass rate over the smaller number flatters it. Compare tasks
solved out of tasks given. This repository has published that exact mistake
before.
