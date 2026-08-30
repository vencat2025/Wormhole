"""Measure whether routing to cheaper models actually costs quality.

Everything else this project reports is about cost. The open question has
always been the other half: routing is only worth anything if the cheaper
model still does the job, and until now the only quality signal was one model
grading another, with no ground truth behind it.

MBPP ships executable test cases, so correctness here is not a matter of
opinion. For each task this runs two arms over the same problems:

  routed   - the model WormHole's router selects for that prompt
  baseline - a fixed strong model, the "always use the flagship" policy

and executes the produced code against MBPP's own assertions. The result is a
pass rate and a cost for each arm, which is the comparison the project's cost
claims have been missing.

Usage:
  python scripts/evaluate_routing_quality.py --n 30
  python scripts/evaluate_routing_quality.py --n 30 --baseline groq/openai/gpt-oss-120b
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import litellm  # noqa: E402
from config import settings  # noqa: E402
from services.dispatcher import calculate_cost, is_model_routable  # noqa: E402
from services.router import route_prompt  # noqa: E402
from services.dispatcher import _load_exhausted_providers  # noqa: E402

litellm.drop_params = True

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
CACHE = os.path.join(ROOT, "data", "benchmark_cache", "mbpp_full.json")

def build_prompt(task: Dict) -> str:
    """MBPP's standard protocol: show one assertion so the model knows the
    required function name and signature.

    Without it the tests call a name the model was never told, and everything
    fails regardless of the model -- which is a harness bug that looks exactly
    like every model being terrible.
    """
    return (f"{task['text']}\n\n"
            f"Your code must pass this test:\n{task['test_list'][0]}\n\n"
            "Reply with ONLY the Python function. No prose, no markdown fences.")


def load_tasks(n: int) -> List[Dict]:
    if os.path.exists(CACHE):
        recs = json.load(open(CACHE))
    else:
        raw = httpx.get(MBPP_URL, timeout=90, follow_redirects=True).text
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(recs, open(CACHE, "w"))
    usable = [r for r in recs if r.get("test_list") and r.get("text")]
    # Evenly spaced rather than the first n, which cluster by topic.
    step = max(1, len(usable) // n)
    return usable[::step][:n]


def run_tests(code: str, tests: List[str]) -> bool:
    """True if the generated code passes MBPP's own assertions."""
    code = code.replace("```python", "").replace("```", "")
    program = code + "\n" + "\n".join(tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=25)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


async def answer(model: str, prompt: str, max_tokens: int):
    """Return (code, usage) -- usage being the provider's own token counts.

    A reasoning model that runs out of budget returns finish_reason "length"
    and often no content at all. That is the harness cutting it off, not the
    model answering wrongly, and counting it as a failure understates the model
    badly -- it read as -58 points before this was caught.

    The usage half matters because this script prints a cost next to a pass
    rate, and the pass rate is a measurement: the code is executed and it either
    passes or it does not. Estimating the cost beside it -- this previously
    guessed tokens as len(text) // 4 -- dressed an approximation in the same
    clothes as the measurement standing next to it. Reasoning models make that
    guess badly in particular, because tokens they spend thinking never appear
    in the text at all.
    """
    kwargs = {"api_base": settings.OLLAMA_BASE_URL} if model.startswith("ollama/") else {}
    try:
        r = await litellm.acompletion(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens, **kwargs)
        choice = r.choices[0]
        content = choice.message.content or ""
        u = getattr(r, "usage", None)
        usage = None
        if u is not None:
            pt = getattr(u, "prompt_tokens", None)
            ct = getattr(u, "completion_tokens", None)
            if pt is not None or ct is not None:
                usage = (pt or 0, ct or 0)
        if choice.finish_reason == "length" and not content.strip():
            print(f"      {model}: truncated at {max_tokens} tokens, no content; "
                  f"raise --max-tokens", file=sys.stderr)
            return None, usage
        return content, usage
    except Exception as err:
        print(f"      {model}: {type(err).__name__}", file=sys.stderr)
        return None, None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of MBPP tasks")
    ap.add_argument("--baseline", default=None, help="fixed model for the comparison arm")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between tasks")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="output budget. Reasoning models spend most of it thinking; "
                         "too low truncates them mid-thought and scores it as a wrong answer")
    args = ap.parse_args()

    # Honour providers already known to be out of credit, so the baseline arm
    # is not silently pointed at a model that cannot answer.
    _load_exhausted_providers()

    baseline = args.baseline or next(
        (m.id for m in sorted(settings.CANDIDATE_MODELS, key=lambda c: -c.input_cost_per_1k)
         if is_model_routable(m.id)), settings.FALLBACK_MODEL)

    tasks = load_tasks(args.n)
    print(f"Tasks: {len(tasks)} MBPP problems (executable tests)")
    print(f"Baseline arm: {baseline}\n")

    # A handful of tasks cannot separate the arms, and the resulting number
    # looks authoritative anyway. Measured at --n 4: routing scored 2 against
    # the baseline's 4, which reads as a rout and is a coin landing badly.
    # Say so before the run rather than under the result, where it would look
    # like an excuse.
    if len(tasks) < 20:
        print(f"NOTE: {len(tasks)} tasks is too few to tell the arms apart. Expect swings "
              f"of tens of points from luck alone.\n      Use --n 24 or more before "
              f"quoting the outcome.\n", file=sys.stderr)

    stats = {"routed": {"pass": 0, "cost": 0.0, "errors": 0, "estimated": 0, "models": {}},
             "baseline": {"pass": 0, "cost": 0.0, "errors": 0, "estimated": 0}}

    for i, t in enumerate(tasks, 1):
        prompt = build_prompt(t)
        routed_model, _ = await route_prompt(t["text"])
        if not is_model_routable(routed_model):
            routed_model = settings.FALLBACK_MODEL

        for arm, model in (("routed", routed_model), ("baseline", baseline)):
            code, usage = await answer(model, prompt, args.max_tokens)
            if code is None:
                # The call failed. That is an availability problem, not the
                # model getting the answer wrong, and conflating the two would
                # make this benchmark report a lie.
                stats[arm]["errors"] += 1
                ok = False
            else:
                ok = run_tests(code, t["test_list"])
                stats[arm]["pass"] += ok
            # Prefer the provider's own token counts. Fall back to the
            # character heuristic only where a provider reports nothing, and
            # count how often that happened so the footer can say so rather
            # than presenting a mixed figure as if it were all measured.
            if usage:
                stats[arm]["cost"] += calculate_cost(model, usage[0], usage[1])
            else:
                stats[arm]["cost"] += calculate_cost(model, len(prompt) // 4, len(code or "") // 4)
                stats[arm]["estimated"] += 1
            if arm == "routed":
                stats["routed"]["models"][model] = stats["routed"]["models"].get(model, 0) + 1
                mark = "pass" if ok else "FAIL"
        print(f"  [{i:>3}/{len(tasks)}] routed={routed_model:26s} {mark}")
        await asyncio.sleep(args.pause)

    n = len(tasks)
    print("\n" + "=" * 62)
    print(f"{'arm':10s} {'pass rate':>12s} {'cost':>12s}")
    for arm in ("routed", "baseline"):
        st = stats[arm]
        scored = n - st["errors"]
        rate = (st["pass"] / scored * 100) if scored else 0.0
        note = f"   ({st['errors']} call errors excluded)" if st["errors"] else ""
        print(f"{arm:10s} {st['pass']}/{scored} ({rate:>5.1f}%) ${st['cost']:>10.5f}{note}")
    if stats["baseline"]["errors"] == n:
        print("\nBaseline could not be reached at all; pass --baseline with a model "
              "you have credit for. No comparison is possible from this run.")
        return 1
    saved = stats["baseline"]["cost"] - stats["routed"]["cost"]
    if stats["baseline"]["cost"]:
        print(f"\ncost difference: ${saved:.5f} ({saved/stats['baseline']['cost']*100:.1f}% lower)")
    # Lead with tasks solved out of tasks given. Excluding an arm's failed calls
    # from its denominator flatters it: an arm that errored twice and solved the
    # same number of tasks scores several points "higher" purely because it was
    # asked fewer questions. Report the like-for-like figure first and the
    # error-adjusted one second, clearly labelled.
    r_pass, b_pass = stats["routed"]["pass"], stats["baseline"]["pass"]
    print(f"tasks solved (of {n} given): routed {r_pass}, baseline {b_pass} "
          f"-> {(r_pass - b_pass) / n * 100:+.1f} points")

    r_scored = n - stats["routed"]["errors"]
    b_scored = n - stats["baseline"]["errors"]
    if stats["routed"]["errors"] or stats["baseline"]["errors"]:
        print(f"  excluding calls that failed outright: "
              f"{r_pass}/{r_scored} vs {b_pass}/{b_scored}. Those failures are an "
              f"availability problem, but the arms then answered different numbers "
              f"of questions, so this figure is not a like-for-like comparison.")
    print("\nrouted traffic went to:")
    for m, c in sorted(stats["routed"]["models"].items(), key=lambda kv: -kv[1]):
        print(f"  {m:30s} {c}")
    print("\nCorrectness is MBPP's own assertions, executed. Not a model's opinion.")
    est = stats["routed"]["estimated"] + stats["baseline"]["estimated"]
    if est:
        print(f"Cost: {est} of {n * 2} calls returned no usage, so those are estimated "
              f"from text length; the rest are the providers' own token counts.")
    else:
        print("Cost is the providers' own reported token counts, at litellm's published rates.")
    if n < 20:
        print(f"\nWith only {n} tasks this comparison is noise. Re-run with --n 24 or more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
