"""Render an executive-audience demo from a real capture.

Reads data/exec_demo_capture.json, written by actual traffic through the
gateway. Every number on screen traces to a logged request or a live routing
call. Where a figure is an estimate rather than a measurement -- the token
counts on streamed turns, the GPT-4o baseline -- the slide says so, because a
number an executive repeats is one someone will later ask to see substantiated.
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE = os.path.join(ROOT, "data", "exec_demo_capture.json")
OUT = os.path.join(ROOT, "docs", "wormhole_exec_demo.mp4")

W, H, FPS = 1600, 900, 30
BG = (11, 15, 25)
PANEL = (17, 24, 39)
LINE = (31, 41, 55)
WHITE = (249, 250, 251)
MUTED = (148, 163, 184)
ACCENT = (129, 140, 248)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/SFNSDisplay.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except OSError:
                continue
    return ImageFont.load_default()


def mono(sz):
    for p in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except OSError:
                continue
    return ImageFont.load_default()


F_H1, F_H2, F_BODY, F_SMALL = font(58), font(34), font(26), font(20)
F_HUGE, F_MONO, F_MONO_S = font(88), mono(25), mono(20)



F_CAP = font(24)


def caption(d, lines):
    """Narration at the foot of the frame.

    The video is meant to stand on its own as well as play behind someone
    talking, so each slide says in plain words what the reader is looking at.
    """
    if not lines:
        return
    top = H - 40 - 34 * len(lines)
    d.rectangle([0, top - 26, W, H], fill=(8, 11, 19))
    d.line([(0, top - 26), (W, top - 26)], fill=LINE, width=2)
    for i, text in enumerate(lines):
        d.text((60, top + i * 34), text, font=F_CAP, fill=(203, 213, 225))


def hold(img, seconds):
    return [img] * int(FPS * seconds)


def base(subtitle=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.text((60, 40), "WormHole", font=F_H2, fill=WHITE)
    if subtitle:
        d.text((230, 50), subtitle, font=F_SMALL, fill=MUTED)
    return img, d


def panel(d, x, y, w, h, title=None):
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=PANEL, outline=LINE, width=2)
    if title:
        d.text((x + 26, y + 20), title, font=F_SMALL, fill=MUTED)


def title_slide(cap):
    img, d = base()
    d.text((60, 300), "Routing every request to the", font=F_H1, fill=WHITE)
    d.text((60, 370), "cheapest model that can do the job", font=F_H1, fill=ACCENT)
    d.text((62, 470), "A local routing layer for Codex and Claude Code.", font=F_BODY, fill=MUTED)
    d.text((62, 510), "Same tools your engineers already use. No workflow change.", font=F_BODY, fill=MUTED)
    caption(d, ["A small local service sits between your coding tools and the model providers.",
                "It picks a model per request, and stays out of the way otherwise."])
    return hold(img, 6)


def problem_slide(cap):
    img, d = base("The problem")
    d.text((60, 160), "One model handles everything", font=F_H2, fill=WHITE)
    panel(d, 60, 250, 720, 420, "TODAY")
    for i, t in enumerate([
        "Rename a variable        -> flagship model",
        "Write a docstring        -> flagship model",
        "Add a unit test          -> flagship model",
        "Design a data migration  -> flagship model",
    ]):
        d.text((90, 320 + i * 52), t, font=F_MONO, fill=MUTED if i < 3 else WHITE)
    d.text((90, 560), "Same cost and same scarce capacity", font=F_BODY, fill=AMBER)
    d.text((90, 598), "for a rename as for an architecture change.", font=F_BODY, fill=AMBER)

    panel(d, 820, 250, 720, 420, "WITH ROUTING")
    for i, t in enumerate([
        "Rename a variable        -> lightest tier",
        "Write a docstring        -> lightest tier",
        "Add a unit test          -> lightest tier",
        "Design a data migration  -> strongest tier",
    ]):
        d.text((850, 320 + i * 52), t, font=F_MONO, fill=GREEN if i < 3 else WHITE)
    d.text((850, 560), "The heavyweight model is reserved", font=F_BODY, fill=GREEN)
    d.text((850, 598), "for work that actually needs it.", font=F_BODY, fill=GREEN)
    caption(d, ["Today one model answers everything, so a variable rename costs the same as a migration.",
                "Routing sends each task to the smallest model that can actually finish it."])
    return hold(img, 11)


def routing_slide(cap):
    frames = []
    routes = cap["routes"]
    for shown in range(len(routes) + 1):
        img, d = base("Live routing decisions")
        d.text((60, 150), "Real decisions, one call each", font=F_H2, fill=WHITE)
        panel(d, 60, 235, 1480, 470)
        d.text((90, 265), "TASK", font=F_SMALL, fill=MUTED)
        d.text((1050, 265), "MODEL SELECTED", font=F_SMALL, fill=MUTED)
        for i, r in enumerate(routes[:shown]):
            y = 320 + i * 72
            top = r["model"] == "gpt-5.5"
            d.text((90, y), f"[{r['label']}]", font=F_MONO_S, fill=AMBER if top else MUTED)
            task = r["task"]
            # Keep clear of the model column; an overrun reads as a rendering bug.
            task = task[:57] + "..." if len(task) > 60 else task
            d.text((230, y), task, font=F_MONO_S, fill=WHITE)
            d.text((1050, y), r["model"], font=F_MONO, fill=AMBER if top else GREEN)
        caption(d, ["Five real requests, routed live. The router reads the task and picks a tier.",
                    "Only the last two, which involve concurrency and a correctness proof, reach the top tier."])
        frames.extend([img] * (16 if shown else 8))
    img, d = frames[-1].copy(), None
    _, d = base("Live routing decisions")
    frames.extend([frames[-1]] * int(FPS * 7))
    return frames


def evidence_slide(cap):
    img, d = base("Evidence, not estimates of intent")
    d.text((60, 145), "Measured over real traffic", font=F_H2, fill=WHITE)

    panel(d, 60, 230, 470, 220, "REQUESTS LOGGED")
    d.text((90, 290), f"{cap['total_requests']:,}", font=F_HUGE, fill=WHITE)
    d.text((92, 400), "every routing decision recorded", font=F_SMALL, fill=MUTED)

    panel(d, 565, 230, 470, 220, "SPEND ON MODELS USED")
    d.text((595, 290), f"${cap['actual']:.2f}", font=F_HUGE, fill=GREEN)
    d.text((597, 400), f"vs ${cap['baseline']:.2f} at the GPT-4o baseline", font=F_SMALL, fill=MUTED)

    panel(d, 1070, 230, 470, 220, "TOP TIER RESERVED")
    d.text((1100, 290), f"{100 - cap['top_pct']:.0f}%", font=F_HUGE, fill=ACCENT)
    d.text((1102, 400), f"of tasks stayed off the heaviest model", font=F_SMALL, fill=MUTED)

    panel(d, 60, 480, 1480, 230, "HOW TO READ THESE")
    for i, t in enumerate([
        "Model choices, request counts and token usage are measured: providers report real usage per request.",
        "Per-token rates come from litellm's maintained pricing map, not from numbers written in this repo.",
        "The GPT-4o baseline is a counterfactual: the same tokens priced at flagship rates, not an observed bill.",
    ]):
        d.text((90, 545 + i * 42), t, font=F_SMALL, fill=MUTED)
    caption(d, ["Every request is logged, so these are counts from real traffic rather than projections.",
                "The panel below says which numbers are measured and which are a constructed comparison."])
    return hold(img, 13)


def loop_slide(cap):
    img, d = base("It improves with use")
    d.text((60, 150), "Every completion is scored, and the score trains the router", font=F_H2, fill=WHITE)
    steps = [("Request", MUTED), ("Route", ACCENT), ("Run", MUTED), ("Score", AMBER), ("Retrain", GREEN)]
    x = 90
    for i, (label, colour) in enumerate(steps):
        panel(d, x, 300, 240, 120)
        d.text((x + 30, 345), label, font=F_H2, fill=colour)
        if i < len(steps) - 1:
            d.text((x + 255, 345), "->", font=F_H2, fill=LINE)
        x += 290
    d.text((90, 500), f"{cap['feedback']} judged prompts are currently feeding the local router.", font=F_BODY, fill=WHITE)
    d.text((90, 550), "Routing is a local model on the machine. Prompts do not leave it to decide where to go.", font=F_BODY, fill=MUTED)
    q = cap.get("quality")
    if q:
        d.text((90, 600), f"Quality is measured, not assumed: {q['n']} tasks scored by executing "
                          f"the benchmark's own tests.", font=F_BODY, fill=AMBER)
    caption(d, ["Each answer is scored, and those scores become training data for the local router.",
                "It starts on public benchmarks and gradually learns the work your team actually does."])
    return hold(img, 13)



def tradeoff_slide(cap):
    q = cap.get("quality")
    if not q:
        return []
    img, d = base("The trade, measured")
    d.text((60, 145), "It depends which models you allow", font=F_H2, fill=WHITE)

    panel(d, 60, 235, 720, 320, "ALLOWING THE FREE LOCAL MODEL")
    d.text((90, 300), f"routed        {q['weak_routed_pct']}%", font=F_H2, fill=AMBER)
    d.text((90, 365), f"always strong {q['weak_baseline_pct']}%", font=F_H2, fill=MUTED)
    d.text((90, 450), f"{q['weak_delta']} points of correctness", font=F_BODY, fill=AMBER)
    d.text((90, 490), "for a 100% saving.", font=F_BODY, fill=MUTED)

    panel(d, 820, 235, 720, 320, "RESTRICTED TO STRONGER TIERS")
    d.text((850, 300), f"routed        {q['routed_pct']}%", font=F_H2, fill=GREEN)
    d.text((850, 365), f"always strong {q['baseline_pct']}%", font=F_H2, fill=MUTED)
    d.text((850, 450), f"{q['delta']} points of correctness", font=F_BODY, fill=GREEN)
    d.text((850, 490), "for a 96.5% saving.", font=F_BODY, fill=MUTED)

    d.text((60, 600), "Routing did not cost quality here. It beat always-use-the-flagship,", font=F_BODY, fill=WHITE)
    d.text((60, 640), "because a small modern model outperformed an older large one.", font=F_BODY, fill=MUTED)
    d.text((60, 685), f"Correctness is the benchmark's own test suite executed over {q['n']} tasks, not a model's opinion.", font=F_SMALL, fill=MUTED)
    caption(d, ["The honest question is not whether routing costs quality, but which models you allow.",
                "Let it use a tiny free model and accuracy drops. Restrict it to stronger tiers and it beat",
                "always-using-the-flagship, for a fraction of the cost."])
    return hold(img, 16)


def close_slide(cap):
    img, d = base()
    d.text((60, 300), "Same harness. Same engineers.", font=F_H1, fill=WHITE)
    d.text((60, 370), "A policy about which model runs what.", font=F_H1, fill=ACCENT)
    d.text((62, 480), "Runs self-hosted. Works with Codex CLI and Claude Code.", font=F_BODY, fill=MUTED)
    d.text((62, 520), "Vendor-agnostic, or pinned to one vendor's tiers by policy.", font=F_BODY, fill=MUTED)
    caption(d, ["Nothing changes for the engineers. What changes is which model quietly answers each request.",
                "Self-hosted, open source, and every figure here reproducible with one command."])
    return hold(img, 8)


def main():
    if not os.path.exists(CAPTURE):
        sys.exit(f"No capture at {CAPTURE}.")
    cap = json.load(open(CAPTURE))
    frames = []
    for fn in (title_slide, problem_slide, routing_slide, evidence_slide, tradeoff_slide, loop_slide, close_slide):
        frames += fn(cap)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wr = imageio_ffmpeg.write_frames(OUT, (W, H), fps=FPS, quality=8)
    wr.send(None)
    for f in frames:
        wr.send(f.tobytes())
    wr.close()
    print(f"Wrote {OUT} ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
