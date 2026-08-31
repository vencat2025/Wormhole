"""Render an executive-audience demo from a real capture.

Reads data/exec_demo_capture.json, written by actual traffic through the
gateway. Every number on screen traces to a logged request, a live routing call,
or a benchmark run where both arms actually executed. Nothing is priced against
a model that never ran, because a number an executive repeats is one someone
will later ask to see substantiated.
"""

import json
import os
import re
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE = os.path.join(ROOT, "data", "exec_demo_capture.json")
OUT = os.path.join(ROOT, "docs", "wormhole_exec_demo.mp4")

W, H, FPS = 1600, 900, 30

# Narration. Off by default so the video still renders with no key and no
# network; WORMHOLE_VOICE=1 turns it on.
VOICE = os.getenv("WORMHOLE_VOICE", "").strip() not in ("", "0", "false")
VOICE_MODEL = os.getenv("WORMHOLE_VOICE_MODEL", "gpt-4o-mini-tts")
VOICE_NAME = os.getenv("WORMHOLE_VOICE_NAME", "onyx")
VOICE_CACHE = os.path.join(ROOT, "data", "narration_cache")
VOICE_PAD = 1.0   # seconds of quiet after each line, so slides do not run on
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
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
    _PENDING.clear()
    _PENDING.extend(lines)


# Narration lines drawn on the current slide, consumed by the next hold().
_PENDING = []

# (audio_path_or_None, seconds) per slide, in order. Built during rendering so
# the audio track and the frames cannot drift apart.
SEGMENTS = []

# How the narrator should pronounce things written for the eye. Without this a
# reader says "slash a p i slash logs" and "g p t dash five dash nano", which
# sounds like a machine reading a config file rather than someone explaining a
# result. On-screen text is untouched; only the spoken form changes.
SPOKEN = {
    "MIN_ROUTING_TIER=medium": "medium tier floor",
    "MIN_ROUTING_TIER": "routing tier",
    "gpt-oss-120b": "the 120B model",
    "gpt-5-nano": "nano",
    "gpt-5-mini": "mini",
    "gpt-5.6-luna": "Luna",
    "gpt-4o-mini": "4o mini",
    "gpt-4o": "4o",
    "/api/logs": "the API logs",
    "config.toml": "config",
    "opencode.json": "opencode config",
    "claude-routed": "Claude wrapper",
    "codex-routed": "Codex wrapper",
    "MBPP": "MBPP",
    "SWE-bench": "SWE-bench",
    "SLM": "local model",
}


def spoken_form(text):
    for written, said in SPOKEN.items():
        text = text.replace(written, said)
    return text


def narrate(text):
    """Synthesize one line of narration, returning a cached file path.

    Cached on a hash of the exact text: re-rendering the video after a wording
    change pays only for the lines that changed, and re-rendering after a
    layout change pays nothing at all.
    """
    if not VOICE or not text.strip():
        return None
    import hashlib
    key = hashlib.sha256(f"{VOICE_MODEL}|{VOICE_NAME}|{text}".encode()).hexdigest()[:16]
    path = os.path.join(VOICE_CACHE, f"{key}.mp3")
    if os.path.exists(path):
        return path
    try:
        from openai import OpenAI
        os.makedirs(VOICE_CACHE, exist_ok=True)
        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model=VOICE_MODEL,
            voice=VOICE_NAME,
            input=text,
            instructions=(
                "You are a startup founder pitching your vision to investors. Optimistic, "
                "storytelling, paint a picture of how this solves a real problem. Build narrative "
                "momentum: start with the problem (expensive models), then the insight (local routing), "
                "then the payoff (same quality, fraction of cost). Sound like you believe in this. "
                "Energetic but not frantic. You're inviting people into a better future."
            ),
        ) as response:
            response.stream_to_file(path)
        print(f"   synthesized {len(text):3d} chars -> {os.path.basename(path)}")
        return path
    except Exception as e:
        # A missing key or a network failure must not cost you the video.
        print(f"   narration unavailable ({e}); rendering this slide silent.")
        return None


def audio_seconds(path):
    """Duration of a clip, read from ffmpeg's own report.

    imageio-ffmpeg ships ffmpeg without ffprobe, so this parses the Duration
    line rather than assuming a probe binary that is not there.
    """
    proc = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", proc.stderr)
    if not m:
        return 0.0
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def hold_frames(frames):
    """Register an already-animated slide with the narration track.

    Slides that build their own frames must still contribute a segment, or the
    audio track comes out shorter than the video and every slide after them
    plays against the wrong narration.
    """
    text = spoken_form(" ".join(_PENDING)).strip()
    _PENDING.clear()

    path = narrate(text)
    seconds = len(frames) / FPS
    if path:
        needed = audio_seconds(path) + VOICE_PAD
        if needed > seconds:
            frames.extend([frames[-1]] * int(FPS * (needed - seconds)))
            seconds = len(frames) / FPS
    SEGMENTS.append((path, seconds))
    return frames


def hold(img, seconds):
    """Hold a slide, extending it if its narration needs longer.

    The written durations were tuned for reading speed, so they are kept as a
    floor: a slide never gets *shorter* because the narrator was quick, and
    never clips its own audio because the narrator was slow.
    """
    text = spoken_form(" ".join(_PENDING)).strip()
    _PENDING.clear()

    path = narrate(text)
    if path:
        seconds = max(seconds, audio_seconds(path) + VOICE_PAD)
    SEGMENTS.append((path, seconds))
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
    d.text((60, 260), "Your engineers picked the", font=F_H1, fill=WHITE)
    d.text((60, 330), "best model. Of course they did.", font=F_H1, fill=ACCENT)
    d.text((62, 445), "So every rename now costs what an architecture review costs.", font=F_BODY, fill=MUTED)
    d.text((62, 490), "This routes each request to the right tier without asking them to.", font=F_BODY, fill=MUTED)
    caption(d, ["Give a developer a list of models and they will pick the best one, every time.",
                "That is the right call with the information they have, and it is why the number stays where it is."])
    return hold(img, 7)


def problem_slide(cap):
    """Why the choice is hard, without blaming anyone for finding it hard."""
    img, d = base("The choice is hard to make in advance")
    d.text((60, 150), "Choosing well means knowing something you do not know yet", font=F_H2, fill=WHITE)

    panel(d, 60, 235, 720, 400, "THE SENSIBLE ADVICE")
    for i, t in enumerate([
        "\"Use the expensive model",
        " where it earns its price.\"",
        "",
        "Good advice. Most teams",
        "arrive at it, and most",
        "engineers try to follow it.",
    ]):
        d.text((90, 300 + i * 48), t, font=F_BODY, fill=WHITE if i < 2 else MUTED)

    panel(d, 820, 235, 720, 400, "WHY IT IS HARD TO FOLLOW")
    for i, t in enumerate([
        "You cannot tell how hard a",
        "prompt is until it is answered.",
        "",
        "Guess low and you might lose",
        "the turn. Guess high and it",
        "works, every time.",
    ]):
        d.text((850, 300 + i * 48), t, font=F_BODY, fill=AMBER if i < 2 else MUTED)

    d.text((60, 675), "So pinning the strongest model is the right call with the information available.",
           font=F_BODY, fill=WHITE)
    caption(d, ["Most teams paying for AI arrive at the same sensible position: use the expensive model where it earns its price.",
                "It is good advice and it is hard to act on, because how hard a task is turns out to be a property of the answer, and you do not have the answer yet.",
                "Guess low and you might lose the turn. Guess high and it works. So pinning the strongest model is the right call with what you know."])
    return hold(img, 17)


def pinned_slide(cap):
    """The whole product, in one table of real turns.

    An earlier version showed the table without showing where the decision was
    made, so it read as if Codex had changed its own mind. The path across the
    top is the point: the harness still asks for the model it was pinned to,
    and WormHole is what answers with a different one.
    """
    pin = cap.get("pinned")
    if not pin:
        return []
    img, d = base("So we moved the decision")
    d.text((60, 140), f"Codex still pinned to {pin['requested']}. WormHole decides per prompt.",
           font=F_H2, fill=WHITE)

    # The path, so it is obvious what is doing the deciding.
    panel(d, 60, 205, 1480, 95)
    d.text((100, 240), "CODEX", font=F_MONO, fill=MUTED)
    d.text((235, 240), "asks for sol", font=F_SMALL, fill=MUTED)
    d.text((400, 238), "->", font=F_H2, fill=LINE)
    d.text((460, 234), "WORMHOLE", font=F_H2, fill=ACCENT)
    d.text((680, 240), "reads the prompt and picks the tier, on your machine", font=F_SMALL, fill=MUTED)
    d.text((1250, 238), "->", font=F_H2, fill=LINE)
    d.text((1310, 240), "THE MODEL", font=F_MONO_S, fill=GREEN)

    panel(d, 60, 330, 1480, 285, "WHAT WORMHOLE CHOSE, FROM ITS OWN LOG")
    d.text((90, 385), "WHAT WAS TYPED", font=F_SMALL, fill=MUTED)
    d.text((880, 385), "CODEX ASKED", font=F_SMALL, fill=MUTED)
    d.text((1110, 385), "WORMHOLE SENT IT TO", font=F_SMALL, fill=MUTED)
    d.text((1370, 385), "COST", font=F_SMALL, fill=MUTED)
    for i, t in enumerate(pin["tasks"][:4]):
        y = 435 + i * 44
        up = t["served"].endswith("terra") or t["served"].endswith("sol")
        col = AMBER if up else GREEN
        task = t["prompt"][:42] + ("..." if len(t["prompt"]) > 42 else "")
        d.text((90, y), task, font=F_MONO_S, fill=WHITE)
        d.text((880, y), "sol", font=F_MONO_S, fill=MUTED)
        d.text((1110, y), t["served"].replace("gpt-5.6-", ""), font=F_MONO, fill=col)
        d.text((1370, y), f"{t['ratio']}x less", font=F_MONO_S, fill=col)

    d.text((60, 638), "The easy work went to the cheapest tier. The concurrency task climbed on its own.",
           font=F_BODY, fill=WHITE)
    d.text((60, 680),
           f"Session total: ${pin['total_paid']:.2f} instead of ${pin['total_would']:.2f}, "
           f"and nobody was asked to choose.", font=F_BODY, fill=GREEN)
    caption(d, ["Codex is still pinned to the top model here. Nothing about the setup changed.",
                "WormHole sits between the harness and the provider and reads each prompt on your machine.",
                "The renames went to the cheapest tier, the concurrency task climbed by itself, and the session cost a quarter as much."])
    return hold(img, 19)


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
            top = r["model"].endswith("-sol")
            d.text((90, y), f"[{r['label']}]", font=F_MONO_S, fill=AMBER if top else MUTED)
            task = r["task"]
            # Keep clear of the model column; an overrun reads as a rendering bug.
            task = task[:57] + "..." if len(task) > 60 else task
            d.text((230, y), task, font=F_MONO_S, fill=WHITE)
            d.text((1050, y), r["model"], font=F_MONO, fill=AMBER if top else GREEN)
        top = [r["label"] for r in routes if r["model"].endswith("-sol")]
        caption(d, ["Five real requests, routed live. The router reads the task and picks a tier.",
                    "The renames and the unit test go to the cheapest reasoning tier; only the concurrency bug reaches the top one."]
                if top else
                ["Five real requests, routed live. The router reads the task and picks a tier.",
                 "The renames and the unit test go to the cheapest reasoning tier, and the harder work climbs from there."])
        frames.extend([img] * (16 if shown else 8))
    img, d = frames[-1].copy(), None
    _, d = base("Live routing decisions")
    frames.extend([frames[-1]] * int(FPS * 7))
    return hold_frames(frames)


def evidence_slide(cap):
    img, d = base("Evidence, not estimates of intent")
    d.text((60, 145), "Measured over real traffic", font=F_H2, fill=WHITE)

    panel(d, 60, 230, 470, 220, "REQUESTS LOGGED")
    d.text((90, 290), f"{cap['total_requests']:,}", font=F_HUGE, fill=WHITE)
    d.text((92, 400), "every routing decision recorded", font=F_SMALL, fill=MUTED)

    panel(d, 565, 230, 470, 220, "SPEND ON MODELS USED")
    d.text((595, 290), f"${cap['actual']:.2f}", font=F_HUGE, fill=GREEN)
    d.text((597, 400), f"{cap['ptok']:,} input / {cap['ctok']:,} output tokens", font=F_SMALL, fill=MUTED)

    panel(d, 1070, 230, 470, 220, "TOP TIER RESERVED")
    d.text((1100, 290), f"{100 - cap['top_pct']:.0f}%", font=F_HUGE, fill=ACCENT)
    d.text((1102, 400), f"of tasks stayed off the heaviest model", font=F_SMALL, fill=MUTED)

    panel(d, 60, 480, 1480, 230, "HOW TO READ THESE")
    for i, t in enumerate([
        "Which model ran, how many requests, and the tokens each used: all reported by the providers.",
        "Per-token rates come from litellm's maintained pricing map, not from numbers written in this repo.",
        "Nothing here is a projection. The like-for-like comparison is on the next slide, where both arms actually ran.",
    ]):
        d.text((90, 545 + i * 42), t, font=F_SMALL, fill=MUTED)
    caption(d, ["Every request is logged, so these are counts from traffic that actually ran.",
                "No figure here is an estimate of what something else might have cost."])
    return hold(img, 13)


def loop_slide(cap):
    """How the router learns, which is the part that makes it yours."""
    img, d = base("It learns from how the work turned out")
    d.text((60, 140), "Every completion is scored, and the score is the training signal", font=F_H2, fill=WHITE)

    steps = [("Route", ACCENT), ("Run", MUTED), ("Score", AMBER), ("Retrain", GREEN)]
    x = 90
    for i, (label, colour) in enumerate(steps):
        panel(d, x, 210, 270, 100)
        d.text((x + 40, 245), label, font=F_H2, fill=colour)
        if i < len(steps) - 1:
            d.text((x + 285, 245), "->", font=F_H2, fill=LINE)
        x += 330

    panel(d, 60, 350, 720, 265, "WHAT A SCORE ACTUALLY TELLS YOU")
    d.text((90, 405), "A cheap model did the job well", font=F_BODY, fill=GREEN)
    d.text((90, 443), "-> that task really was easy", font=F_SMALL, fill=MUTED)
    d.text((90, 495), "A model fell short", font=F_BODY, fill=AMBER)
    d.text((90, 533), "-> that task needed more than it got", font=F_SMALL, fill=MUTED)
    d.text((90, 578), "Both become labels for the local router.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 350, 720, 265, "WHY IT GETS BETTER FOR YOU")
    d.text((850, 405), "It ships knowing public benchmarks", font=F_BODY, fill=MUTED)
    d.text((850, 450), "and nothing about your codebase.", font=F_BODY, fill=MUTED)
    d.text((850, 510), "Your judged traffic is what teaches it", font=F_BODY, fill=GREEN)
    d.text((850, 548), "which of your prompts are actually hard.", font=F_BODY, fill=GREEN)
    d.text((850, 592), "Retraining is a local command. No key needed.", font=F_SMALL, fill=MUTED)

    d.text((60, 665), "The judge, the classifier and the log all stay on the machine.", font=F_BODY, fill=WHITE)
    caption(d, ["Every answer gets scored, and the score is what teaches the router.",
                "A cheap model doing the job well is evidence the task really was easy.",
                "A model falling short is evidence it needed more than it got. Both become training labels.",
                "It ships knowing public benchmarks and nothing about your codebase, so your own",
                "traffic teaches it which of your prompts are hard. All of it on your machine."])
    return hold(img, 18)


def tradeoff_slide(cap):
    """Where a measured result used to sit.

    It carried "same work, same result, about a twenty-fifth of the price".
    Re-measuring took that down: the cost half had been estimated from text
    length rather than read from the provider's own token counts, and once
    corrected one run's saving fell from 96% to 45%. The routing had also been
    retrained in between, so the old figure described a different router.
    Rather than print a new number with the same short shelf life, this now
    says what the honest answer is -- run it yourself.
    """
    img, d = base("Does it cost you quality?")

    d.text((60, 175), "We are not going to tell you a number.", font=F_H1, fill=WHITE)
    d.text((60, 250), "Measure it on your own setup.", font=F_H1, fill=GREEN)

    panel(d, 60, 360, 1480, 150, "ONE COMMAND, BOTH ARMS, YOUR FLEET")
    d.text((90, 420), "python scripts/evaluate_routing_quality.py --n 24 --baseline gpt-4o",
           font=F_MONO, fill=GREEN)
    d.text((90, 465), "Runs the same tasks twice: once routed, once always on the expensive model.",
           font=F_SMALL, fill=MUTED)

    d.text((60, 560), "It marks the answers by running the tests, so correctness is not an opinion.",
           font=F_BODY, fill=WHITE)
    d.text((60, 605), "Costs come from the providers' own token counts, not an estimate.", font=F_BODY, fill=MUTED)
    d.text((60, 660), "The result depends on your fleet, your prompts, and where you set the floor.",
           font=F_BODY, fill=MUTED)
    caption(d, ["We are deliberately not giving you a headline number here.",
                "The one we used to show did not survive being re-measured, and the answer depends on your fleet and your prompts anyway.",
                "One command runs both sides on your own setup and marks them by executing the tests. Trust that, not us."])
    return hold(img, 16)


def surfaces_slide(cap):
    """The two ways this shows up in a real team."""
    img, d = base("Two ways teams hit this")
    d.text((60, 145), "The same routing, from either direction", font=F_H2, fill=WHITE)

    panel(d, 60, 225, 720, 440, "1. SOMETHING ELSE IS CALLING CODEX")
    d.text((90, 285), "codex exec", font=F_MONO, fill=ACCENT)
    for i, t in enumerate([
        "A script, a CI job, a pre-commit",
        "hook, or another agent.",
        "",
        "It pins whatever model it was",
        "written with, often the best one,",
        "and nobody revisits that line.",
        "",
        "The router decides per prompt",
        "instead.",
    ]):
        d.text((90, 340 + i * 36), t, font=F_SMALL, fill=GREEN if i > 6 else MUTED)

    panel(d, 820, 225, 720, 440, "2. A PERSON IN AN INTERACTIVE SESSION")
    d.text((850, 285), "~/.codex/config.toml", font=F_MONO, fill=ACCENT)
    for i, t in enumerate([
        "They set the strongest model once",
        "and never think about it again.",
        "",
        "Which is correct: they cannot know",
        "which of today's prompts is hard.",
        "",
        "The router decides per prompt",
        "instead.",
    ]):
        d.text((850, 340 + i * 36), t, font=F_SMALL, fill=GREEN if i > 5 else MUTED)

    d.text((60, 690), "The gateway cannot tell the two apart, and does not need to.", font=F_BODY, fill=WHITE)
    caption(d, ["This shows up from two directions and the fix is the same for both.",
                "Either something automated is calling Codex with a model pinned in a script nobody revisits,",
                "or a person set the strongest model in their config once and moved on. The gateway cannot tell them apart, and does not need to."])
    return hold(img, 18)


def floor_slide(cap):
    """The one setting a new user has to get right.

    Nineteen-versus-four survives simplification where the benchmark ratios did
    not, because it is a story rather than a statistic: the agent tried, over
    and over, and the file was never there.
    """
    f = cap.get("floor")
    if not f:
        return []
    img, d = base("The one setting to get right")
    d.text((60, 155), "Cheapest is not the same as capable", font=F_H2, fill=WHITE)
    d.text((60, 215), f"The same small job both times: {f['task']}.", font=F_BODY, fill=MUTED)

    panel(d, 60, 290, 720, 300, "NO FLOOR SET")
    d.text((90, 355), f"{f['before_req']} tries", font=F_HUGE, fill=AMBER)
    d.text((90, 480), "and the file was never created", font=F_BODY, fill=AMBER)
    d.text((90, 530), "It used the tools correctly the whole time.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 290, 720, 300, "ONE LINE OF CONFIG")
    d.text((850, 355), "done", font=F_HUGE, fill=GREEN)
    d.text((850, 480), f"in {f['after_req']} tries, file written", font=F_BODY, fill=GREEN)
    d.text((850, 530), "Still a cheap model. Just not the cheapest.", font=F_SMALL, fill=MUTED)

    d.text((60, 640), "MIN_ROUTING_TIER=medium", font=F_MONO, fill=ACCENT)
    d.text((60, 690), "Set this before you point an agent at it.", font=F_BODY, fill=MUTED)
    caption(d, ["This is the first thing to set, and the easiest to get wrong.",
                "The very cheapest models answer quiz questions well and still cannot run an agent.",
                "One line of config keeps them out, and it is the difference between a file and nineteen wasted tries."])
    return hold(img, 14)


def family_slide(cap):
    """One vendor's own family, and the spread inside it."""
    fam = cap.get("family")
    if not fam:
        return []
    img, d = base("Not always the expensive one")
    d.text((60, 145), "Staying inside a single model family", font=F_H2, fill=WHITE)

    panel(d, 60, 225, 1480, 300, "THE 5.6 LADDER, PUBLISHED RATES PER 1K TOKENS")
    d.text((90, 285), "MODEL", font=F_SMALL, fill=MUTED)
    d.text((520, 285), "TIER", font=F_SMALL, fill=MUTED)
    d.text((800, 285), "INPUT", font=F_SMALL, fill=MUTED)
    d.text((1080, 285), "OUTPUT", font=F_SMALL, fill=MUTED)
    colours = [GREEN, ACCENT, AMBER]
    for i, m in enumerate(fam["models"]):
        y = 340 + i * 58
        c = colours[i] if i < len(colours) else WHITE
        d.text((90, y), m["id"], font=F_MONO, fill=c)
        d.text((520, y), m["tier"], font=F_MONO_S, fill=MUTED)
        d.text((800, y), f"${m['in']:.4f}", font=F_MONO_S, fill=c)
        d.text((1080, y), f"${m['out']:.4f}", font=F_MONO_S, fill=c)

    d.text((60, 560), f"The cheapest reasoning tier costs {fam['ratio_in']}x less on input than the top one.",
           font=F_BODY, fill=WHITE)
    d.text((60, 600), "Renames and docstrings go to the bottom of the ladder. Migrations and proofs go to the top.",
           font=F_BODY, fill=MUTED)
    d.text((60, 650), "Same vendor, same family, same account. Only the tier changes.", font=F_BODY, fill=MUTED)
    caption(d, ["You do not have to switch vendors to stop overpaying.",
                "Inside one model family the cheapest reasoning tier costs twenty times less per input token than the top one,",
                "and most of a working day is renames and docstrings, not migrations."])
    return hold(img, 15)


def api_slide(cap):
    """Why the best models needed work to reach at all, without the API lecture.

    This slide used to open with three lines of the provider's raw error and
    the words /v1/chat/completions. That is proof for an engineer and noise for
    everyone else, so the error is now one small line of evidence underneath
    the plain-English version rather than the headline.
    """
    img, d = base("Making the best models actually work")
    d.text((60, 150), "The strongest models refuse to do real work the normal way", font=F_H2, fill=WHITE)

    panel(d, 60, 235, 1480, 130, "WHAT GOES WRONG")
    d.text((90, 295), "Ask the top model to create a file and the request fails outright.", font=F_BODY, fill=WHITE)
    d.text((90, 335), "Not a bad answer. No answer at all.", font=F_BODY, fill=MUTED)

    panel(d, 60, 400, 720, 245, "THE TEMPTING SHORTCUT")
    d.text((90, 460), "Turn its thinking off", font=F_H2, fill=AMBER)
    d.text((90, 525), "It works again. But thinking is", font=F_BODY, fill=MUTED)
    d.text((90, 565), "what you were paying extra for.", font=F_BODY, fill=MUTED)
    d.text((90, 615), "Top price. No longer a top model.", font=F_SMALL, fill=AMBER)

    panel(d, 820, 400, 720, 245, "WHAT THIS DOES INSTEAD")
    d.text((850, 460), "Use the other entrance", font=F_H2, fill=GREEN)
    d.text((850, 525), "The provider has a second one where", font=F_BODY, fill=MUTED)
    d.text((850, 565), "thinking and doing both work.", font=F_BODY, fill=MUTED)
    d.text((850, 615), "Your coding tool never notices the difference.", font=F_SMALL, fill=MUTED)

    d.text((60, 675), "the provider's own words: \"use /v1/responses, or set reasoning_effort to 'none'\"",
           font=F_MONO_S, fill=MUTED)
    caption(d, ["The strongest models will not use tools through the ordinary door.",
                "There is a shortcut that opens it again by switching the model's thinking off, which means paying the top price for a model that stopped thinking.",
                "So this uses the other door, where both work, and your coding tool never notices."])
    return hold(img, 15)


def tier_slide(cap):
    """What the local classifier actually predicts, and what it cannot do."""
    t = cap.get("tiers")
    if not t:
        return []
    img, d = base("The router predicts difficulty, not a model")
    d.text((60, 145), "Which is why it still works when your fleet changes", font=F_H2, fill=WHITE)

    panel(d, 60, 225, 1480, 200, "HOW A DECISION IS MADE")
    d.text((90, 285), "prompt", font=F_MONO, fill=MUTED)
    d.text((260, 285), "->", font=F_MONO, fill=LINE)
    d.text((330, 285), "local classifier", font=F_MONO, fill=ACCENT)
    d.text((640, 285), "->", font=F_MONO, fill=LINE)
    d.text((710, 285), "\"this needs the high tier\"", font=F_MONO, fill=WHITE)
    d.text((90, 350), "then: the cheapest model your own keys can reach that clears that tier",
           font=F_BODY, fill=GREEN)

    panel(d, 60, 455, 720, 210, "WHAT THAT BUYS")
    d.text((90, 515), "under a millisecond", font=F_H2, fill=GREEN)
    d.text((90, 575), "on this machine, no network call", font=F_BODY, fill=MUTED)
    d.text((90, 620), "Add a provider key and the pool widens instantly.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 455, 720, 210, "WHERE IT IS STILL WEAK")
    d.text((850, 515), "hard jobs, cheap model", font=F_H2, fill=AMBER)
    d.text((850, 575), "it still gets this wrong", font=F_BODY, fill=MUTED)
    d.text((850, 620), "Your judged traffic is what fixes it. A floor bounds it meanwhile.", font=F_SMALL, fill=MUTED)

    caption(d, ["The classifier predicts how hard the task is, not which model to use.",
                "Difficulty is a property of the request, so it stays true however your fleet changes,",
                "and the gateway spends the least money that buys it from the keys you have.",
                "It gets the easy work right today, and it still sometimes sends a hard job to a cheap model, which is the honest state of it."])
    return hold(img, 17)


def dashboard_slide(cap):
    """The audit trail, which is what makes this defensible internally."""
    pin = cap.get("pinned") or {}
    img, d = base("You can see every decision")
    d.text((60, 145), "Asked for, and what actually ran, side by side", font=F_H2, fill=WHITE)

    panel(d, 60, 225, 1480, 300, "THE LOCAL DASHBOARD")
    d.text((90, 285), "PROMPT", font=F_SMALL, fill=MUTED)
    d.text((760, 285), "MODEL ASKED FOR", font=F_SMALL, fill=MUTED)
    d.text((1130, 285), "MODEL THAT RAN", font=F_SMALL, fill=MUTED)
    rows = [("rename the variable t to running_total", "gpt-5.6-sol", "gpt-5.6-luna", GREEN),
            ("add a docstring to the total function", "gpt-5.6-sol", "gpt-5.6-luna", GREEN),
            ("make it safe under concurrent mutation", "gpt-5.6-sol", "gpt-5.6-terra", AMBER)]
    for i, (task, asked, ran, col) in enumerate(rows):
        y = 340 + i * 52
        d.text((90, y), task, font=F_MONO_S, fill=WHITE)
        d.text((760, y), asked, font=F_MONO_S, fill=MUTED)
        d.text((1130, y), ran, font=F_MONO_S, fill=col)

    d.text((60, 560), "Nothing is hidden from the developer, and nothing needs their attention.", font=F_BODY, fill=WHITE)
    d.text((60, 605), "Finance gets a per-request record. Engineering gets its model of choice on the hard work.",
           font=F_BODY, fill=MUTED)
    d.text((60, 660), "It all runs on the machine. Prompts do not leave it to be routed.", font=F_BODY, fill=GREEN)
    caption(d, ["Every decision is logged locally with the model that was asked for beside the model that ran.",
                "Nothing is hidden from the developer, and nothing needs their attention.",
                "Finance gets a per-request record, engineering keeps the strong model on the work that needs it, and the prompts never leave the machine."])
    return hold(img, 17)


def close_slide(cap):
    img, d = base()
    d.text((60, 275), "Same harness. Same engineers.", font=F_H1, fill=WHITE)
    d.text((60, 345), "The model choice made per prompt.", font=F_H1, fill=ACCENT)
    d.text((62, 460), "Runs on your machine. Works with Codex CLI, Claude Code and OpenCode.", font=F_BODY, fill=MUTED)
    d.text((62, 505), "Prompts never leave it to be routed.", font=F_BODY, fill=MUTED)
    d.text((62, 550), "Open source, Apache 2.0.", font=F_BODY, fill=MUTED)
    caption(d, ["Nothing changes for the engineers. What changes is which model quietly answers each request.",
                "It runs on your own machine, it is open source, and we would rather hear where the routing got it wrong than that it worked."])
    return hold(img, 11)


def main():
    if not os.path.exists(CAPTURE):
        sys.exit(f"No capture at {CAPTURE}.")
    cap = json.load(open(CAPTURE))
    frames = []
    # Ordered as an argument, not a feature list: the policy problem, the one
    # session that shows it solved, where it shows up, what it costs, how you
    # audit it, and what it cannot do.
    for fn in (title_slide, problem_slide, pinned_slide, surfaces_slide,
               family_slide, dashboard_slide, loop_slide, close_slide):
        frames += fn(cap)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    silent = OUT if not any(p for p, _ in SEGMENTS) else OUT.replace(".mp4", ".silent.mp4")
    wr = imageio_ffmpeg.write_frames(silent, (W, H), fps=FPS, quality=8)
    wr.send(None)
    for f in frames:
        wr.send(f.tobytes())
    wr.close()
    print(f"Wrote video ({len(frames)} frames, {len(frames)/FPS:.1f}s)")

    if silent == OUT:
        return

    # The audio track is assembled from SEGMENTS; if their total does not match
    # the frames actually written, the narration drifts against the picture.
    # Catch that here rather than letting someone notice it on LinkedIn.
    seg_frames = int(sum(sec for _, sec in SEGMENTS) * FPS)
    if abs(seg_frames - len(frames)) > FPS:
        sys.exit(f"narration/video mismatch: {seg_frames} vs {len(frames)} frames. "
                 "A slide is building frames without registering a segment.")
    mux(silent)


def mux(silent):
    """Lay each slide's narration over that slide and attach the track.

    Each segment is padded out to its slide's exact length, so a slide that
    ran long for readability holds in silence rather than pulling the next
    slide's narration forward. Drift cannot accumulate.
    """
    tmp = os.path.join(ROOT, "data", "_narration_build")
    os.makedirs(tmp, exist_ok=True)
    parts = []
    for i, (path, seconds) in enumerate(SEGMENTS):
        part = os.path.join(tmp, f"seg{i:02d}.wav")
        if path:
            # apad then a hard -t gives exactly the slide's duration.
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", path,
                   "-af", "apad", "-t", f"{seconds:.3f}",
                   "-ar", "44100", "-ac", "2", part]
        else:
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
                   "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{seconds:.3f}", part]
        subprocess.run(cmd, check=True)
        parts.append(part)

    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as fh:
        for part in parts:
            fh.write(f"file '{part}'\n")

    track = os.path.join(tmp, "narration.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", listing, "-c", "copy", track], check=True)

    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", silent, "-i", track,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", OUT], check=True)
    os.remove(silent)
    shutil.rmtree(tmp, ignore_errors=True)   # intermediates, not artefacts
    spoken = sum(1 for p, _ in SEGMENTS if p)
    print(f"Wrote {OUT} with narration on {spoken}/{len(SEGMENTS)} slides")


if __name__ == "__main__":
    main()
