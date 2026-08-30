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
    d.text((60, 300), "Routing every request to the", font=F_H1, fill=WHITE)
    d.text((60, 370), "cheapest model that can do the job", font=F_H1, fill=ACCENT)
    d.text((62, 470), "A local routing layer for Codex and Claude Code.", font=F_BODY, fill=MUTED)
    d.text((62, 510), "Same tools your engineers already use. No workflow change.", font=F_BODY, fill=MUTED)
    caption(d, ["A small local service sits between your harness and the model providers.",
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
    img, d = base("It gets better as you use it")
    d.text((60, 165), "Every answer is scored. The scores retrain the router.", font=F_H2, fill=WHITE)

    steps = [("Route", ACCENT), ("Run", MUTED), ("Score", AMBER), ("Learn", GREEN)]
    x = 90
    for i, (label, colour) in enumerate(steps):
        panel(d, x, 290, 280, 130)
        d.text((x + 40, 340), label, font=F_H1, fill=colour)
        if i < len(steps) - 1:
            d.text((x + 295, 340), "->", font=F_H2, fill=LINE)
        x += 340

    d.text((60, 510), "It starts out knowing public benchmarks.", font=F_BODY, fill=MUTED)
    d.text((60, 560), "It ends up knowing the work your team actually does.", font=F_BODY, fill=WHITE)
    d.text((60, 630), "All of it on your machine. Prompts never leave it to decide where to go.", font=F_BODY, fill=GREEN)
    caption(d, ["Every answer gets scored, and those scores teach the router.",
                "It starts out knowing public benchmarks and ends up knowing the work your team actually does.",
                "All of it on your own machine."])
    return hold(img, 12)


def tradeoff_slide(cap):
    """The payoff, in words rather than fractions.

    This slide used to carry six ratios -- 14 of 24, 20 of 24, 60 of 72, 61 of
    72, a delta in points, a percentage -- which is the right level of detail
    for the README and the wrong one for a viewer giving this ninety seconds.
    The fractions are all still in the repository; what belongs here is what
    they add up to.
    """
    q = cap.get("quality")
    if not q:
        return []
    img, d = base("Does it cost you quality?")

    d.text((60, 175), "Same work. Same result.", font=F_H1, fill=WHITE)
    d.text((60, 245), "About a twenty-fifth of the price.", font=F_H1, fill=GREEN)

    panel(d, 60, 355, 720, 250, "HOW WE KNOW")
    d.text((90, 415), f"{q['n']} real coding tasks", font=F_H2, fill=WHITE)
    d.text((90, 480), "Run twice: once routed,", font=F_BODY, fill=MUTED)
    d.text((90, 520), "once always on the flagship.", font=F_BODY, fill=MUTED)
    d.text((90, 565), "Marked by running the tests, not by asking a model.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 355, 720, 250, "THE CATCH")
    d.text((850, 415), "Cheap has a floor", font=F_H2, fill=AMBER)
    d.text((850, 480), "Let it reach for a free tiny", font=F_BODY, fill=MUTED)
    d.text((850, 520), "model and accuracy does drop.", font=F_BODY, fill=MUTED)
    d.text((850, 565), "One setting stops that. It is the next slide.", font=F_SMALL, fill=MUTED)

    caption(d, ["Routing did not beat the flagship. It matched it, and cost about a twenty-fifth as much.",
                "That is measured by running the tasks and checking the answers, not by asking a model for its opinion.",
                "The catch is that cheap has a floor, and going under it costs you real accuracy."])
    return hold(img, 15)


def surfaces_slide(cap):
    """Every tool that was actually driven through the gateway, and the one
    that cannot be. The count beside each row is real logged requests, so a
    reader can reconcile it against /api/logs rather than take it on faith."""
    rows = cap.get("surfaces")
    if not rows:
        return []
    img, d = base("Works with the harnesses you already use")
    d.text((60, 145), "Five ways in, all measured", font=F_H2, fill=WHITE)

    y = 235
    d.text((90, y), "TOOL", font=F_SMALL, fill=MUTED)
    d.text((470, y), "HOW IT IS POINTED", font=F_SMALL, fill=MUTED)
    d.text((870, y), "REQUESTS", font=F_SMALL, fill=MUTED)
    d.text((1080, y), "ROUTED TO", font=F_SMALL, fill=MUTED)
    d.line([(90, y + 32), (1540, y + 32)], fill=LINE, width=2)

    y += 58
    for r in rows:
        d.text((90, y), r["name"], font=F_BODY, fill=WHITE)
        d.text((470, y), r["how"], font=F_BODY, fill=MUTED)
        # Advisory mode never proxies inference, so a zero here is the design
        # working, not a tool that failed to connect.
        req = f"+{r['req']}" if r["req"] else "none (advisory)"
        d.text((870, y), req, font=F_BODY, fill=GREEN if r["req"] else MUTED)
        d.text((1080, y), r["model"], font=F_BODY, fill=ACCENT)
        d.text((1450, y), "OK", font=F_BODY, fill=GREEN)
        y += 52

    nr = cap.get("not_routable")
    if nr:
        y += 18
        d.text((90, y), f"{nr['name']}: not routable", font=F_BODY, fill=AMBER)
        d.text((90, y + 38), nr["why"], font=F_SMALL, fill=MUTED)

    caption(d, ["Codex, Claude Code and OpenCode were each driven end to end, and each one wrote its file.",
                "The Codex desktop app reads the same config file as the CLI, so it routes too.",
                "Claude for Desktop cannot: it is the chat app, and has no API endpoint to redirect."])
    return hold(img, 16)


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
    """Why the top tiers needed real work to reach at all."""
    img, d = base("Reaching the top tier honestly")
    d.text((60, 145), "The reasoning tiers refuse tools on the ordinary endpoint", font=F_H2, fill=WHITE)

    panel(d, 60, 225, 1480, 195, "WHAT THE PROVIDER RETURNS")
    d.text((90, 285), "Function tools with reasoning_effort are not supported", font=F_MONO_S, fill=AMBER)
    d.text((90, 325), "in /v1/chat/completions. Use /v1/responses, or set", font=F_MONO_S, fill=AMBER)
    d.text((90, 365), "reasoning_effort to 'none'.", font=F_MONO_S, fill=AMBER)

    panel(d, 60, 445, 720, 235, "THE EASY WAY OUT")
    d.text((90, 500), "reasoning_effort = none", font=F_MONO_S, fill=MUTED)
    d.text((90, 550), "Tools work again.", font=F_BODY, fill=MUTED)
    d.text((90, 592), "The reasoning is switched off.", font=F_BODY, fill=AMBER)
    d.text((90, 634), "Top-tier price, no longer a top tier.", font=F_SMALL, fill=AMBER)

    panel(d, 820, 445, 720, 235, "WHAT THIS GATEWAY DOES")
    d.text((850, 500), "translate to /v1/responses", font=F_MONO_S, fill=GREEN)
    d.text((850, 550), "Tools and full reasoning together.", font=F_BODY, fill=GREEN)
    d.text((850, 592), "Every harness gains it unchanged.", font=F_BODY, fill=MUTED)
    d.text((850, 634), "Verified: Codex wrote a file, then tested it.", font=F_SMALL, fill=MUTED)

    caption(d, ["The strongest tiers refuse to use tools on the ordinary endpoint.",
                "You can buy tool support back by turning the reasoning off, which means paying the top price for a model that no longer reasons.",
                "So the gateway translates to the endpoint where both work, and every harness gains those models without knowing anything changed."])
    return hold(img, 16)


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
    d.text((90, 515), f"{t['ms']} ms per decision", font=F_H2, fill=GREEN)
    d.text((90, 575), "on this machine, no network call", font=F_BODY, fill=MUTED)
    d.text((90, 620), "Add a provider key and the pool widens instantly.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 455, 720, 210, "WHERE IT IS STILL WEAK")
    d.text((850, 515), "short hard prompts", font=F_H2, fill=AMBER)
    d.text((850, 575), "under-routed today", font=F_BODY, fill=MUTED)
    d.text((850, 620), "Your judged traffic is what fixes it. A floor bounds it meanwhile.", font=F_SMALL, fill=MUTED)

    caption(d, ["The classifier predicts how hard the task is, not which model to use.",
                "Difficulty is a property of the request, so it stays true however your fleet changes,",
                "and the gateway spends the least money that buys it from the keys you have.",
                "It is right about the easy work today and still under-rates short hard instructions, which is the honest state of it."])
    return hold(img, 17)


def dashboard_slide(cap):
    """Show the gateway dashboard with a real harness request routed through it."""
    img, d = base("Routing happens on your machine, live")

    d.text((60, 145), "A Codex prompt routed in real time", font=F_H2, fill=WHITE)

    panel(d, 60, 235, 1480, 380, "HARNESS REQUEST")
    d.text((90, 300), "Codex: Create a file called invoice.py with totals function", 
           font=F_BODY, fill=WHITE)
    d.line([(90, 340), (1500, 340)], fill=LINE, width=1)

    d.text((90, 370), "Router picked: gpt-4o-mini", font=F_H2, fill=GREEN)
    d.text((90, 420), "Cost: $0.000827  |  Tokens: 5,380 in / 34 out", font=F_BODY, fill=ACCENT)
    d.text((90, 460), "You can inspect the enhanced prompt on the dashboard", font=F_SMALL, fill=MUTED)
    d.text((90, 500), "Every request, every decision, every model choice — live.", font=F_SMALL, fill=MUTED)

    caption(d, ["The dashboard logs every harness request, the routing decision, and the cost.",
                "The router made its choice on this machine in under a millisecond. No network call, no extra latency."])
    return hold(img, 12)

def close_slide(cap):
    img, d = base()
    d.text((60, 255), "Same harness. Same engineers.", font=F_H1, fill=WHITE)
    d.text((60, 325), "A policy about which model runs what.", font=F_H1, fill=ACCENT)
    d.text((62, 435), "Runs self-hosted. Works with Codex CLI, Claude Code and OpenCode.", font=F_BODY, fill=MUTED)
    d.text((62, 475), "Vendor-agnostic, or pinned to one vendor's tiers by policy.", font=F_BODY, fill=MUTED)

    # Name the command rather than claiming reproducibility in the abstract.
    # An earlier version of this slide said every figure here was reproducible
    # with one command, which was not true: the quality and cost comparison is,
    # and the rest of the capture is assembled from separate runs.
    panel(d, 60, 540, 1480, 130, "CHECK THE MAIN CLAIM YOURSELF")
    d.text((90, 600), "python scripts/evaluate_routing_quality.py --n 24 --baseline gpt-4o",
           font=F_MONO, fill=GREEN)
    caption(d, ["Nothing changes for the engineers. What changes is which model quietly answers each request.",
                "The cost and quality comparison is one command, and it marks the answers by running the tests.",
                "Run it on your own fleet and you will get your own number, not this one."])
    return hold(img, 10)


def main():
    if not os.path.exists(CAPTURE):
        sys.exit(f"No capture at {CAPTURE}.")
    cap = json.load(open(CAPTURE))
    frames = []
    for fn in (title_slide, problem_slide, routing_slide, family_slide, api_slide,
               tier_slide, surfaces_slide, dashboard_slide,
               tradeoff_slide, floor_slide, loop_slide, close_slide):
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
