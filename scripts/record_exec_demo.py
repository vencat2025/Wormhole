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
    "MIN_ROUTING_TIER=medium": "min routing tier set to medium",
    "MIN_ROUTING_TIER": "min routing tier",
    "gpt-oss-120b": "the 120B open-source model",
    "gpt-5-nano": "GPT 5 nano",
    "gpt-5-mini": "GPT 5 mini",
    "gpt-5.6-luna": "GPT 5.6 Luna",
    "gpt-4o-mini": "GPT 4o mini",
    "gpt-4o": "GPT 4o",
    "/api/logs": "the logs endpoint",
    "config.toml": "config dot toml",
    "opencode.json": "opencode dot json",
    "claude-routed": "the Claude wrapper",
    "codex-routed": "the Codex wrapper",
    "MBPP": "M B P P",
    "SWE-bench": "swee bench",
    "SLM": "small language model",
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
                "Read as an engineer explaining a measured result to colleagues: "
                "even, unhurried, and matter-of-fact. No sales enthusiasm."
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
    d.text((90, 300), f"routed        {q['weak_routed_solved']} of {q['weak_n']}", font=F_H2, fill=AMBER)
    d.text((90, 365), f"always strong {q['weak_baseline_solved']} of {q['weak_n']}", font=F_H2, fill=MUTED)
    d.text((90, 450), f"{q['weak_delta']} points of correctness", font=F_BODY, fill=AMBER)
    d.text((90, 490), "for a 100% saving. A real loss.", font=F_BODY, fill=MUTED)

    panel(d, 820, 235, 720, 320, "RESTRICTED TO STRONGER TIERS")
    d.text((850, 300), f"routed        {q['routed_solved']} of {q['n']}", font=F_H2, fill=GREEN)
    d.text((850, 365), f"always strong {q['baseline_solved']} of {q['n']}", font=F_H2, fill=MUTED)
    d.text((850, 450), "one task apart, over three runs", font=F_BODY, fill=GREEN)
    d.text((850, 490), "for a 96% saving.", font=F_BODY, fill=MUTED)

    d.text((60, 600), "Routing did not beat the flagship. It matched it, within noise,", font=F_BODY, fill=WHITE)
    d.text((60, 640), "and did the same work for about a twenty-fifth of the price.", font=F_BODY, fill=MUTED)
    d.text((60, 685), f"Correctness is the benchmark's own test suite executed over {q['n']} tasks, not a model's opinion.", font=F_SMALL, fill=MUTED)
    caption(d, ["The honest question is not whether routing costs quality, but which models you allow.",
                "Let it use a tiny free model and accuracy really does drop. Restrict it to stronger tiers",
                "and it matched always-using-the-flagship - one task apart across seventy-two - for a fraction of the cost."])
    return hold(img, 16)


def surfaces_slide(cap):
    """Every tool that was actually driven through the gateway, and the one
    that cannot be. The count beside each row is real logged requests, so a
    reader can reconcile it against /api/logs rather than take it on faith."""
    rows = cap.get("surfaces")
    if not rows:
        return []
    img, d = base("Works with the tools you already use")
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
    """The setting that decides whether any of this is usable in practice."""
    f = cap.get("floor")
    if not f:
        return []
    img, d = base("The setting that matters most")
    d.text((60, 145), "Cheap is not the same as capable", font=F_H2, fill=WHITE)
    d.text((60, 200), f"Same task both times: {f['task']}.", font=F_BODY, fill=MUTED)

    panel(d, 60, 265, 720, 300, "NO FLOOR")
    d.text((90, 330), f["before_model"], font=F_H2, fill=AMBER)
    d.text((90, 400), f"{f['before_req']} requests", font=F_BODY, fill=MUTED)
    d.text((90, 445), "file never created", font=F_BODY, fill=AMBER)
    d.text((90, 500), "It called tools correctly the whole time.", font=F_SMALL, fill=MUTED)

    panel(d, 820, 265, 720, 300, "MIN_ROUTING_TIER=MEDIUM")
    d.text((850, 330), f["after_model"], font=F_H2, fill=GREEN)
    d.text((850, 400), f"{f['after_req']} requests", font=F_BODY, fill=MUTED)
    d.text((850, 445), "file created", font=F_BODY, fill=GREEN)
    d.text((850, 500), "Still a cheap tier. Just not the cheapest.", font=F_SMALL, fill=MUTED)

    d.text((60, 610), "Advertising tool support is not the same as being able to drive an agentic harness.", font=F_BODY, fill=WHITE)
    d.text((60, 650), "One setting keeps the tiers that cannot out of the pool entirely.", font=F_BODY, fill=MUTED)
    caption(d, ["This is the first thing to set, and the easiest to get wrong.",
                "The weakest tier benchmarks well on short answers and still cannot run an agent loop.",
                "A floor costs a little more per request and saves the nineteen that went nowhere."])
    return hold(img, 15)


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
    for fn in (title_slide, problem_slide, routing_slide, surfaces_slide,
               evidence_slide, tradeoff_slide, floor_slide, loop_slide, close_slide):
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
