"""Render a demo video from a real captured run.

Every figure shown here comes from `data/demo_capture.json`, which is written
by an actual request through the gateway. Nothing is typed in by hand: if the
run routed somewhere unexpected or scored badly, the video says so. A demo
that cannot be reproduced from the tool it demonstrates is a slide deck.
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE = os.path.join(PROJECT_ROOT, "data", "demo_capture.json")
OUT_MP4 = os.path.join(PROJECT_ROOT, "docs", "wormhole_terminal_demo.mp4")

W, H = 1280, 720
BG, FG = (13, 17, 23), (201, 209, 217)
GREEN, CYAN, DIM, YELLOW = (63, 185, 80), (121, 192, 255), (110, 118, 129), (210, 168, 78)
FPS = 30


def _font(size, bold=False):
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


MONO, MONO_B = _font(19), _font(19)
TITLE = _font(27)


def frame(lines, title="WormHole — routed agentic run"):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=(22, 27, 34))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([18 + i * 22, 16, 30 + i * 22, 28], fill=c)
    d.text((100, 12), title, font=TITLE, fill=DIM)
    y = 70
    for text, colour in lines:
        d.text((28, y), text, font=MONO, fill=colour)
        y += 27
    return img


def typed(lines, prefix, text, colour):
    """Yield frames revealing `text` a few characters at a time."""
    out = []
    step = max(1, len(text) // 22)
    for i in range(0, len(text) + step, step):
        out.append(frame(lines + [(prefix + text[:i] + "█", colour)]))
    return out


def build(cap):
    frames, lines = [], []

    def hold(n=1):
        frames.extend([frame(lines)] * n)

    lines.append(("$ ", FG))
    lines.pop()
    frames += typed(lines, "$ ", "codex exec -s workspace-write \"Create fib.py that prints the first 10 Fibonacci numbers, then run it\"", FG)
    lines.append(("$ codex exec -s workspace-write \"Create fib.py ... then run it\"", FG))
    hold(20)

    lines.append(("", FG))
    lines.append(("  WormHole gateway", CYAN))
    hold(10)
    lines.append((f"    router   -> {cap['model']}", GREEN))
    hold(18)
    enh = cap["enhancer"]
    lines.append((f"    enhancer -> {enh}" + ("  (model already in a strong tier)" if enh == "bypassed" else ""), GREEN if enh != "bypassed" else DIM))
    hold(18)
    lines.append((f"    tokens   -> {cap['ptok']} in / {cap['ctok']} out  (provider-reported)", DIM))
    hold(18)
    lines.append((f"    cost     -> ${cap['actual']:.6f}   vs ${cap['baseline']:.6f} at the GPT-4o baseline", DIM))
    hold(30)

    lines.append(("", FG))
    lines.append(("  Files written to the workspace:", CYAN))
    for f in cap["files"]:
        lines.append((f"    {f}", GREEN))
    hold(25)

    for src_line in cap["fib"].strip().split("\n")[:8]:
        lines.append((f"    {src_line[:88]}", FG))
    hold(45)

    lines.append(("", FG))
    lines.append(("  The completion is scored, and the score trains the router.", CYAN))
    hold(12)
    lines.append((f"    judge score -> {cap['judge']}/10", YELLOW))
    hold(12)
    lines.append((f"    retrain     -> {cap['feedback']} judged prompts fed back into the local router", GREEN))
    hold(60)
    return frames


def main():
    if not os.path.exists(CAPTURE):
        sys.exit(f"No capture at {CAPTURE}. Run a request through the gateway first.")
    cap = json.load(open(CAPTURE))
    frames = build(cap)
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    writer = imageio_ffmpeg.write_frames(OUT_MP4, (W, H), fps=FPS, quality=8)
    writer.send(None)
    for f in frames:
        writer.send(f.tobytes())
    writer.close()
    print(f"Wrote {OUT_MP4} ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
