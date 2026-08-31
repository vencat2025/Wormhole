"""Cut the README's inline hook out of the narrated demo video.

GitHub will not embed an mp4: its markdown sanitiser drops <video>, and a
committed .mp4 renders as a link rather than a player. A GIF plays inline
anywhere, so the README's hook is a GIF and the narrated video sits under it as
a link.

That trade costs the audio, and the narration now carries a lot of the meaning,
so this deliberately does not try to compress three minutes into a loop. It
takes two beats that stand up silently:

  1. the routing table building a row at a time, which shows the whole idea
     without a word: five tasks, and not all of them need the expensive model
  2. the result, which is the reason to keep watching

Both are cut from the real video, so this cannot drift from what the video
says -- if a slide changes, re-run this and the hook changes with it.

Usage:
  python scripts/make_highlight_gif.py
"""

import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SOURCE = os.path.join(ROOT, "docs", "wormhole_exec_demo.mp4")
OUT = os.path.join(ROOT, "docs", "wormhole_highlight.gif")
WORK = os.path.join(ROOT, "data", "_gif_build")

# (start seconds, duration) in the narrated video. Slide boundaries move when
# narration is re-recorded, so check these against a fresh render rather than
# assuming they still point at the right moment.
BEATS = [
    (12.0, 5.0),    # the policy that does not work, and why
    (41.0, 7.5),    # pinned to sol, and what actually ran
]

WIDTH = 820
FPS = 10


def run(args):
    subprocess.run(args, check=True)


def main() -> int:
    if not os.path.exists(SOURCE):
        sys.exit(f"No video at {SOURCE}. Run scripts/record_exec_demo.py first.")
    os.makedirs(WORK, exist_ok=True)

    parts = []
    for i, (start, length) in enumerate(BEATS):
        part = os.path.join(WORK, f"beat{i}.mp4")
        run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{length}",
             "-i", SOURCE, "-an", "-vf", f"scale={WIDTH}:-2", part])
        parts.append(part)

    listing = os.path.join(WORK, "list.txt")
    with open(listing, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    joined = os.path.join(WORK, "joined.mp4")
    run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listing, "-c", "copy", joined])

    # A generated palette rather than the default web one: these slides are
    # flat blocks of colour with fine mono type over them, and the default
    # palette bands the panels and muddies the text.
    palette = os.path.join(WORK, "palette.png")
    run([FFMPEG, "-y", "-loglevel", "error", "-i", joined,
         "-vf", f"fps={FPS},scale={WIDTH}:-1:flags=lanczos,palettegen=max_colors=128", palette])
    run([FFMPEG, "-y", "-loglevel", "error", "-i", joined, "-i", palette,
         "-lavfi", f"fps={FPS},scale={WIDTH}:-1:flags=lanczos[x];"
                   "[x][1:v]paletteuse=dither=bayer:bayer_scale=3", OUT])

    size = os.path.getsize(OUT)
    print(f"Wrote {OUT} ({size / 1024:.0f} KB)")
    if size > 5 * 1024 * 1024:
        print("Warning: over 5 MB. Shorten a beat or drop the frame rate; a "
              "README hook that takes a moment to load is not a hook.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
