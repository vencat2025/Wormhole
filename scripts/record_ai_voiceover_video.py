import os
import sys
import subprocess
import imageio
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_MP4 = os.path.join(PROJECT_ROOT, "docs", "wormhole_ai_demo.mp4")
TEMP_AUDIO = os.path.join(PROJECT_ROOT, "docs", "narration.wav")
TEMP_VIDEO = os.path.join(PROJECT_ROOT, "docs", "temp_video.mp4")

WIDTH, HEIGHT = 1000, 600
BACKGROUND_COLOR = (11, 15, 25)
TEXT_COLOR = (249, 250, 251)
SUBTEXT_COLOR = (156, 163, 175)
GREEN_COLOR = (16, 185, 129)
ACCENT_COLOR = (99, 102, 241)
PANEL_COLOR = (17, 24, 39)
BORDER_COLOR = (31, 41, 55)

NARRATION_SCRIPT = (
    "Welcome to WormHole, the 100% enterprise production ready AI inference gateway designed to cut API costs by over 90 percent while preserving output quality. "
    "WormHole features sub 2 millisecond local SLMs, selective prompt enhancement, Bearer key authentication, SSE token streaming, and automatic circuit breaker failovers. "
    "Whether you are integrating standard OpenAI SDKs, custom harnesses, or tools like Claude Code and Cursor, simply point your base URL to WormHole to instantly cut your enterprise LLM spend."
)

def generate_voiceover_audio():
    print("🎙️ Synthesizing Enterprise AI Voiceover Audio (Daniel Natural Cadence 44.1kHz)...")
    os.makedirs(os.path.dirname(TEMP_AUDIO), exist_ok=True)
    cmd = ["say", "-v", "Daniel", "-r", "165", "-o", TEMP_AUDIO, "--data-format=LEI16@44100", NARRATION_SCRIPT]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        cmd = ["say", "-v", "Samantha", "-r", "165", "-o", TEMP_AUDIO, "--data-format=LEI16@44100", NARRATION_SCRIPT]
        subprocess.run(cmd, check=True)
    print("✅ Enterprise Voiceover audio generated successfully.")

def draw_header(draw, title="⚡ WORMHOLE - ENTERPRISE PRODUCTION READY AI GATEWAY"):
    draw.rectangle([0, 0, WIDTH, 60], fill=PANEL_COLOR)
    draw.line([0, 60, WIDTH, 60], fill=BORDER_COLOR, width=2)
    draw.text((20, 18), title, fill=TEXT_COLOR, font_size=20)
    draw.rectangle([WIDTH - 210, 15, WIDTH - 20, 45], fill=GREEN_COLOR)
    draw.text((WIDTH - 200, 22), "ENTERPRISE HARDENED", fill=(255, 255, 255), font_size=12)

def create_frame_title():
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    
    draw.text((50, 130), "WormHole Enterprise AI Inference Gateway", fill=(255, 255, 255), font_size=32)
    draw.text((50, 175), "Sub-2ms Local SLMs | 90%+ Cost Savings | 100% Production Ready", fill=SUBTEXT_COLOR, font_size=18)
    
    boxes = [
        ("🔑 Bearer Auth & Security", "wh_live_... tenant key validation"),
        ("⚡ SSE Token Streaming", "stream=True real-time protocol"),
        ("⚡ Model 2: Router SLM", "Sub-2ms local benchmark routing"),
        ("🛡️ Circuit Breaker Failover", "Auto 5xx outage failover handling")
    ]
    
    for idx, (b_title, b_sub) in enumerate(boxes):
        x = 50 + (idx % 2) * 440
        y = 240 + (idx // 2) * 130
        draw.rectangle([x, y, x + 410, y + 100], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, y + 20), b_title, fill=ACCENT_COLOR, font_size=18)
        draw.text((x + 20, y + 55), b_sub, fill=SUBTEXT_COLOR, font_size=14)
        
    return img

def create_frame_harness():
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ ENTERPRISE HARNESS INTEGRATION (CLAUDE CODE, CODEX, CURSOR)")
    
    draw.rectangle([40, 90, WIDTH - 40, HEIGHT - 40], fill=(15, 23, 42), outline=BORDER_COLOR, width=2)
    draw.rectangle([40, 90, WIDTH - 40, 125], fill=(30, 41, 59))
    draw.ellipse([55, 103, 67, 115], fill=(239, 68, 68))
    draw.ellipse([75, 103, 87, 115], fill=(245, 158, 11))
    draw.ellipse([95, 103, 107, 115], fill=(34, 197, 94))
    draw.text((120, 100), "bash - enterprise developer workstation", fill=SUBTEXT_COLOR, font_size=14)
    
    lines = [
        ("🤖 CLAUDE CODE CLI INTEGRATION:", ACCENT_COLOR),
        ("   export ANTHROPIC_BASE_URL=\"http://127.0.0.1:8000/v1\"", TEXT_COLOR),
        ("   export ANTHROPIC_API_KEY=\"wh_live_demo123456789\"", TEXT_COLOR),
        (" ", TEXT_COLOR),
        ("💻 CURSOR / VS CODE / OPENAI CODEX HARNESS INTEGRATION:", ACCENT_COLOR),
        ("   export OPENAI_BASE_URL=\"http://127.0.0.1:8000/v1\"", TEXT_COLOR),
        ("   export OPENAI_API_KEY=\"wh_live_demo123456789\"", TEXT_COLOR),
        (" ", TEXT_COLOR),
        ("🎉 Result: 100% of Developer Coding Traffic Routed & Optimized (90%+ Cost Cut!)", GREEN_COLOR)
    ]
    
    y_pos = 145
    for text, color in lines:
        draw.text((60, y_pos), text, fill=color, font_size=15)
        y_pos += 30
        
    return img

def render_video():
    generate_voiceover_audio()
    import soundfile as sf
    data, samplerate = sf.read(TEMP_AUDIO)
    audio_duration = float(len(data)) / samplerate
    
    fps = 10
    total_frames = int(audio_duration * fps)
    
    print(f"🎥 Rendering Enterprise HD Video Frames ({total_frames} frames @ {fps} FPS)...")
    writer = imageio.get_writer(TEMP_VIDEO, fps=fps)
    
    title_f = create_frame_title()
    harness_f = create_frame_harness()
    
    f_per_scene = total_frames // 2
    import numpy as np

    for _ in range(f_per_scene):
        writer.append_data(np.array(title_f))
    for _ in range(total_frames - f_per_scene):
        writer.append_data(np.array(harness_f))
        
    writer.close()
    
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("🎬 Merging Enterprise AI Voiceover Audio with Video into MP4...")
    merge_cmd = [
        ffmpeg_exe, "-y",
        "-i", TEMP_VIDEO,
        "-i", TEMP_AUDIO,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        OUTPUT_MP4
    ]
    subprocess.run(merge_cmd, check=True)
    
    print(f"🎉 Enterprise AI Voiceover Video successfully updated at: {OUTPUT_MP4}")

if __name__ == "__main__":
    render_video()
