import os
import sys
import subprocess
import imageio
from PIL import Image, ImageDraw

OUTPUT_MP4 = "/Users/venkat/Documents/AI/WormHole/docs/wormhole_ai_demo.mp4"
TEMP_AUDIO = "/Users/venkat/Documents/AI/WormHole/docs/narration.wav"
TEMP_VIDEO = "/Users/venkat/Documents/AI/WormHole/docs/temp_video.mp4"

WIDTH, HEIGHT = 1000, 600
BACKGROUND_COLOR = (11, 15, 25)
TEXT_COLOR = (249, 250, 251)
SUBTEXT_COLOR = (156, 163, 175)
GREEN_COLOR = (16, 185, 129)
ACCENT_COLOR = (99, 102, 241)
PANEL_COLOR = (17, 24, 39)
BORDER_COLOR = (31, 41, 55)

NARRATION_SCRIPT = (
    "Welcome to WormHole, the enterprise AI inference middleware layer designed to cut API costs by over 90 percent, while preserving completion quality. "
    "Unlike basic routers that forward raw prompts to budget models, WormHole uses a dual local model architecture. "
    "Model 1 quality enriches user prompts in under 1 millisecond. "
    "Model 2 evaluates prompt complexity against frontier LLM benchmarks in under 2 milliseconds, and routes tasks to the lowest cost capable model. "
    "As completions deliver, an asynchronous LLM judge rates quality scores from 1 to 10, exporting datasets for continuous SLM retraining. "
    "Here on the live Web Dashboard, you can track real-time cost savings, compare original versus enhanced prompts side by side, and trigger online SLM retraining."
)

def generate_voiceover_audio():
    print("🎙️ Synthesizing High-Fidelity AI Voiceover Audio (44.1kHz Daniel Natural Cadence)...")
    os.makedirs(os.path.dirname(TEMP_AUDIO), exist_ok=True)
    
    # Try high-quality Daniel voice at 170 WPM and 44.1kHz audio
    cmd = ["say", "-v", "Daniel", "-r", "170", "-o", TEMP_AUDIO, "--data-format=LEI16@44100", NARRATION_SCRIPT]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to Samantha at 170 WPM
        cmd = ["say", "-v", "Samantha", "-r", "170", "-o", TEMP_AUDIO, "--data-format=LEI16@44100", NARRATION_SCRIPT]
        subprocess.run(cmd, check=True)
    print("✅ High-Fidelity AI Voiceover audio generated successfully.")

def draw_header(draw, title="⚡ WORMHOLE - ENTERPRISE AI COST REDUCER DEMO"):
    draw.rectangle([0, 0, WIDTH, 60], fill=PANEL_COLOR)
    draw.line([0, 60, WIDTH, 60], fill=BORDER_COLOR, width=2)
    draw.text((20, 18), title, fill=TEXT_COLOR, font_size=20)
    draw.rectangle([WIDTH - 180, 15, WIDTH - 20, 45], fill=ACCENT_COLOR)
    draw.text((WIDTH - 170, 22), "LOCAL SLM ACTIVE", fill=(255, 255, 255), font_size=12)

def create_frame_title():
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    
    draw.text((50, 140), "WormHole AI Inference Gateway", fill=(255, 255, 255), font_size=36)
    draw.text((50, 190), "Enterprise Cost Optimization & Quality Flywheel", fill=SUBTEXT_COLOR, font_size=20)
    
    boxes = [
        ("✨ Model 1: Local Enhancer SLM", "Enriches prompts for quality in <1ms"),
        ("🎯 Model 2: Local Router SLM", "Routes tasks via sub-2ms SLMs (<$0 cost)"),
        ("⚖️ LLM-as-a-Judge", "Auto-evaluates quality on 1.0-10.0 scale"),
        ("🎓 Fine-Tuning Flywheel", "Exports JSONL datasets for local retraining")
    ]
    
    for idx, (b_title, b_sub) in enumerate(boxes):
        x = 50 + (idx % 2) * 440
        y = 260 + (idx // 2) * 120
        draw.rectangle([x, y, x + 410, y + 90], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, y + 18), b_title, fill=ACCENT_COLOR, font_size=18)
        draw.text((x + 20, y + 50), b_sub, fill=SUBTEXT_COLOR, font_size=14)
        
    return img

def create_frame_terminal_case(case_num, prompt_text, model, cost_act, cost_base, pct_saved):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, f"⚡ DEMO CASE {case_num}: REAL-TIME SLM ROUTING")
    
    draw.rectangle([40, 90, WIDTH - 40, HEIGHT - 40], fill=(15, 23, 42), outline=BORDER_COLOR, width=2)
    draw.rectangle([40, 90, WIDTH - 40, 125], fill=(30, 41, 59))
    draw.ellipse([55, 103, 67, 115], fill=(239, 68, 68))
    draw.ellipse([75, 103, 87, 115], fill=(245, 158, 11))
    draw.ellipse([95, 103, 107, 115], fill=(34, 197, 94))
    draw.text((120, 100), "bash - wormhole client_harness", fill=SUBTEXT_COLOR, font_size=14)
    
    lines = [
        (f"📥 INPUT PROMPT: \"{prompt_text}\"", TEXT_COLOR),
        ("✨ Model 1 (Enhancer SLM): Quality enriched prompt in <1ms", ACCENT_COLOR),
        (f"🎯 Model 2 (Router SLM): Selected '{model}' (<2ms inference)", GREEN_COLOR),
        (f"💡 Router Reasoning: Matched frontier benchmark pass rate requirements.", SUBTEXT_COLOR),
        (f"💰 Actual API Cost:   ${cost_act:.6f}", TEXT_COLOR),
        (f"📊 Baseline Cost (GPT-4o): ${cost_base:.6f}", SUBTEXT_COLOR),
        (f"🎉 NET SAVINGS:        ${(cost_base - cost_act):.6f} ({pct_saved}% Saved!)", GREEN_COLOR)
    ]
    
    y_pos = 150
    for text, color in lines:
        draw.text((60, y_pos), text, fill=color, font_size=16)
        y_pos += 35
        
    return img

def create_frame_dashboard():
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ WORMHOLE LIVE WEB DASHBOARD (http://127.0.0.1:8000)")
    
    cards = [
        ("TOTAL REQUESTS", "41", "Active Gateway"),
        ("TOTAL SAVINGS", "$0.0258", "93.4% vs GPT-4o"),
        ("ACTUAL SPEND", "$0.0018", "Baseline: $0.0276"),
        ("AVG JUDGE SCORE", "8.5 / 10", "Auto-Evaluated")
    ]
    
    for idx, (title, val, sub) in enumerate(cards):
        x = 40 + idx * 230
        draw.rectangle([x, 90, x + 210, 190], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
        draw.text((x + 15, 105), title, fill=SUBTEXT_COLOR, font_size=12)
        val_color = GREEN_COLOR if "SAVINGS" in title else (255, 255, 255)
        draw.text((x + 15, 130), val, fill=val_color, font_size=24)
        draw.text((x + 15, 165), sub, fill=SUBTEXT_COLOR, font_size=12)
        
    draw.rectangle([40, 220, WIDTH - 40, HEIGHT - 40], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
    draw.rectangle([40, 220, WIDTH - 40, 260], fill=(31, 41, 55))
    draw.text((55, 232), "ID", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((150, 232), "ORIGINAL PROMPT", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((400, 232), "TARGET MODEL", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((620, 232), "SAVINGS", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((800, 232), "JUDGE SCORE", fill=SUBTEXT_COLOR, font_size=13)
    
    rows = [
        ("wh-a1b2c3", "Format JSON array of colors...", "gemini-1.5-flash", "Saved $0.000698 (96.9%)", "★ 8.5/10"),
        ("wh-d4e5f6", "Write Python palindrome function...", "gemini-1.5-flash", "Saved $0.000726 (97.1%)", "★ 8.5/10"),
        ("wh-7g8h9i", "Design distributed locking system...", "gemini-1.5-flash", "Saved $0.000847 (97.0%)", "★ 8.5/10")
    ]
    
    y = 280
    for req_id, p_text, model, sav, score in rows:
        draw.text((55, y), req_id, fill=SUBTEXT_COLOR, font_size=13)
        draw.text((150, y), p_text, fill=TEXT_COLOR, font_size=13)
        draw.text((400, y), model, fill=ACCENT_COLOR, font_size=13)
        draw.text((620, y), sav, fill=GREEN_COLOR, font_size=13)
        draw.text((800, y), score, fill=GREEN_COLOR, font_size=13)
        y += 40
        draw.line([40, y - 10, WIDTH - 40, y - 10], fill=BORDER_COLOR, width=1)
        
    return img

def render_video():
    generate_voiceover_audio()
    
    # Query exact audio duration
    import soundfile as sf
    data, samplerate = sf.read(TEMP_AUDIO)
    audio_duration = float(len(data)) / samplerate
    print(f"🔊 Generated audio duration: {audio_duration:.2f}s")
    
    fps = 10
    total_frames = int(audio_duration * fps)
    
    print(f"🎥 Rendering High-Fidelity Video Frames ({total_frames} frames @ {fps} FPS)...")
    writer = imageio.get_writer(TEMP_VIDEO, fps=fps)
    
    title_f = create_frame_title()
    c1_f = create_frame_terminal_case(1, "Format a JSON array containing top 3 primary colors.", "gemini-1.5-flash", 0.000022, 0.000720, 96.9)
    c2_f = create_frame_terminal_case(2, "Write a Python palindrome function with unit tests.", "gemini-1.5-flash", 0.000022, 0.000748, 97.1)
    dash_f = create_frame_dashboard()
    
    f_per_scene = total_frames // 4
    import numpy as np

    for _ in range(f_per_scene):
        writer.append_data(np.array(title_f))
    for _ in range(f_per_scene):
        writer.append_data(np.array(c1_f))
    for _ in range(f_per_scene):
        writer.append_data(np.array(c2_f))
    for _ in range(total_frames - 3 * f_per_scene):
        writer.append_data(np.array(dash_f))
        
    writer.close()
    
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("🎬 Merging High-Fidelity AI Voiceover Audio (192kbps AAC) with Video into final MP4...")
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
    
    print(f"🎉 High-Fidelity AI Voiceover Video successfully created at: {OUTPUT_MP4}")

if __name__ == "__main__":
    render_video()
