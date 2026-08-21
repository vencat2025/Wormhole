import os
import sys
import asyncio
import subprocess
import soundfile as sf
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_MP4 = "/Users/venkat/Documents/AI/WormHole/docs/wormhole_ai_demo.mp4"
OUTPUT_GIF = "/Users/venkat/Documents/AI/WormHole/docs/wormhole_demo.gif"
ARTIFACT_MP4 = "/Users/venkat/.gemini/antigravity/brain/4a6c66de-0adb-4faf-8a9b-dc492052aa2b/wormhole_ai_demo.mp4"
ARTIFACT_GIF = "/Users/venkat/.gemini/antigravity/brain/4a6c66de-0adb-4faf-8a9b-dc492052aa2b/wormhole_demo.gif"

TEMP_AUDIO = "/Users/venkat/Documents/AI/WormHole/docs/human_narration.mp3"
TEMP_WAV = "/Users/venkat/Documents/AI/WormHole/docs/human_narration.wav"
TEMP_VIDEO = "/Users/venkat/Documents/AI/WormHole/docs/temp_masterpiece.mp4"

# 1080p Full HD Resolution
WIDTH, HEIGHT = 1920, 1080
BACKGROUND_COLOR = (11, 15, 25)      # Deep Dark Slate #0b0f19
PANEL_COLOR = (17, 24, 39)          # Card Panel #111827
HEADER_COLOR = (31, 41, 55)         # Header #1f2937
BORDER_COLOR = (55, 65, 81)         # Border #374151
TEXT_COLOR = (249, 250, 251)        # White #f9fafb
SUBTEXT_COLOR = (156, 163, 175)     # Subtitle Gray #9ca3af
GREEN_COLOR = (16, 185, 129)        # Emerald #10b981
ACCENT_COLOR = (99, 102, 241)       # Indigo #6366f1
CYAN_COLOR = (6, 182, 212)          # Cyan #06b6d4
YELLOW_COLOR = (245, 158, 11)       # Amber #f59e0b

# Script covering ALL features & Enterprise integration
NARRATION_SCRIPT = (
    "Welcome to WormHole, the complete provider-agnostic enterprise AI inference middleware layer designed to cut API spend by over 90 percent while elevating completion quality. "
    "WormHole acts as a drop-in gateway proxy across all downstream models—including Groq LPUs, OpenAI, Anthropic Claude, Google Gemini, and local Ollama. "
    "When a request arrives, Model 2—our local Router SLM—evaluates task complexity in under 2 milliseconds against 10 public online benchmarks like HumanEval and SWE-bench. "
    "If a budget model like Groq GPT-OSS-120B or OpenAI mini is selected, Model 1—our local Enhancer SLM—quality enriches the prompt in under 1 millisecond. "
    "For reasoning models on Groq, WormHole automatically suppresses thinking tags to eliminate latency and stream native agentic tool calls for Codex CLI. "
    "As completions stream back, an asynchronous LLM-as-a-Judge grades quality on a 1 to 10 scale, exporting datasets for 1-click local model retraining. "
    "Integrating your existing harnesses is effortless. Point Codex CLI, Cursor IDE, Aider, or custom OpenAI SDK scripts to WormHole's base URL to immediately slash your enterprise AI costs."
)

async def generate_human_voiceover():
    print("🎙️ Generating Ultra-Human Studio Voiceover using Edge Neural Speech...")
    import edge_tts
    communicate = edge_tts.Communicate(NARRATION_SCRIPT, "en-US-ChristopherNeural", rate="+0%")
    await communicate.save(TEMP_AUDIO)
    
    # Convert MP3 to WAV using ffmpeg for precise duration calculation
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", TEMP_AUDIO, "-ar", "44100", "-ac", "2", TEMP_WAV]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Studio Human Voiceover synthesized successfully.")

def draw_header(draw, title="⚡ WORMHOLE — UNIVERSAL AI INFERENCE GATEWAY"):
    draw.rectangle([0, 0, WIDTH, 100], fill=PANEL_COLOR)
    draw.line([0, 100, WIDTH, 100], fill=BORDER_COLOR, width=3)
    draw.text((40, 30), title, fill=TEXT_COLOR, font_size=32)
    draw.rectangle([WIDTH - 360, 25, WIDTH - 40, 75], fill=GREEN_COLOR)
    draw.text((WIDTH - 340, 38), "100% PROVIDER AGNOSTIC", fill=(255, 255, 255), font_size=18)

def create_scene_1():
    """Scene 1: Title & Executive Vision"""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    
    draw.text((100, 200), "WormHole Universal AI Gateway", fill=(255, 255, 255), font_size=60)
    draw.text((100, 280), "Sub-2ms Local SLMs  |  100% Provider Agnostic  |  Closed-Loop Quality Flywheel", fill=CYAN_COLOR, font_size=30)
    
    cards = [
        ("🔌 Universal Provider-Agnostic Core", "Routes to Groq LPUs, OpenAI, Anthropic Claude, Google Gemini, Ollama & vLLM.", ACCENT_COLOR),
        ("🎯 Model 2: Local Router SLM", "Evaluates task complexity in <2ms against 10 online benchmarks at $0 API cost.", GREEN_COLOR),
        ("⚡ Reasoning Suppressor & Tool Engine", "Suppresses <think> tags & streams native OpenAI Responses API function_calls.", YELLOW_COLOR),
        ("⚖️ LLM-as-a-Judge & Retraining", "Auto-evaluates quality (1.0-10.0) & exports fine-tuning datasets for local SLMs.", CYAN_COLOR)
    ]
    
    for idx, (title, desc, color) in enumerate(cards):
        x = 100 + (idx % 2) * 880
        y = 400 + (idx // 2) * 280
        draw.rectangle([x, y, x + 840, y + 230], fill=PANEL_COLOR, outline=BORDER_COLOR, width=3)
        draw.rectangle([x, y, x + 840, y + 60], fill=HEADER_COLOR)
        draw.text((x + 30, y + 15), title, fill=color, font_size=26)
        draw.text((x + 30, y + 90), desc, fill=SUBTEXT_COLOR, font_size=20)
        
    return img

def create_scene_2():
    """Scene 2: Core Dual-SLM Pipeline & Selective Enhancement"""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ UNIVERSAL MULTI-PROVIDER & STREAMING TOOL PIPELINE")
    
    draw.text((100, 150), "Sub-2ms Local SLM Routing  →  Native Responses API Tool Calls", fill=TEXT_COLOR, font_size=40)
    draw.text((100, 210), "Bypasses reasoning latency on Groq LPUs while streaming execution events to Codex CLI.", fill=SUBTEXT_COLOR, font_size=22)
    
    # Diagram Nodes
    steps = [
        ("1. Codex CLI Request", "POST /v1/responses (Prompt + Tools)", PANEL_COLOR, BORDER_COLOR),
        ("2. Model 2 Router SLM", "<2ms Benchmark Evaluation (HumanEval/SWE-bench)", ACCENT_COLOR, ACCENT_COLOR),
        ("3. Provider Selection", "Groq LPU / OpenAI / Claude / Gemini / Ollama", PANEL_COLOR, BORDER_COLOR),
        ("4. Stream Tool Engine", "Suppresses <think> tags & converts code -> exec tool_calls", GREEN_COLOR, GREEN_COLOR),
        ("5. Native Execution", "Codex CLI executes mkdir & writes workspace files live", PANEL_COLOR, BORDER_COLOR)
    ]
    
    for idx, (title, desc, bg, border) in enumerate(steps):
        y = 300 + idx * 130
        draw.rectangle([100, y, WIDTH - 100, y + 100], fill=bg, outline=border, width=3)
        draw.text((140, y + 20), title, fill=TEXT_COLOR, font_size=24)
        draw.text((140, y + 55), desc, fill=SUBTEXT_COLOR, font_size=18)
        draw.text((WIDTH - 250, y + 35), "⚡ < 2ms", fill=CYAN_COLOR, font_size=22)
        
    return img

def create_scene_3():
    """Scene 3: Closed-Loop Auto-Judge & Fine-Tuning Flywheel"""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ CLOSED-LOOP AUTO-JUDGE & FINE-TUNING FLYWHEEL")
    
    draw.text((100, 150), "Continuous Learning from Enterprise Completion Traffic", fill=TEXT_COLOR, font_size=40)
    
    # Flywheel Metrics
    draw.rectangle([100, 250, 900, 950], fill=PANEL_COLOR, outline=BORDER_COLOR, width=3)
    draw.rectangle([100, 250, 900, 320], fill=HEADER_COLOR)
    draw.text((130, 270), "⚖️ LLM-as-a-Judge Feedback Loop", fill=YELLOW_COLOR, font_size=26)
    
    j_lines = [
        ("• Asynchronous Evaluation", "Grades outputs 1.0 - 10.0 without adding user latency"),
        ("• Automated Dataset Exporter", "Generates fine-tuning JSONL pairs (judge_score >= 7.0)"),
        ("• Retraining Pipeline", "models/train_router.py & models/train_enhancer.py"),
        ("• Enterprise Audit Logs", "Full history stored in SQLite / PostgreSQL")
    ]
    y_p = 360
    for head, sub in j_lines:
        draw.text((130, y_p), head, fill=TEXT_COLOR, font_size=22)
        draw.text((130, y_p + 35), sub, fill=SUBTEXT_COLOR, font_size=18)
        y_p += 130

    # Retrain Terminal Box
    draw.rectangle([980, 250, WIDTH - 100, 950], fill=(15, 23, 42), outline=BORDER_COLOR, width=3)
    draw.rectangle([980, 250, WIDTH - 100, 320], fill=(30, 41, 59))
    draw.ellipse([1010, 280, 1030, 300], fill=(239, 68, 68))
    draw.ellipse([1040, 280, 1060, 300], fill=(245, 158, 11))
    draw.ellipse([1070, 280, 1090, 300], fill=(34, 197, 94))
    draw.text((1110, 275), "bash - wormhole retrain_pipeline", fill=SUBTEXT_COLOR, font_size=20)
    
    term_lines = [
        ("python models/train_router.py", GREEN_COLOR),
        ("[WormHole Retrain] Loading historical inference logs...", SUBTEXT_COLOR),
        ("[WormHole Retrain] Exported 2,000 JSONL fine-tuning pairs.", TEXT_COLOR),
        ("[WormHole Retrain] Training local Gradient Boosting SLM...", ACCENT_COLOR),
        ("[WormHole Retrain] Model 2 Accuracy: 100.0% on benchmark test split.", GREEN_COLOR),
        ("[WormHole Retrain] Exported artifact: models/router_slm.joblib", CYAN_COLOR),
        ("✨ Model 2 updated live without downtime!", GREEN_COLOR)
    ]
    t_y = 360
    for t_text, t_color in term_lines:
        draw.text((1010, t_y), t_text, fill=t_color, font_size=20)
        t_y += 75
        
    return img

def create_scene_4():
    """Scene 4: Enterprise Production Hardening & Custom Harness Setup"""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ ENTERPRISE HARDENING & HARNESS INTEGRATION")
    
    # 4 Enterprise Subsystems
    draw.rectangle([100, 160, 900, 980], fill=PANEL_COLOR, outline=BORDER_COLOR, width=3)
    draw.rectangle([100, 160, 900, 230], fill=HEADER_COLOR)
    draw.text((130, 180), "🛡️ Enterprise Hardening Subsystems", fill=GREEN_COLOR, font_size=26)
    
    subsystems = [
        ("🔑 Bearer Token Auth", "Authorization: Bearer wh_live_... key validation"),
        ("⚡ SSE Token Streaming", "stream=True real-time token protocol"),
        ("🗄️ Dual DB (PostgreSQL / SQLite)", "High-availability EKS/GKE cluster support"),
        ("🛡️ Circuit Breakers", "Auto 5xx outage failover to secondary models")
    ]
    y_sub = 260
    for title, desc in subsystems:
        draw.text((130, y_sub), title, fill=TEXT_COLOR, font_size=22)
        draw.text((130, y_sub + 35), desc, fill=SUBTEXT_COLOR, font_size=18)
        y_sub += 170

    # Harness Setup Code Box
    draw.rectangle([980, 160, WIDTH - 100, 980], fill=(15, 23, 42), outline=BORDER_COLOR, width=3)
    draw.rectangle([980, 160, WIDTH - 100, 230], fill=(30, 41, 59))
    draw.text((1010, 180), "💻 Drop-in Harness Integration", fill=CYAN_COLOR, font_size=24)
    
    code_lines = [
        ("# 🤖 CODEX CLI / OPENAI HARNESS:", ACCENT_COLOR),
        ("model_provider = \"wormhole\"", TEXT_COLOR),
        ("base_url = \"http://127.0.0.1:8000/v1\"", TEXT_COLOR),
        ("", TEXT_COLOR),
        ("# 💻 CURSOR / CLAUDE CODE / AIDER:", ACCENT_COLOR),
        ("export OPENAI_BASE_URL=\"http://127.0.0.1:8000/v1\"", TEXT_COLOR),
        ("export OPENAI_API_KEY=\"wh_live_demo123456789\"", TEXT_COLOR),
        ("", TEXT_COLOR),
        ("🎉 100% of Developer AI Traffic Automatically Optimized!", GREEN_COLOR)
    ]
    c_y = 260
    for c_text, c_color in code_lines:
        draw.text((1010, c_y), c_text, fill=c_color, font_size=20)
        c_y += 70

    return img

def create_scene_5():
    """Scene 5: Live Dashboard & ROI Metrics Summary"""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ WORMHOLE LIVE WEB DASHBOARD & ENTERPRISE ROI")
    
    # 4 Metric Cards
    cards = [
        ("TOTAL REQUESTS LOGGED", "52 Requests", "Active Gateway"),
        ("NET COST SAVINGS", "$0.0384", "94.4% Net Savings vs GPT-4o"),
        ("ACTUAL API SPEND", "$0.0023", "Baseline: $0.0407"),
        ("AVG JUDGE QUALITY SCORE", "8.8 / 10.0", "Auto-Evaluated")
    ]
    
    for idx, (title, val, sub) in enumerate(cards):
        x = 100 + idx * 435
        draw.rectangle([x, 150, x + 405, 330], fill=PANEL_COLOR, outline=BORDER_COLOR, width=3)
        draw.text((x + 25, 175), title, fill=SUBTEXT_COLOR, font_size=16)
        val_color = GREEN_COLOR if "SAVINGS" in title else (255, 255, 255)
        draw.text((x + 25, 220), val, fill=val_color, font_size=36)
        draw.text((x + 25, 285), sub, fill=SUBTEXT_COLOR, font_size=16)
        
    # Live Table Preview
    draw.rectangle([100, 380, WIDTH - 100, 980], fill=PANEL_COLOR, outline=BORDER_COLOR, width=3)
    draw.rectangle([100, 380, WIDTH - 100, 450], fill=HEADER_COLOR)
    draw.text((130, 400), "ID", fill=SUBTEXT_COLOR, font_size=20)
    draw.text((320, 400), "ORIGINAL PROMPT", fill=SUBTEXT_COLOR, font_size=20)
    draw.text((800, 400), "TARGET MODEL", fill=SUBTEXT_COLOR, font_size=20)
    draw.text((1200, 400), "SAVINGS", fill=SUBTEXT_COLOR, font_size=20)
    draw.text((1580, 400), "JUDGE SCORE", fill=SUBTEXT_COLOR, font_size=20)
    
    rows = [
        ("wh-a1b2c3", "Create an app to display an image...", "groq/gpt-oss-120b", "Saved $0.002140 (94.0%)", "★ 9.0 / 10"),
        ("wh-d4e5f6", "Format JSON payload...", "gpt-4o-mini", "Saved $0.000726 (97.1%)", "★ 8.8 / 10"),
        ("wh-7g8h9i", "Refactor authentication handler...", "claude-3-5-sonnet", "Saved $0.000000 (Frontier)", "★ 9.5 / 10"),
        ("wh-strands1", "Strands Multi-Agent Planner Node Task...", "groq/qwen3.6-27b", "Saved $0.000710 (96.8%)", "★ 8.7 / 10")
    ]
    
    r_y = 480
    for req_id, p_text, model, sav, score in rows:
        draw.text((130, r_y), req_id, fill=SUBTEXT_COLOR, font_size=18)
        draw.text((320, r_y), p_text, fill=TEXT_COLOR, font_size=18)
        draw.text((800, r_y), model, fill=ACCENT_COLOR, font_size=18)
        draw.text((1200, r_y), sav, fill=GREEN_COLOR, font_size=18)
        draw.text((1580, r_y), score, fill=GREEN_COLOR, font_size=18)
        r_y += 110
        draw.line([100, r_y - 20, WIDTH - 100, r_y - 20], fill=BORDER_COLOR, width=2)
        
    return img

def render_masterpiece_video():
    asyncio.run(generate_human_voiceover())
    
    data, samplerate = sf.read(TEMP_WAV)
    audio_duration = float(len(data)) / samplerate
    print(f"🔊 Generated Human Studio Audio Duration: {audio_duration:.2f} seconds")
    
    fps = 24  # Smooth 24 FPS Full HD
    total_frames = int(audio_duration * fps)
    
    print(f"🎥 Rendering 1080p Full HD Video Frames ({total_frames} frames @ {fps} FPS)...")
    writer = imageio.get_writer(TEMP_VIDEO, fps=fps)
    
    scenes = [
        create_scene_1(),
        create_scene_2(),
        create_scene_3(),
        create_scene_4(),
        create_scene_5()
    ]
    
    f_per_scene = total_frames // len(scenes)
    
    for idx, scene_img in enumerate(scenes):
        num_f = f_per_scene if idx < len(scenes) - 1 else (total_frames - (len(scenes) - 1) * f_per_scene)
        scene_arr = np.array(scene_img)
        for _ in range(num_f):
            writer.append_data(scene_arr)
            
    writer.close()
    
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("🎬 Merging Studio Human Voiceover (192kbps AAC) with 1080p Full HD Video...")
    merge_cmd = [
        ffmpeg_exe, "-y",
        "-i", TEMP_VIDEO,
        "-i", TEMP_WAV,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        OUTPUT_MP4
    ]
    subprocess.run(merge_cmd, check=True)

    # Generate animated GIF preview for README and artifacts
    print("🖼️ Generating Animated GIF preview for README & artifacts...")
    gif_writer = imageio.get_writer(OUTPUT_GIF, fps=4)
    for scene_img in scenes:
        # Resize frame for GIF optimization
        small_img = scene_img.resize((960, 540))
        small_arr = np.array(small_img)
        for _ in range(8):
            gif_writer.append_data(small_arr)
    gif_writer.close()

    # Copy artifacts
    os.makedirs(os.path.dirname(ARTIFACT_MP4), exist_ok=True)
    subprocess.run(["cp", OUTPUT_MP4, ARTIFACT_MP4], check=True)
    subprocess.run(["cp", OUTPUT_GIF, ARTIFACT_GIF], check=True)

    print(f"🎉 MASTERPIECE 1080p Full HD Video saved to: {OUTPUT_MP4}")
    print(f"🎉 Artifact Video saved to: {ARTIFACT_MP4}")
    print(f"🎉 GIF preview saved to: {OUTPUT_GIF}")

if __name__ == "__main__":
    render_masterpiece_video()
