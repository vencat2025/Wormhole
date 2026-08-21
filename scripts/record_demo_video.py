import os
import time
import imageio
from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = "/Users/venkat/Documents/AI/WormHole/docs/wormhole_demo.gif"

WIDTH, HEIGHT = 1000, 600
BACKGROUND_COLOR = (11, 15, 25)  # #0b0f19
TEXT_COLOR = (249, 250, 251)
SUBTEXT_COLOR = (156, 163, 175)
GREEN_COLOR = (16, 185, 129)
ACCENT_COLOR = (99, 102, 241)
PANEL_COLOR = (17, 24, 39)
BORDER_COLOR = (31, 41, 55)

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
    
    # Hero Title
    draw.text((50, 140), "WormHole Universal AI Inference Gateway", fill=(255, 255, 255), font_size=36)
    draw.text((50, 190), "100% Provider-Agnostic (Groq, OpenAI, Anthropic, Gemini, Ollama)", fill=SUBTEXT_COLOR, font_size=20)
    
    # Feature Boxes
    boxes = [
        ("🔌 Provider-Agnostic Core", "Routes to Groq, OpenAI, Claude, Gemini & Ollama"),
        ("🎯 Model 2: Local Router SLM", "Routes tasks via sub-2ms SLMs (<$0 cost)"),
        ("✨ Model 1: Local Enhancer SLM", "Enriches prompts for quality in <1ms"),
        ("🛠️ Stream Tool Engine", "Converts code -> native CLI tool_calls (exec)")
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
    draw_header(draw, f"⚡ DEMO CASE {case_num}: REAL-TIME AGENTIC TOOL EXECUTION")
    
    # Terminal Window
    draw.rectangle([40, 90, WIDTH - 40, HEIGHT - 40], fill=(15, 23, 42), outline=BORDER_COLOR, width=2)
    draw.rectangle([40, 90, WIDTH - 40, 125], fill=(30, 41, 59))
    draw.ellipse([55, 103, 67, 115], fill=(239, 68, 68))
    draw.ellipse([75, 103, 87, 115], fill=(245, 158, 11))
    draw.ellipse([95, 103, 107, 115], fill=(34, 197, 94))
    draw.text((120, 100), "codex - wormhole responses_api_harness", fill=SUBTEXT_COLOR, font_size=14)
    
    lines = [
        (f"📥 PROMPT: \"{prompt_text}\"", TEXT_COLOR),
        ("✨ Model 1 (Enhancer SLM): Quality enriched prompt in <1ms", ACCENT_COLOR),
        (f"🎯 Model 2 (Router SLM): Selected '{model}' (<2ms inference)", GREEN_COLOR),
        ("🔌 Provider Gateway: Dispatched via LiteLLM Multi-Provider Engine", SUBTEXT_COLOR),
        ("🛠️ Tool Engine: Extracted 4 tool_calls (mkdir static / write app.py)", GREEN_COLOR),
        (f"💰 Actual API Cost:   ${cost_act:.6f}  (GPT-4o Baseline: ${cost_base:.6f})", TEXT_COLOR),
        (f"🎉 NET SAVINGS:        ${(cost_base - cost_act):.6f} ({pct_saved}% Saved!)", GREEN_COLOR)
    ]
    
    y_pos = 150
    for text, color in lines:
        draw.text((60, y_pos), text, fill=color, font_size=16)
        y_pos += 35
        
    return img

def create_frame_dashboard(savings_str="$0.0258", pct_str="93.4%", requests_cnt="48"):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw, "⚡ WORMHOLE LIVE WEB DASHBOARD (http://127.0.0.1:8000)")
    
    # Metric Cards
    cards = [
        ("TOTAL REQUESTS", requests_cnt, "Active Gateway"),
        ("TOTAL SAVINGS", savings_str, f"{pct_str} vs GPT-4o"),
        ("ACTUAL SPEND", "$0.0018", "Baseline: $0.0276"),
        ("AVG JUDGE SCORE", "8.8 / 10", "Auto-Evaluated")
    ]
    
    for idx, (title, val, sub) in enumerate(cards):
        x = 40 + idx * 230
        draw.rectangle([x, 90, x + 210, 190], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
        draw.text((x + 15, 105), title, fill=SUBTEXT_COLOR, font_size=12)
        val_color = GREEN_COLOR if "SAVINGS" in title else (255, 255, 255)
        draw.text((x + 15, 130), val, fill=val_color, font_size=24)
        draw.text((x + 15, 165), sub, fill=SUBTEXT_COLOR, font_size=12)
        
    # Table Preview
    draw.rectangle([40, 220, WIDTH - 40, HEIGHT - 40], fill=PANEL_COLOR, outline=BORDER_COLOR, width=2)
    draw.rectangle([40, 220, WIDTH - 40, 260], fill=(31, 41, 55))
    draw.text((55, 232), "ID", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((150, 232), "ORIGINAL PROMPT", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((400, 232), "TARGET MODEL", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((620, 232), "SAVINGS", fill=SUBTEXT_COLOR, font_size=13)
    draw.text((800, 232), "JUDGE SCORE", fill=SUBTEXT_COLOR, font_size=13)
    
    rows = [
        ("wh-a1b2c3", "Create an app to display an image...", "groq/gpt-oss-120b", "Saved $0.002140 (94.0%)", "★ 9.0/10"),
        ("wh-d4e5f6", "Format JSON payload...", "gpt-4o-mini", "Saved $0.000726 (97.1%)", "★ 8.8/10"),
        ("wh-7g8h9i", "Refactor authentication handler...", "claude-3-5-sonnet", "Saved $0.000000 (Frontier)", "★ 9.5/10")
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

def generate_video():
    print("🎥 Generating WormHole High-Definition Animated Demo Recording...")
    frames = []
    
    # 1. Intro Screen (Hold for 3 seconds)
    title_frame = create_frame_title()
    for _ in range(15):
        frames.append(title_frame)
        
    # 2. Case 1 (Hold for 3 seconds)
    c1 = create_frame_terminal_case(1, "Can you create an app to simply display an image on a web page?", "groq/openai/gpt-oss-120b", 0.000135, 0.002250, 94.0)
    for _ in range(15):
        frames.append(c1)
        
    # 3. Case 2 (Hold for 3 seconds)
    c2 = create_frame_terminal_case(2, "Write a Python palindrome function with unit tests.", "groq/openai/gpt-oss-20b", 0.000022, 0.000748, 97.1)
    for _ in range(15):
        frames.append(c2)

    # 4. Live Dashboard Screen (Hold for 4 seconds)
    dash = create_frame_dashboard()
    for _ in range(20):
        frames.append(dash)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimsave(OUTPUT_PATH, frames, fps=5)

    print(f"✅ Demo Video Recording saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_video()
