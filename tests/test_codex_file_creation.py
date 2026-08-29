import os
import json
import pytest
import subprocess
import httpx
from db.database import init_db
from main import app

init_db()

# This test calls real providers and asserts on what a model chose to emit, so
# it fails whenever a key is missing, a quota is spent, or a model simply
# answers in prose that turn. That is not something a newcomer running the
# README's "pytest tests/ -q" should have to interpret, so it is opt-in.
@pytest.mark.skipif(
    os.getenv("WORMHOLE_LIVE_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="Live provider test. Set WORMHOLE_LIVE_TESTS=1 to run it.",
)
@pytest.mark.asyncio
async def test_codex_file_creation_live_proxy():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DIR = os.path.join(PROJECT_ROOT, "scratch", "codex_test_workspace")
    os.makedirs(TEST_DIR, exist_ok=True)

    test_prompts = [
        ("Flask Image App", "Please write a python code block for app.py to display an image on a web page using Flask."),
        ("Node Express Server", "Please write a javascript code block for server.js to run a simple Express server.")
    ]

    headers = {
        "Authorization": "Bearer wh_live_demo123456789",
        "Content-Type": "application/json"
    }

    tools = [
        {
            "type": "function",
            "name": "exec",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    ]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for name, prompt in test_prompts:
            payload = {
                "model": "wormhole-auto",
                "input": [{"type": "message", "role": "user", "content": prompt}],
                "tools": tools,
                "stream": True
            }
            
            executed_cmds = []
            async with client.stream("POST", "/v1/responses", json=payload, headers=headers) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_str)
                        if evt.get("type") in ["response.output_item.added", "response.output_item.done"]:
                            item = evt.get("item", {})
                            if item.get("type") == "function_call" and item.get("name") == "exec":
                                args_raw = item.get("arguments", "{}")
                                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                                cmd = args.get("command", "")
                                if cmd and cmd not in executed_cmds:
                                    executed_cmds.append(cmd)
                    except Exception:
                        pass

            assert len(executed_cmds) > 0, f"No tool calls extracted for {name}"
            for cmd in executed_cmds:
                if "cat << 'EOF'" in cmd or "mkdir -p" in cmd or "echo " in cmd or "touch " in cmd:
                    res = subprocess.run(cmd, shell=True, cwd=TEST_DIR, capture_output=True, text=True)
                    assert res.returncode == 0, f"Command execution failed: {cmd}"

    # Verify created files exist on disk
    files_created = [f for _, _, files in os.walk(TEST_DIR) for f in files]
    assert len(files_created) > 0, "No files were created in workspace"
    assert "app.py" in files_created or "server.js" in files_created
