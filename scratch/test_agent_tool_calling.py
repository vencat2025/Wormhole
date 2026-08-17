import httpx
import json

def test_responses_tool_calling():
    url = "http://127.0.0.1:8000/v1/responses"
    headers = {"Authorization": "Bearer wh_live_demo123456789"}
    
    payload = {
        "model": "gpt-5.6-sol",
        "instructions": "You are a software engineer agent operating in a terminal. Execute actions using tools.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Can you create an app to display an image?"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I can create an app.py file with Flask to serve static/image.jpg."}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Make the changes."}]
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec",
                "description": "Runs a shell command in the workspace directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command line string"}
                    },
                    "required": ["command"]
                }
            }
        ],
        "stream": True
    }

    print("=== Sending Agent Tool Calling Request to /v1/responses ===")
    events = []
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
        for line in r.iter_lines():
            if line:
                print("RAW LINE:", line)
                if line.startswith("data: "):
                    events.append(line[6:])
    
    print("\n=== Parsing Events Received ===")
    function_calls = []
    text_deltas = []
    for evt_str in events:
        if evt_str == "[DONE]":
            continue
        try:
            evt = json.loads(evt_str)
            evt_type = evt.get("type")
            if evt_type in ["response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done"]:
                if evt.get("item", {}).get("type") == "function_call" or "call_id" in evt:
                    function_calls.append(evt)
            elif evt_type in ["response.text.delta", "response.output_text.delta"]:
                text_deltas.append(evt.get("delta", ""))
        except Exception as e:
            pass

    print(f"Total Function Call Events: {len(function_calls)}")
    print(f"Total Text Delta Chunks: {len(text_deltas)}")
    if text_deltas:
        print("Text Content Received:", "".join(text_deltas)[:200])

if __name__ == "__main__":
    test_responses_tool_calling()
