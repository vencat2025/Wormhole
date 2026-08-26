"""Translation between the Anthropic Messages API and the OpenAI shape.

Claude Code speaks the Anthropic wire format, which differs from OpenAI's in
three ways that each break tool use independently:

  - tools carry `input_schema`, not `function.parameters`;
  - tool calls and their results are content blocks inside messages, not a
    separate `tool_calls` field and `tool` role;
  - responses are a list of typed content blocks, so a tool call has nowhere
    to go unless a `tool_use` block is emitted for it.

Getting any one of these wrong makes the model look like it answered in prose
while silently doing nothing.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple


def _blocks_to_text(content: Any) -> str:
    """Flatten Anthropic content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    return ""


def convert_anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Anthropic `input_schema` -> OpenAI `function.parameters`."""
    if not tools:
        return None
    converted = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Already OpenAI-shaped (some clients mix formats).
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            converted.append(t)
            continue
        name = t.get("name")
        if not name:
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return converted or None


def convert_anthropic_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if kind == "none":
        return "none"
    return "auto"


def convert_anthropic_messages(raw_request: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Any]:
    """Build OpenAI messages/tools/tool_choice from an Anthropic request."""
    messages: List[Dict[str, Any]] = []

    system = raw_request.get("system")
    if system:
        system_text = _blocks_to_text(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for m in raw_request.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content or "")})
            continue

        # A single Anthropic message can hold text, tool calls and tool
        # results at once; OpenAI needs those split across messages.
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    text_parts.append(block)
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or block.get("id") or "",
                    "content": _blocks_to_text(block.get("content")) or "(no output)",
                })
            elif btype == "image":
                text_parts.append("[image omitted]")

        if role == "assistant":
            if text_parts or tool_calls:
                msg: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
        else:
            # Tool results must precede any new user text so the assistant
            # tool_calls message they answer stays adjacent to them.
            messages.extend(tool_results)
            joined = "\n".join(p for p in text_parts if p)
            if joined:
                messages.append({"role": "user", "content": joined})

    tools = convert_anthropic_tools(raw_request.get("tools"))
    tool_choice = convert_anthropic_tool_choice(raw_request.get("tool_choice"))
    return messages, tools, tool_choice


def build_anthropic_message(
    message_id: str,
    model: str,
    text: str,
    tool_calls: Optional[List[Dict[str, Any]]],
    input_tokens: int,
    output_tokens: int,
) -> Dict[str, Any]:
    """Assemble a non-streaming Anthropic response with tool_use blocks."""
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
        except (ValueError, TypeError):
            parsed = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": fn.get("name", ""),
            "input": parsed,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        # Claude Code continues the agent loop only on "tool_use"; reporting
        # "end_turn" alongside a tool call ends the turn instead.
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def sse(event_type: str, payload: Dict[str, Any]) -> str:
    """Anthropic SSE frames carry both an event name and a data line."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
