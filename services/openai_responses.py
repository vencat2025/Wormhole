"""Outbound bridge from chat-completions shape to OpenAI's /v1/responses API.

Some OpenAI models refuse function tools on /v1/chat/completions:

    Function tools with reasoning_effort are not supported for gpt-5.6-sol
    in /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Measured against a live account, all three 5.6 variants (sol, terra, luna)
reject tools there. The error offers two ways out, and they are not equivalent:

  * reasoning_effort='none' keeps the call on chat completions but switches the
    reasoning off. For the tier you reach for *because* it reasons, that is a
    downgrade wearing the model's name.
  * /v1/responses accepts tools with reasoning intact.

So the top tier only earns its price through the Responses API, and this module
is what lets the rest of the gateway keep speaking chat completions. It takes
chat-completions kwargs, calls /v1/responses, and translates the reply back --
including streaming -- so every dispatch path is unchanged.

The objects returned here are duck-typed to the attributes the dispatcher
actually reads (choices[0].delta.content, .tool_calls[i].function.arguments,
usage.prompt_tokens, ...) rather than being real litellm models.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import litellm

logger = logging.getLogger("wormhole.responses_api")


# ---------------------------------------------------------------- request ---

def _to_responses_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Flatten chat-completions tools into the Responses API's flat shape.

    Chat nests the schema under "function"; Responses puts name/parameters at
    the top level. Anything already flat is passed through untouched.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict):
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", "") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        elif t.get("type") == "function" and "name" in t:
            out.append(t)
    return out or None


def _to_responses_tool_choice(choice: Any) -> Any:
    """Translate tool_choice, which nests the name on chat and not on Responses."""
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        return {"type": "function", "name": choice["function"].get("name", "")}
    return choice


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a chat message list into Responses API input items.

    Tool traffic changes shape entirely: an assistant turn that called a tool
    becomes a standalone function_call item, and the tool's reply becomes a
    function_call_output item keyed by call_id, rather than both living inside
    the message list as they do on chat completions.
    """
    items: List[Dict[str, Any]] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id") or "",
                "output": content if isinstance(content, str) else json.dumps(content or ""),
            })
            continue

        if role == "assistant" and m.get("tool_calls"):
            if content:
                items.append({"role": "assistant", "content": _as_text(content)})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", "") if isinstance(tc, dict) else "",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "{}",
                })
            continue

        # Responses has no "system" role; the equivalent is "developer".
        if role == "system":
            role = "developer"
        items.append({"role": role or "user", "content": _as_text(content)})
    return items


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and "text" in c:
                parts.append(str(c["text"]))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


# --------------------------------------------------------------- response ---

class _Fn:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, index: int, call_id: str, name: str = "", arguments: str = ""):
        self.index = index
        self.id = call_id
        self.type = "function"
        self.function = _Fn(name, arguments)

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class _Delta:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[_ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class _Message:
    def __init__(self, content: str = "", tool_calls: Optional[List[_ToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class _Usage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Choice:
    def __init__(self, delta=None, message=None, finish_reason=None):
        self.index = 0
        self.delta = delta
        self.message = message
        self.finish_reason = finish_reason


class _Chunk:
    """One streamed chunk, shaped like a litellm chat-completion chunk."""

    def __init__(self, delta: Optional[_Delta] = None, usage: Optional[_Usage] = None,
                 finish_reason: Optional[str] = None):
        self.choices = [_Choice(delta=delta or _Delta(), finish_reason=finish_reason)] if delta or finish_reason else []
        self.usage = usage


class _Response:
    """A complete non-streamed reply, shaped like a litellm ModelResponse."""

    def __init__(self, content: str, tool_calls: List[_ToolCall], usage: _Usage, finish_reason: str):
        self.choices = [_Choice(message=_Message(content, tool_calls or None), finish_reason=finish_reason)]
        self.usage = usage


def _usage_from(obj: Any) -> Optional[_Usage]:
    u = getattr(obj, "usage", None)
    if u is None:
        return None
    # The Responses API names these input_tokens/output_tokens; litellm
    # sometimes normalises them to the chat names. Accept either.
    p = getattr(u, "input_tokens", None)
    if p is None:
        p = getattr(u, "prompt_tokens", None)
    c = getattr(u, "output_tokens", None)
    if c is None:
        c = getattr(u, "completion_tokens", None)
    if p is None and c is None:
        return None
    return _Usage(p or 0, c or 0)


# ------------------------------------------------------------------ entry ---

def _event_type(ev: Any) -> str:
    t = getattr(ev, "type", "")
    return getattr(t, "value", t) or ""


async def _stream_as_chat(stream):
    """Re-emit Responses stream events as chat-completion chunks.

    Tool calls arrive as an output_item.added naming the function, then a run
    of argument deltas. Chat completions expect the same split, so the mapping
    is mostly bookkeeping: give each function_call item a stable integer index,
    because the dispatcher keys its in-flight calls on that.
    """
    index_for_item: Dict[str, int] = {}
    final_usage: Optional[_Usage] = None

    async for ev in stream:
        etype = _event_type(ev)

        if etype == "response.output_text.delta":
            delta = getattr(ev, "delta", "") or ""
            if delta:
                yield _Chunk(delta=_Delta(content=delta))

        elif etype == "response.output_item.added":
            item = getattr(ev, "item", None)
            if item is not None and getattr(item, "type", "") == "function_call":
                item_id = getattr(item, "id", "") or ""
                idx = len(index_for_item)
                index_for_item[item_id] = idx
                yield _Chunk(delta=_Delta(tool_calls=[_ToolCall(
                    index=idx,
                    call_id=getattr(item, "call_id", "") or item_id,
                    name=getattr(item, "name", "") or "",
                    arguments="",
                )]))

        elif etype == "response.function_call_arguments.delta":
            item_id = getattr(ev, "item_id", "") or ""
            idx = index_for_item.get(item_id, 0)
            delta = getattr(ev, "delta", "") or ""
            if delta:
                # id/name empty: the dispatcher already opened this call and
                # only appends arguments from here on.
                yield _Chunk(delta=_Delta(tool_calls=[_ToolCall(
                    index=idx, call_id="", name="", arguments=delta,
                )]))

        elif etype == "response.completed":
            final_usage = _usage_from(getattr(ev, "response", None))

    finish = "tool_calls" if index_for_item else "stop"
    yield _Chunk(delta=_Delta(), usage=final_usage, finish_reason=finish)


def _collect(resp) -> _Response:
    """Fold a non-streamed Responses reply into chat-completion shape."""
    text_parts: List[str] = []
    tool_calls: List[_ToolCall] = []

    for item in (getattr(resp, "output", None) or []):
        itype = getattr(item, "type", "")
        if itype == "function_call":
            tool_calls.append(_ToolCall(
                index=len(tool_calls),
                call_id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                name=getattr(item, "name", "") or "",
                arguments=getattr(item, "arguments", "") or "{}",
            ))
        elif itype == "message":
            for c in (getattr(item, "content", None) or []):
                if getattr(c, "type", "") in ("output_text", "text"):
                    text_parts.append(getattr(c, "text", "") or "")

    usage = _usage_from(resp) or _Usage()
    finish = "tool_calls" if tool_calls else "stop"
    return _Response("".join(text_parts), tool_calls, usage, finish)


async def aresponses_as_chat(**kwargs):
    """Call /v1/responses using chat-completions kwargs, and translate back.

    Drop-in for litellm.acompletion for the models that need it.
    """
    model = kwargs.get("model", "")
    # These are OpenAI-only models; litellm needs the provider prefix because
    # the 5.6 ids are newer than its model map.
    if "/" not in model:
        model = f"openai/{model}"

    payload: Dict[str, Any] = {
        "model": model,
        "input": _to_responses_input(kwargs.get("messages") or []),
        "stream": bool(kwargs.get("stream")),
    }

    # The caller's conversation key, so the provider can find the warm prefix.
    # Dropping it is how a session pays full rate for the same 31k tokens on
    # every turn; see the note in dispatcher.acompletion_with_backoff.
    if kwargs.get("prompt_cache_key"):
        payload["prompt_cache_key"] = kwargs["prompt_cache_key"]

    tools = _to_responses_tools(kwargs.get("tools"))
    if tools:
        payload["tools"] = tools
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = _to_responses_tool_choice(kwargs["tool_choice"])

    # Reasoning models spend output budget thinking before they answer, so a
    # ceiling tuned for a chat reply cuts them off mid-thought. Carry the
    # caller's value across under the name Responses uses, and floor it.
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_tokens:
        payload["max_output_tokens"] = max(int(max_tokens), 2000)

    # temperature/top_p are deliberately not forwarded: the reasoning tiers
    # reject anything but their default and fail the whole call over it.

    if payload["stream"]:
        stream = await litellm.aresponses(**payload)
        return _stream_as_chat(stream)

    resp = await litellm.aresponses(**payload)
    return _collect(resp)
