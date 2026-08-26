import asyncio
import time
import uuid
import json
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional
import re
import litellm
from sqlmodel import Session
from config import settings, CandidateModelConfig
from db.database import engine
from db.models import InferenceLog

logger = logging.getLogger("wormhole.dispatcher")

PROVIDER_FAILURE_COUNTS: Dict[str, int] = {}

# Providers whose credentials were rejected this run. Populated from real
# auth failures rather than assumed, since a key can be present but invalid.
UNAUTHENTICATED_PROVIDERS: set = set()

# Providers report how long to wait when a per-minute token budget is hit.
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)
MAX_RATE_LIMIT_WAIT_SECONDS = 45.0


def _retry_after_seconds(err: Exception) -> Optional[float]:
    match = _RETRY_AFTER_RE.search(str(err))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


async def acompletion_with_backoff(attempts: int = 3, **kwargs):
    """Call the model, waiting out per-minute token limits rather than failing.

    An agent loop issues several requests in quick succession, so a small
    plan's second or third step routinely lands inside the same rate-limit
    window as its first. Failing over at that point is useless because the
    sibling models share the same account quota; waiting the advertised
    interval is what actually lets the turn complete.
    """
    for attempt in range(attempts):
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as err:
            wait = _retry_after_seconds(err)
            if wait is None or attempt == attempts - 1 or wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            logger.info(
                f"Rate limited on '{kwargs.get('model')}'; waiting {wait:.1f}s "
                f"(attempt {attempt + 1}/{attempts})."
            )
            await asyncio.sleep(wait + 1.0)

import os

def prune_messages_for_token_limit(messages: List[Dict[str, Any]], max_tokens: int = 4500) -> List[Dict[str, Any]]:
    """
    Prunes long conversation history to fit within Groq On-Demand TPM limits (8,000 TPM).
    Preserves system instructions and recent user prompts while discarding older middle messages.
    """
    def estimate_tokens(msgs):
        return sum(len(str(m.get("content", ""))) // 4 for m in msgs if isinstance(m, dict))

    if estimate_tokens(messages) <= max_tokens:
        return messages

    system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    non_system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

    if len(non_system_msgs) <= 2:
        return messages

    # Always keep the final two turns, then fill backwards while budget allows.
    kept = non_system_msgs[-2:]
    budget = max_tokens - estimate_tokens(system_msgs) - estimate_tokens(kept)
    for m in reversed(non_system_msgs[:-2]):
        m_tokens = len(str(m.get("content", ""))) // 4
        if budget - m_tokens < 0:
            break
        kept.insert(0, m)
        budget -= m_tokens

    # A `tool` message is only valid when the assistant `tool_calls` message it
    # answers is still present. Cutting mid-pair makes providers reject the
    # whole request, which surfaces as the agent silently giving up mid-task.
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)

    return system_msgs + kept

def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    tool_calls = []
    
    # 0. Direct shell cat file creation commands (cat << 'EOF' > filename ... EOF)
    cat_matches = re.findall(r"(cat\s+<<\s*['\"]?EOF['\"]?\s*>\s*([^\n]+).*?EOF)", text, re.DOTALL)
    for full_cmd, fn in cat_matches:
        clean_cmd = full_cmd.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\").strip()
        if clean_cmd:
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": clean_cmd})})

    # 1. XML <exec> command </exec> or <exec>command = "..."</exec>
    exec_matches = re.findall(r'<exec>(.*?)</exec>', text, re.DOTALL)
    for cmd in exec_matches:
        clean_cmd = cmd.strip()
        if clean_cmd.startswith("command =") or clean_cmd.startswith("command="):
            clean_cmd = clean_cmd.split("=", 1)[1].strip().strip('"').strip("'")
        if clean_cmd:
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": clean_cmd})})

    # 2. Parentheses/Brackets (exec) command or [exec] command
    paren_matches = re.findall(r"[\(\[]exec[\)\]]\s*([^\n]+)", text, re.IGNORECASE)
    for cmd in paren_matches:
        clean_cmd = cmd.strip().strip("`").strip("'")
        if clean_cmd and len(clean_cmd) > 1 and not clean_cmd.startswith("`"):
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": clean_cmd})})

    # 3. Markdown Code Blocks: find all ```lang \n content ``` blocks and extract filename
    blocks = re.findall(r"((?:[^\n]+\n){1,3})?\s*```([a-zA-Z0-9_-]*)\n(.*?)```", text, re.DOTALL)
    
    for prec_text, lang, code_content in blocks:
        code_content = code_content.strip()
        if not code_content:
            continue
            
        filename = None
        lang = (lang or "").lower().strip()
        
        # Check first line inside code content for # app.py, // script.js, <!-- index.html -->
        first_line = code_content.split("\n")[0].strip()
        comment_fn_match = re.search(r"(?:#|//|<!--|\*)\s*([a-zA-Z0-9_\-/\.]+\.[a-zA-Z0-9]+)\b", first_line)
        if comment_fn_match:
            fn = comment_fn_match.group(1).strip()
            if fn and "." in fn and not fn.startswith("http"):
                filename = fn

        if not filename and prec_text:
            fn_matches = re.findall(r"\b([a-zA-Z0-9_\-/\.]+\.[a-zA-Z0-9]+)\b", prec_text)
            for fn in fn_matches:
                if fn and not fn.startswith("http") and not fn.endswith(".com") and not fn.endswith(".org"):
                    filename = fn
                    break

        # Fallback intelligent filename inferencing based on code language or content heuristics
        if not filename:
            if lang in ["python", "py"] or "import " in code_content or "def " in code_content:
                filename = "app.py"
            elif lang in ["html"] or "<!DOCTYPE html>" in code_content or "<html" in code_content or "<body>" in code_content:
                filename = "index.html"
            elif lang in ["javascript", "js", "node"] or "const express" in code_content or "require(" in code_content:
                filename = "server.js"
            elif lang in ["bash", "sh", "shell"] or "#!/bin/" in code_content or "npm install" in code_content or "pip install" in code_content:
                filename = "setup.sh"
            elif lang in ["json"] or "{" in code_content:
                filename = "package.json" if "dependencies" in code_content else "config.json"
            else:
                filename = "index.html" if "<" in code_content else "app.py"

        if filename:
            existing_files = [json.loads(tc["arguments"]).get("command", "") for tc in tool_calls]
            if not any(f"> {filename}" in ef for ef in existing_files):
                dir_name = os.path.dirname(filename)
                mkdir_cmd = f"mkdir -p {dir_name} && " if dir_name else ""
                cmd = f"{mkdir_cmd}cat << 'EOF' > {filename}\n{code_content}\nEOF"
                tool_calls.append({"name": "exec", "arguments": json.dumps({"command": cmd})})

    # 4. Un-fenced HTML / Code Extraction (when models omit ``` fences)
    if not tool_calls:
        # Match all HTML documents in text
        html_matches = re.findall(r"(<!DOCTYPE html>.*?</html>|<html.*?>.*?</html>)", text, re.DOTALL | re.IGNORECASE)
        for idx, html_code in enumerate(html_matches):
            clean_html = html_code.strip()
            filename = "index.html" if idx == 0 else f"page_{idx+1}.html"
            cmd = f"cat << 'EOF' > {filename}\n{clean_html}\nEOF"
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": cmd})})

        # Match un-fenced Node / Express server code
        node_matches = re.findall(r"(const express = require.*?;.*?\napp\.listen\(.*?\);)", text, re.DOTALL)
        for node_code in node_matches:
            cmd = f"cat << 'EOF' > server.js\n{node_code.strip()}\nEOF"
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": cmd})})

        # Match un-fenced Python Flask app code
        flask_matches = re.findall(r"(from flask import.*?\napp = Flask.*?\nif __name__ == .__main__.:.*?\n\s+app\.run\(.*?\))", text, re.DOTALL)
        for py_code in flask_matches:
            cmd = f"cat << 'EOF' > app.py\n{py_code.strip()}\nEOF"
            tool_calls.append({"name": "exec", "arguments": json.dumps({"command": cmd})})

    return tool_calls

EXEC_TOOL_PREFERENCE = ("shell", "exec_command", "local_shell", "bash", "container.exec", "exec")

# Tools an agentic coding turn cannot work without, across the harnesses this
# gateway serves. Trimming by size alone would drop the shell and file tools
# (they carry the longest schemas) and keep trivia, leaving a coding agent
# that cannot read or write anything.
CORE_TOOL_NAMES = frozenset(EXEC_TOOL_PREFERENCE) | {
    # Codex CLI
    "apply_patch", "write_stdin", "update_plan", "view_image", "request_user_input",
    # Claude Code
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit", "TodoWrite",
}


def select_tools_for_budget(
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int = 1500,
    max_description_chars: int = 700,
) -> Optional[List[Dict[str, Any]]]:
    """Trim the tool payload to fit a provider's per-request token budget.

    Harnesses advertise every tool the session has, which can run to tens of
    thousands of tokens regardless of the task. Small providers reject the
    request outright, and the user sees the agent refuse to do the work.
    Core execution tools are kept unconditionally; the rest are admitted
    cheapest-first so the widest selection survives.
    """
    if not tools:
        return tools

    def cost(tool: Dict[str, Any]) -> int:
        return len(json.dumps(tool)) // 4

    trimmed = []
    for t in tools:
        fn = dict(t.get("function") or {})
        desc = fn.get("description") or ""
        if len(desc) > max_description_chars:
            fn["description"] = desc[:max_description_chars].rstrip() + "..."
        trimmed.append({**t, "function": fn})

    core = [t for t in trimmed if (t.get("function") or {}).get("name") in CORE_TOOL_NAMES]
    optional = [t for t in trimmed if (t.get("function") or {}).get("name") not in CORE_TOOL_NAMES]

    selected = list(core)
    budget = max_tokens - sum(cost(t) for t in core)
    for t in sorted(optional, key=cost):
        c = cost(t)
        if c > budget:
            continue
        selected.append(t)
        budget -= c

    if len(selected) != len(tools):
        dropped = len(tools) - len(selected)
        logger.info(f"Tool budget: kept {len(selected)} tools, dropped {dropped} to fit {max_tokens} tokens.")
    return selected or None


def adapt_tool_calls_to_schema(
    calls: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Retarget text-extracted shell calls onto the harness's real exec tool.

    `extract_tool_calls_from_text` emits a generic `exec(command=<string>)`.
    The client may instead expose `shell`, and may expect `command` as an argv
    array. Emitting a name or shape the client did not advertise makes it drop
    the call, which reads to the user as the model refusing to do the work.
    """
    if not tools or not calls:
        return calls

    by_name = {}
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            by_name[fn["name"]] = fn

    target = next((n for n in EXEC_TOOL_PREFERENCE if n in by_name), None)
    if target is None:
        return []

    params = (by_name[target].get("parameters") or {}).get("properties") or {}
    wants_argv = (params.get("command") or {}).get("type") == "array"
    arg_key = "command" if "command" in params else ("input" if "input" in params else "command")

    adapted = []
    for call in calls:
        try:
            command = json.loads(call["arguments"]).get("command", "")
        except (ValueError, KeyError):
            continue
        if not command:
            continue
        value = ["bash", "-lc", command] if wants_argv else command
        adapted.append({"name": target, "arguments": json.dumps({arg_key: value})})
    return adapted


def get_model_config(model_id: str) -> CandidateModelConfig:
    for m in settings.CANDIDATE_MODELS:
        if m.id == model_id:
            return m
    return CandidateModelConfig(
        id=model_id,
        name=model_id,
        provider="custom",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        description="Dynamic model",
        speed_tier="medium",
        intelligence_tier="medium"
    )

def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    config = get_model_config(model_id)
    input_cost = (prompt_tokens / 1000.0) * config.input_cost_per_1k
    output_cost = (completion_tokens / 1000.0) * config.output_cost_per_1k
    return round(input_cost + output_cost, 6)

def record_provider_success(model_id: str):
    PROVIDER_FAILURE_COUNTS[model_id] = 0
    cfg = settings.model_config_for(model_id)
    if cfg:
        UNAUTHENTICATED_PROVIDERS.discard(cfg.provider)

def record_provider_failure(model_id: str, error: Optional[Exception] = None):
    PROVIDER_FAILURE_COUNTS[model_id] = PROVIDER_FAILURE_COUNTS.get(model_id, 0) + 1

    # A rejected key or an exhausted balance is a property of the provider,
    # not of one model, and no number of retries will fix either. Marking the
    # provider dead takes its whole fleet out of routing on the first failure
    # instead of burning a separate circuit-breaker cycle on each of its
    # models -- which, in an agent loop, costs the user several dead turns.
    if error is not None and (_is_auth_error(error) or _is_provider_exhausted(error)):
        cfg = settings.model_config_for(model_id)
        if cfg and cfg.provider not in UNAUTHENTICATED_PROVIDERS:
            UNAUTHENTICATED_PROVIDERS.add(cfg.provider)
            reason = "rejected its credentials" if _is_auth_error(error) else "reported an exhausted quota or balance"
            logger.warning(f"Provider '{cfg.provider}' {reason}; excluding its models from routing.")

def _is_auth_error(error: Exception) -> bool:
    if isinstance(error, litellm.AuthenticationError):
        return True
    text = str(error).lower()
    return "authenticationerror" in text or "incorrect api key" in text or "no api key" in text

def _is_provider_exhausted(error: Exception) -> bool:
    """Distinguish a depleted account from an ordinary per-minute rate limit.

    Both arrive as 429s, but a per-minute limit names a retry interval and
    clears on its own, whereas a depleted balance or exhausted daily quota
    persists until someone pays or the day rolls over. Only the latter should
    take the provider out of routing.
    """
    if _retry_after_seconds(error) is not None:
        return False
    text = str(error).lower()
    return any(s in text for s in (
        "resource_exhausted",
        "credits are depleted",
        "insufficient_quota",
        "exceeded your current quota",
        "billing",
    ))

def is_circuit_open(model_id: str) -> bool:
    return PROVIDER_FAILURE_COUNTS.get(model_id, 0) >= settings.CIRCUIT_BREAKER_THRESHOLD

def is_model_routable(model_id: str, need_tools: bool = False) -> bool:
    """Whether traffic can actually be sent to this model right now.

    The router's job is to pick the cheapest capable model; this decides
    whether that pick is reachable. Without it a sound routing decision can
    still land on a model with no key, a rejected key, or no tool support,
    and the failover chain quietly undoes the routing.
    """
    cfg = settings.model_config_for(model_id)
    if cfg is None:
        return False  # not in the fleet; nothing knows how to price or reach it
    if not settings.provider_has_credentials(cfg.provider):
        return False
    if cfg.provider in UNAUTHENTICATED_PROVIDERS:
        return False
    if need_tools and not cfg.supports_tools:
        return False
    return not is_circuit_open(model_id)


# Practical per-request token ceilings. Groq's on-demand tier counts the whole
# request against a per-minute budget, so a large prompt is rejected outright
# rather than queued. Local models have no such ceiling.
PROVIDER_REQUEST_TOKEN_LIMITS = {"groq": 8000}


def estimate_request_tokens(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> int:
    total = sum(len(str(m.get("content", ""))) // 4 for m in messages if isinstance(m, dict))
    total += sum(len(json.dumps(m.get("tool_calls"))) // 4 for m in messages
                 if isinstance(m, dict) and m.get("tool_calls"))
    if tools:
        total += len(json.dumps(tools)) // 4
    return total


def model_can_accept(model_id: str, est_tokens: int) -> bool:
    cfg = settings.model_config_for(model_id)
    if cfg is None:
        return True
    limit = PROVIDER_REQUEST_TOKEN_LIMITS.get(cfg.provider)
    return limit is None or est_tokens <= limit


def pick_model_for_size(model_id: str, est_tokens: int, need_tools: bool = False) -> str:
    """Swap in a model that can physically accept this request.

    A harness system prompt can exceed a small provider's entire per-minute
    budget on its own, and no amount of history pruning fixes that. Choosing a
    model without that ceiling is the difference between the turn running and
    the turn failing.
    """
    if model_can_accept(model_id, est_tokens):
        return model_id
    for m in sorted(settings.CANDIDATE_MODELS, key=lambda c: c.input_cost_per_1k):
        if model_can_accept(m.id, est_tokens) and is_model_routable(m.id, need_tools=need_tools):
            logger.info(
                f"Request of ~{est_tokens} tokens exceeds the ceiling for '{model_id}'; using '{m.id}'."
            )
            return m.id
    return model_id

async def dispatch_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None
) -> Dict[str, Any]:
    tools = select_tools_for_budget(tools)
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    active_model = selected_model
    if is_circuit_open(selected_model):
        active_model = settings.FALLBACK_MODEL
        router_reasoning += f" | Circuit Breaker Active for {selected_model} (Failed {PROVIDER_FAILURE_COUNTS.get(selected_model)} times). Automatic failover to {active_model}."

    messages_to_send = list(original_messages)

    prompt_tokens = 0
    completion_tokens = 0
    completion_text = ""
    native_tool_calls: List[Dict[str, Any]] = []

    try:
        extra_kwargs = {}
        if active_model.startswith("ollama/"):
            extra_kwargs["api_base"] = "http://127.0.0.1:11434"
        if active_model.startswith("groq/"):
            extra_kwargs["reasoning_format"] = "hidden"
        if tools:
            extra_kwargs["tools"] = tools
            if tool_choice and tool_choice != "none":
                extra_kwargs["tool_choice"] = tool_choice
            else:
                extra_kwargs["tool_choice"] = "auto"

        response = await acompletion_with_backoff(
            model=active_model,
            messages=messages_to_send,
            temperature=0.7,
            **extra_kwargs
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        record_provider_success(active_model)
        
        choice = response.choices[0]
        completion_text = choice.message.content or ""
        native_tool_calls = [
            tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else dict(tc)
            for tc in (getattr(choice.message, "tool_calls", None) or [])
        ]

        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", len(enhanced_prompt) // 4)
            completion_tokens = getattr(usage, "completion_tokens", len(completion_text) // 4)
        else:
            prompt_tokens = len(enhanced_prompt) // 4
            completion_tokens = len(completion_text) // 4
            
    except Exception as e:
        record_provider_failure(active_model, e)
        logger.warning(f"Target model call failed for '{active_model}' ({e}). Attempting failover candidates.")
        
        fallback_candidates = [
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3.6-27b",
            # Local, so it has no shared account quota to exhaust.
            "ollama/qwen2.5-coder:7b",
        ]
        
        success = False
        for candidate in fallback_candidates:
            if candidate == active_model:
                continue
            try:
                logger.info(f"Attempting non-streaming fallback model: {candidate}")
                fb_kwargs = dict(extra_kwargs)
                response = await acompletion_with_backoff(
                    model=candidate,
                    messages=messages_to_send,
                    temperature=0.7,
                    **fb_kwargs
                )
                choice = response.choices[0]
                completion_text = choice.message.content or ""
                native_tool_calls = [
                    tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else dict(tc)
                    for tc in (getattr(choice.message, "tool_calls", None) or [])
                ]
                active_model = candidate
                success = True
                record_provider_success(candidate)
                break
            except Exception as fb_err:
                logger.warning(f"Candidate '{candidate}' failed: {fb_err}")
                
        if not success:
            completion_text = f"Processed request for: '{original_prompt}'. All configured target model providers returned rate-limit limits."
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        prompt_tokens = len(enhanced_prompt) // 4
        completion_tokens = len(completion_text) // 4

    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(active_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
    cost_savings = round(max(0.0, baseline_cost - actual_cost), 6)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=active_model,
        router_reasoning=router_reasoning,
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=cost_savings,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        completion=completion_text
    )

    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save log entry: {db_err}")

    savings_pct = round((cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0, 1)

    metrics_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "actual_cost_usd": actual_cost,
        "baseline_cost_usd": baseline_cost,
        "cost_savings_usd": cost_savings,
        "savings_percentage": savings_pct
    }

    return {
        "request_id": request_id,
        "completion": completion_text,
        "tool_calls": native_tool_calls,
        "selected_model": active_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "actual_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "cost_savings": cost_savings,
        "latency_ms": latency_ms,
        "router_reasoning": router_reasoning,
        "metrics": metrics_dict
    }

async def dispatch_streaming_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None
) -> AsyncGenerator[str, None]:
    tools = select_tools_for_budget(tools)
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    messages_to_send = list(original_messages)

    if tools:
        agentic_directive = {
            "role": "system",
            "content": (
                "CRITICAL AGENTIC SYSTEM DIRECTIVE: You are executing in an automated software development CLI harness with shell command tool capabilities ('exec'). "
                "The user wants you to create the files and build the app directly in their workspace NOW. "
                "DO NOT output conversational explanations, step-by-step tutorial guides, or markdown overviews. "
                "YOU MUST IMMEDIATELY CALL THE 'exec' TOOL (or output <exec>cat << 'EOF' > index.html ... </exec>) TO CREATE ALL NECESSARY FILES AND DIRECTORIES IN THE WORKSPACE IMMEDIATELY."
            )
        }
    if selected_model.startswith("groq/"):
        messages_to_send = prune_messages_for_token_limit(messages_to_send)

    role_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    full_completion = ""
    has_native_tool_calls = False
    try:
        extra_kwargs = {}
        if selected_model.startswith("ollama/"):
            extra_kwargs["api_base"] = "http://127.0.0.1:11434"
        if tools:
            extra_kwargs["tools"] = tools
            if tool_choice and tool_choice != "none":
                extra_kwargs["tool_choice"] = tool_choice
            else:
                extra_kwargs["tool_choice"] = "auto"

        response_stream = await acompletion_with_backoff(
            model=selected_model,
            messages=messages_to_send,
            temperature=0.7,
            stream=True,
            **extra_kwargs
        )
        async for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta_obj = choice.delta
                delta_dict = {}
                if hasattr(delta_obj, "content") and delta_obj.content:
                    full_completion += delta_obj.content
                    if not tools:
                        delta_dict["content"] = delta_obj.content

                if hasattr(delta_obj, "tool_calls") and delta_obj.tool_calls:
                    has_native_tool_calls = True
                    delta_dict["tool_calls"] = [
                        tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else dict(tc)
                        for tc in delta_obj.tool_calls
                    ]
                if hasattr(delta_obj, "role") and delta_obj.role and not tools:
                    delta_dict["role"] = delta_obj.role

                if delta_dict:
                    out_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": selected_model,
                        "choices": [{"index": 0, "delta": delta_dict, "finish_reason": choice.finish_reason}]
                    }
                    yield f"data: {json.dumps(out_chunk)}\n\n"

        # If no native tool calls were emitted and tools were requested, extract tool calls from completion text
        if tools and not has_native_tool_calls:
            extracted = adapt_tool_calls_to_schema(extract_tool_calls_from_text(full_completion), tools)
            if extracted:
                logger.info(f"Extracted {len(extracted)} tool calls from completion text stream for Chat Completions API.")
                for idx, tc in enumerate(extracted):
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": selected_model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": idx,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"]
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls" if idx == len(extracted) - 1 else None
                        }]
                    }
                    yield f"data: {json.dumps(tool_chunk)}\n\n"
            else:
                # No tool calls extracted; stream buffered text content to client
                words = full_completion.split(" ")
                for idx, word in enumerate(words):
                    delta = word if idx == len(words) - 1 else word + " "
                    out_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": selected_model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(out_chunk)}\n\n"
    except Exception as e:
        record_provider_failure(selected_model, e)
        logger.warning(f"Target streaming model call failed for '{selected_model}' ({e}). Checking exception payload for tool calls.")
        extracted_from_err = adapt_tool_calls_to_schema(extract_tool_calls_from_text(str(e)), tools)
        if tools and extracted_from_err:
            logger.info(f"Extracted {len(extracted_from_err)} tool calls from exception payload for Chat Completions API.")
            for idx, tc in enumerate(extracted_from_err):
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": selected_model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"]
                                }
                            }]
                        },
                        "finish_reason": "tool_calls" if idx == len(extracted_from_err) - 1 else None
                    }]
                }
                yield f"data: {json.dumps(tool_chunk)}\n\n"
        else:
            # Retry on a sibling model, then report the real failure. Inventing
            # a plausible-looking answer here would hand the caller code that
            # no model actually produced.
            for candidate in ("groq/openai/gpt-oss-120b", "groq/qwen/qwen3.6-27b", "ollama/qwen2.5-coder:7b"):
                if candidate == selected_model:
                    continue
                try:
                    logger.info(f"Attempting fallback model: {candidate}")
                    cand_kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
                    if candidate.startswith("ollama/"):
                        cand_kwargs["api_base"] = "http://127.0.0.1:11434"
                    retry = await acompletion_with_backoff(
                        model=candidate,
                        messages=messages_to_send,
                        temperature=0.7,
                        **cand_kwargs
                    )
                    msg = retry.choices[0].message
                    full_completion = msg.content or ""
                    native = getattr(msg, "tool_calls", None) or []
                    for idx, tc in enumerate(native):
                        tool_chunk = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": candidate,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [{
                                    "index": idx,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                                }]},
                                "finish_reason": "tool_calls" if idx == len(native) - 1 else None
                            }]
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
                    record_provider_success(candidate)
                    break
                except Exception as fb_err:
                    logger.warning(f"Candidate '{candidate}' failed: {fb_err}")
            else:
                full_completion = f"All model providers failed for this request. Last error: {e}"

            if full_completion:
                out_chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": selected_model,
                    "choices": [{"index": 0, "delta": {"content": full_completion}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(out_chunk)}\n\n"

    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    prompt_tokens = len(enhanced_prompt) // 4
    completion_tokens = len(full_completion) // 4
    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=selected_model,
        router_reasoning=router_reasoning + " | Streamed via SSE",
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=round(max(0.0, baseline_cost - actual_cost), 6),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=150.0,
        completion=full_completion
    )
    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save streaming log: {db_err}")

async def dispatch_responses_streaming_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None
) -> AsyncGenerator[str, None]:
    """
    Executes Responses API streaming events for OpenAI Codex CLI (v0.142+).
    Yields response.created, response.output_item.added, response.text.delta, response.text.done, and response.completed.
    """
    tools = select_tools_for_budget(tools)
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    resp_id = f"resp-{request_id}"
    item_id = f"item-{request_id}"

    # 1. response.created
    event_created = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "in_progress",
            "model": selected_model
        }
    }
    yield f"data: {json.dumps(event_created)}\n\n"

    # 2. response.output_item.added
    event_item = {
        "type": "response.output_item.added",
        "response_id": resp_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": ""}]
        }
    }
    yield f"data: {json.dumps(event_item)}\n\n"

    # 3. response.content_part.added
    event_part = {
        "type": "response.content_part.added",
        "response_id": resp_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""}
    }
    yield f"data: {json.dumps(event_part)}\n\n"

    messages_to_send = list(original_messages)

    if tools:
        agentic_directive = {
            "role": "system",
            "content": (
                "CRITICAL AGENTIC SYSTEM DIRECTIVE: You are executing in an automated software development CLI harness with shell command tool capabilities ('exec'). "
                "The user wants you to create the files and build the app directly in their workspace NOW. "
                "DO NOT output conversational explanations, step-by-step tutorial guides, or markdown overviews. "
                "YOU MUST IMMEDIATELY CALL THE 'exec' TOOL (or output <exec>cat << 'EOF' > index.html ... </exec>) TO CREATE ALL NECESSARY FILES AND DIRECTORIES IN THE WORKSPACE IMMEDIATELY."
            )
        }
        messages_to_send.insert(0, agentic_directive)

    if selected_model.startswith("groq/"):
        messages_to_send = prune_messages_for_token_limit(messages_to_send)

    full_completion = ""
    active_fn_calls = {}
    streamed_text = False
    try:
        extra_kwargs = {}
        if selected_model.startswith("ollama/"):
            extra_kwargs["api_base"] = "http://127.0.0.1:11434"
        if tools:
            extra_kwargs["tools"] = tools
            if tool_choice and tool_choice != "none":
                extra_kwargs["tool_choice"] = tool_choice
            else:
                extra_kwargs["tool_choice"] = "auto"

        response_stream = await acompletion_with_backoff(
            model=selected_model,
            messages=messages_to_send,
            temperature=0.7,
            stream=True,
            **extra_kwargs
        )
        async for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta_obj = chunk.choices[0].delta
                
                # A. Text delta. Streamed as it arrives and also buffered, so
                # the text fallback below can still mine it for tool calls if
                # the model never emitted a native one.
                delta_content = getattr(delta_obj, "content", "") or ""
                if delta_content:
                    full_completion += delta_content
                    streamed_text = True
                    delta_evt = {
                        "type": "response.text.delta",
                        "response_id": resp_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta_content
                    }
                    yield f"data: {json.dumps(delta_evt)}\n\n"
                    delta_evt_opt = {
                        "type": "response.output_text.delta",
                        "response_id": resp_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta_content
                    }
                    yield f"data: {json.dumps(delta_evt_opt)}\n\n"

                # B. Function Call / Tool Call Delta
                tool_calls = getattr(delta_obj, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        if idx not in active_fn_calls:
                            call_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                            fn_name = getattr(tc.function, "name", "") or ""
                            fn_item_id = f"item-fn-{request_id}-{idx}"
                            active_fn_calls[idx] = {
                                "item_id": fn_item_id,
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": ""
                            }
                            # Send output_item.added for function_call
                            event_fn_item = {
                                "type": "response.output_item.added",
                                "response_id": resp_id,
                                "output_index": idx + 1,
                                "item": {
                                    "id": fn_item_id,
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": fn_name,
                                    "arguments": ""
                                }
                            }
                            yield f"data: {json.dumps(event_fn_item)}\n\n"

                        fn_data = active_fn_calls[idx]
                        args_delta = getattr(tc.function, "arguments", "") or ""
                        if args_delta:
                            fn_data["arguments"] += args_delta
                            event_fn_delta = {
                                "type": "response.function_call_arguments.delta",
                                "response_id": resp_id,
                                "item_id": fn_data["item_id"],
                                "output_index": idx + 1,
                                "call_id": fn_data["call_id"],
                                "delta": args_delta
                            }
                            yield f"data: {json.dumps(event_fn_delta)}\n\n"

        # Finalize any active function calls
        for idx, fn_data in active_fn_calls.items():
            event_fn_done = {
                "type": "response.function_call_arguments.done",
                "response_id": resp_id,
                "item_id": fn_data["item_id"],
                "output_index": idx + 1,
                "call_id": fn_data["call_id"],
                "arguments": fn_data["arguments"]
            }
            yield f"data: {json.dumps(event_fn_done)}\n\n"

            event_fn_item_done = {
                "type": "response.output_item.done",
                "response_id": resp_id,
                "output_index": idx + 1,
                "item": {
                    "id": fn_data["item_id"],
                    "type": "function_call",
                    "call_id": fn_data["call_id"],
                    "name": fn_data["name"],
                    "arguments": fn_data["arguments"],
                    "status": "completed"
                }
            }
            yield f"data: {json.dumps(event_fn_item_done)}\n\n"

        # Dynamic conversion of text tool tags (<exec> ... </exec>) into native Responses API function_call events
        if not active_fn_calls:
            extracted = adapt_tool_calls_to_schema(extract_tool_calls_from_text(full_completion), tools)
            if extracted:
                logger.info(f"Extracted {len(extracted)} tool calls from text stream for Codex CLI execution.")
                for idx, tc in enumerate(extracted):
                    fn_item_id = f"item-fn-{request_id}-{idx+1}"
                    fn_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    
                    event_fn_item = {
                        "type": "response.output_item.added",
                        "response_id": resp_id,
                        "output_index": idx + 1,
                        "item": {
                            "id": fn_item_id,
                            "type": "function_call",
                            "call_id": fn_call_id,
                            "name": tc["name"],
                            "arguments": ""
                        }
                    }
                    yield f"data: {json.dumps(event_fn_item)}\n\n"

                    event_fn_delta = {
                        "type": "response.function_call_arguments.delta",
                        "response_id": resp_id,
                        "item_id": fn_item_id,
                        "output_index": idx + 1,
                        "call_id": fn_call_id,
                        "delta": tc["arguments"]
                    }
                    yield f"data: {json.dumps(event_fn_delta)}\n\n"

                    event_fn_done = {
                        "type": "response.function_call_arguments.done",
                        "response_id": resp_id,
                        "item_id": fn_item_id,
                        "output_index": idx + 1,
                        "call_id": fn_call_id,
                        "arguments": tc["arguments"]
                    }
                    yield f"data: {json.dumps(event_fn_done)}\n\n"

                    event_fn_item_done = {
                        "type": "response.output_item.done",
                        "response_id": resp_id,
                        "output_index": idx + 1,
                        "item": {
                            "id": fn_item_id,
                            "type": "function_call",
                            "call_id": fn_call_id,
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                            "status": "completed"
                        }
                    }
                    yield f"data: {json.dumps(event_fn_item_done)}\n\n"
            elif not streamed_text and full_completion:
                # Nothing reached the client yet, so flush the buffer.
                delta_evt = {
                    "type": "response.text.delta",
                    "response_id": resp_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": full_completion
                }
                yield f"data: {json.dumps(delta_evt)}\n\n"
                delta_evt_opt = {
                    "type": "response.output_text.delta",
                    "response_id": resp_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": full_completion
                }
                yield f"data: {json.dumps(delta_evt_opt)}\n\n"

    except Exception as e:
        record_provider_failure(selected_model, e)
        logger.warning(f"Target model call failed for '{selected_model}' ({e}). Attempting failover candidates.")
        
        fallback_candidates = [
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3.6-27b",
            # Local, so it has no shared account quota to exhaust.
            "ollama/qwen2.5-coder:7b",
        ]
        
        success = False
        for candidate in fallback_candidates:
            if candidate == selected_model:
                continue
            try:
                logger.info(f"Attempting fallback model: {candidate}")
                cand_messages = list(messages_to_send)
                if candidate.startswith("groq/"):
                    cand_messages = prune_messages_for_token_limit(cand_messages)

                cand_kwargs = {}
                if candidate.startswith("ollama/"):
                    cand_kwargs["api_base"] = "http://127.0.0.1:11434"
                if tools and (not candidate.startswith("groq/") or "qwen" in candidate.lower()) and not "gpt-oss" in candidate.lower():
                    cand_kwargs["tools"] = tools
                    if tool_choice and tool_choice != "none":
                        cand_kwargs["tool_choice"] = tool_choice
                    else:
                        cand_kwargs["tool_choice"] = "auto"

                fallback_stream = await acompletion_with_backoff(
                    model=candidate,
                    messages=cand_messages,
                    temperature=0.7,
                    stream=True,
                    **cand_kwargs
                )
                async for chunk in fallback_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_obj = chunk.choices[0].delta
                        delta_content = getattr(delta_obj, "content", "") or ""
                        if delta_content:
                            full_completion += delta_content
                            delta_evt = {
                                "type": "response.text.delta",
                                "response_id": resp_id,
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": delta_content
                            }
                            yield f"data: {json.dumps(delta_evt)}\n\n"
                            delta_evt_opt = {
                                "type": "response.output_text.delta",
                                "response_id": resp_id,
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": delta_content
                            }
                            yield f"data: {json.dumps(delta_evt_opt)}\n\n"

                        tool_calls = getattr(delta_obj, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                idx = getattr(tc, "index", 0) or 0
                                if idx not in active_fn_calls:
                                    call_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                                    fn_name = getattr(tc.function, "name", "") or ""
                                    fn_item_id = f"item-fn-{request_id}-{idx}"
                                    active_fn_calls[idx] = {
                                        "item_id": fn_item_id,
                                        "call_id": call_id,
                                        "name": fn_name,
                                        "arguments": ""
                                    }
                                    event_fn_item = {
                                        "type": "response.output_item.added",
                                        "response_id": resp_id,
                                        "output_index": idx + 1,
                                        "item": {
                                            "id": fn_item_id,
                                            "type": "function_call",
                                            "call_id": call_id,
                                            "name": fn_name,
                                            "arguments": ""
                                        }
                                    }
                                    yield f"data: {json.dumps(event_fn_item)}\n\n"

                                fn_data = active_fn_calls[idx]
                                args_delta = getattr(tc.function, "arguments", "") or ""
                                if args_delta:
                                    fn_data["arguments"] += args_delta
                                    event_fn_delta = {
                                        "type": "response.function_call_arguments.delta",
                                        "response_id": resp_id,
                                        "item_id": fn_data["item_id"],
                                        "output_index": idx + 1,
                                        "call_id": fn_data["call_id"],
                                        "delta": args_delta
                                    }
                                    yield f"data: {json.dumps(event_fn_delta)}\n\n"
                record_provider_success(candidate)
                success = True
                break
            except Exception as candidate_err:
                logger.warning(f"Candidate '{candidate}' failed: {candidate_err}")
                continue

        if not success:
            logger.error("All model candidates exhausted. Generating dynamic response.")
            full_completion = f"Processed request for: '{original_prompt}'. All configured target model providers returned rate-limit limits."
            words = full_completion.split(" ")
            for idx, word in enumerate(words):
                delta = word if idx == len(words) - 1 else word + " "
                delta_evt = {
                    "type": "response.text.delta",
                    "response_id": resp_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta
                }
                yield f"data: {json.dumps(delta_evt)}\n\n"

        # Dynamic conversion of text tool tags (<exec> ... </exec>) into native Responses API function_call events
        if not active_fn_calls:
            extracted = adapt_tool_calls_to_schema(extract_tool_calls_from_text(full_completion), tools)
            if extracted:
                logger.info(f"Extracted {len(extracted)} tool calls from text stream for Codex CLI execution.")
                for idx, tc in enumerate(extracted):
                    fn_item_id = f"item-fn-{request_id}-{idx+1}"
                    fn_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    
                    event_fn_item = {
                        "type": "response.output_item.added",
                        "response_id": resp_id,
                        "output_index": idx + 1,
                        "item": {
                            "id": fn_item_id,
                            "type": "function_call",
                            "call_id": fn_call_id,
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    yield f"data: {json.dumps(event_fn_item)}\n\n"

                    event_fn_done = {
                        "type": "response.function_call_arguments.done",
                        "response_id": resp_id,
                        "item_id": fn_item_id,
                        "output_index": idx + 1,
                        "call_id": fn_call_id,
                        "arguments": tc["arguments"]
                    }
                    yield f"data: {json.dumps(event_fn_done)}\n\n"

                    event_fn_item_done = {
                        "type": "response.output_item.done",
                        "response_id": resp_id,
                        "output_index": idx + 1,
                        "item": {
                            "id": fn_item_id,
                            "type": "function_call",
                            "call_id": fn_call_id,
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                            "status": "completed"
                        }
                    }
                    yield f"data: {json.dumps(event_fn_item_done)}\n\n"

    # 4. response.text.done & response.output_text.done
    event_text_done = {
        "type": "response.text.done",
        "response_id": resp_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_completion
    }
    yield f"data: {json.dumps(event_text_done)}\n\n"

    event_output_text_done = {
        "type": "response.output_text.done",
        "response_id": resp_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_completion
    }
    yield f"data: {json.dumps(event_output_text_done)}\n\n"

    # 5. response.content_part.done
    event_part_done = {
        "type": "response.content_part.done",
        "response_id": resp_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": full_completion}
    }
    yield f"data: {json.dumps(event_part_done)}\n\n"

    # 6. response.output_item.done
    event_item_done = {
        "type": "response.output_item.done",
        "response_id": resp_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": full_completion}]
        }
    }
    yield f"data: {json.dumps(event_item_done)}\n\n"

    # 7. response.completed
    event_completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "completed",
            "model": selected_model,
            "output": [
                {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_completion}]
                }
            ]
        }
    }
    yield f"data: {json.dumps(event_completed)}\n\n"
    yield "data: [DONE]\n\n"

    # Log metrics to DB
    prompt_tokens = len(enhanced_prompt) // 4
    completion_tokens = len(full_completion) // 4
    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=selected_model,
        router_reasoning=router_reasoning + " | Responses API Streamed",
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=round(max(0.0, baseline_cost - actual_cost), 6),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=150.0,
        completion=full_completion
    )
    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save responses streaming log: {db_err}")


async def dispatch_anthropic_streaming_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """Stream an Anthropic Messages response for Claude Code.

    Text and tool calls are separate indexed content blocks in this protocol,
    so a tool call has to open its own block and stream its arguments as
    `input_json_delta`. Emitting it as text would make the client display the
    call instead of executing it.
    """
    from services.anthropic_api import sse

    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{request_id}"

    messages_to_send = prune_messages_for_token_limit(list(original_messages))

    # Choose the model against the untrimmed request, then trim only if the
    # chosen model actually has a ceiling. Trimming first would discard tools
    # to fit a provider we are not going to use.
    selected_model = pick_model_for_size(
        selected_model, estimate_request_tokens(messages_to_send, tools), need_tools=bool(tools)
    )
    cfg = settings.model_config_for(selected_model)
    ceiling = PROVIDER_REQUEST_TOKEN_LIMITS.get(cfg.provider) if cfg else None
    if ceiling is not None:
        overhead = estimate_request_tokens(messages_to_send, None)
        tools = select_tools_for_budget(tools, max_tokens=max(800, ceiling - overhead - 500))

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": selected_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": max(1, len(enhanced_prompt) // 4), "output_tokens": 0},
        },
    })

    block_index = 0
    text_block_open = False
    full_completion = ""
    active_tools: Dict[int, Dict[str, Any]] = {}
    stop_reason = "end_turn"

    async def open_text_block():
        return sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })

    try:
        extra_kwargs: Dict[str, Any] = {}
        if selected_model.startswith("ollama/"):
            extra_kwargs["api_base"] = "http://127.0.0.1:11434"
        if tools:
            extra_kwargs["tools"] = tools
            extra_kwargs["tool_choice"] = tool_choice if (tool_choice and tool_choice != "none") else "auto"
        if max_tokens:
            extra_kwargs["max_tokens"] = max_tokens

        stream = await acompletion_with_backoff(
            model=selected_model,
            messages=messages_to_send,
            temperature=0.7,
            stream=True,
            **extra_kwargs
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            text = getattr(delta, "content", "") or ""
            if text:
                if not text_block_open:
                    yield await open_text_block()
                    text_block_open = True
                full_completion += text
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                })

            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", 0) or 0
                if idx not in active_tools:
                    # Text and tool blocks cannot share an index.
                    if text_block_open:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        text_block_open = False
                        block_index += 1
                    tool_id = getattr(tc, "id", None) or f"toolu_{uuid.uuid4().hex[:16]}"
                    name = getattr(tc.function, "name", "") or ""
                    active_tools[idx] = {"block": block_index, "id": tool_id, "name": name}
                    yield sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
                    })
                    stop_reason = "tool_use"

                args = getattr(tc.function, "arguments", "") or ""
                if args:
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": active_tools[idx]["block"],
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    })

        if text_block_open:
            yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        for meta in active_tools.values():
            yield sse("content_block_stop", {"type": "content_block_stop", "index": meta["block"]})

    except Exception as e:
        record_provider_failure(selected_model, e)
        logger.warning(f"Anthropic streaming failed for '{selected_model}' ({e}). Attempting failover.")

        est = estimate_request_tokens(messages_to_send, tools)
        candidates = [c for c in (
            settings.AGENTIC_MODEL,
            "groq/qwen/qwen3.6-27b",
            "ollama/qwen2.5-coder:7b",
        ) if c != selected_model and model_can_accept(c, est)]

        recovered = False
        for candidate in candidates:
            try:
                logger.info(f"Attempting fallback model: {candidate}")
                cand_kwargs: Dict[str, Any] = {}
                if candidate.startswith("ollama/"):
                    cand_kwargs["api_base"] = "http://127.0.0.1:11434"
                if tools:
                    cand_kwargs["tools"] = tools
                    cand_kwargs["tool_choice"] = "auto"
                if max_tokens:
                    cand_kwargs["max_tokens"] = max_tokens
                retry = await acompletion_with_backoff(
                    model=candidate, messages=messages_to_send, temperature=0.7, **cand_kwargs
                )
                msg_obj = retry.choices[0].message
                text = msg_obj.content or ""
                calls = getattr(msg_obj, "tool_calls", None) or []

                if text:
                    if not text_block_open:
                        yield await open_text_block()
                        text_block_open = True
                    full_completion += text
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })
                if text_block_open:
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    text_block_open = False

                for call in calls:
                    block_index += 1
                    yield sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.id or f"toolu_{uuid.uuid4().hex[:16]}",
                            "name": call.function.name,
                            "input": {},
                        },
                    })
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": call.function.arguments or "{}"},
                    })
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    stop_reason = "tool_use"

                record_provider_success(candidate)
                selected_model = candidate
                recovered = True
                break
            except Exception as fb_err:
                logger.warning(f"Candidate '{candidate}' failed: {fb_err}")

        if not recovered:
            if not text_block_open and not active_tools:
                yield await open_text_block()
                text_block_open = True
            msg = f"WormHole could not complete this request: {e}"
            full_completion += msg
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": msg},
            })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": max(1, len(full_completion) // 4)},
    })
    yield sse("message_stop", {"type": "message_stop"})

    _log_inference(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=selected_model,
        router_reasoning=router_reasoning + " | Anthropic Messages Streamed",
        completion=full_completion,
    )


def _log_inference(request_id, original_prompt, enhanced_prompt, enhancer_model,
                   router_model, selected_model, router_reasoning, completion):
    prompt_tokens = len(enhanced_prompt) // 4
    completion_tokens = len(completion) // 4
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
    try:
        with Session(engine) as session:
            session.add(InferenceLog(
                request_id=request_id,
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=enhancer_model,
                router_model=router_model,
                selected_model=selected_model,
                router_reasoning=router_reasoning,
                actual_cost=actual_cost,
                baseline_cost=baseline_cost,
                cost_savings=round(max(0.0, baseline_cost - actual_cost), 6),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=150.0,
                completion=completion,
            ))
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save inference log: {db_err}")
