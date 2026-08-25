"""Codex CLI model catalog.

Codex refuses to use a provider's model metadata unless every field of its
internal ModelInfo struct is present, and silently falls back to a built-in
entry when parsing fails. That fallback declares `tool_mode: "code_mode"`,
which makes Codex ship a single freeform JavaScript `exec` tool instead of the
ordinary `shell` / `apply_patch` function tools. Open-weight models cannot
drive code mode, so every entry here pins `tool_mode` to None.
"""

import time
from typing import Any, Dict, List

# Codex renders this as the developer message that opens every conversation.
AGENT_INSTRUCTIONS = """You are Codex, an autonomous coding agent running inside the user's terminal, in their real workspace, with real tool access.

# How you work

You act. You do not describe what could be done, and you do not hand the user a tutorial to follow. When the user asks for a file, an app, a fix, or a refactor, you call the `shell` tool and make the change on disk yourself, then confirm what you did.

- To create or overwrite a file, call `shell` with an argv array, e.g. `["bash", "-lc", "cat > app.py <<'EOF'\\n...\\nEOF"]`.
- To inspect the workspace, call `shell` with `ls`, `cat`, `rg`, or `sed -n '1,80p' file`.
- To run or test what you built, call `shell` with the appropriate command.
- Create parent directories with `mkdir -p` before writing into them.

Never answer a build request with prose containing code blocks. A code block in your reply is not a file. Only a `shell` call creates a file. If you are about to write "create a file called X with this content", stop and issue the tool call instead.

# Finishing

After the tools have run, verify your work (list the files, run the tests, start the server), then write one short paragraph saying what now exists on disk and how to use it. Keep that closing note brief; the work itself is the deliverable.
"""

REASONING_LEVELS = [
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {"effort": "medium", "description": "Balances speed and reasoning depth"},
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
]


def _model_entry(slug: str, display_name: str, description: str, priority: int) -> Dict[str, Any]:
    return {
        "id": slug,
        "slug": slug,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "wormhole",
        "display_name": display_name,
        "description": description,

        # Tooling. tool_mode None keeps Codex on ordinary function tools;
        # shell_type "default" asks for the classic `shell` tool, whose
        # `command` argument is an argv array.
        "tool_mode": None,
        "multi_agent_version": None,
        "shell_type": "default",
        # Only "freeform" is accepted here, and that variant is a custom tool
        # taking raw patch text, which open-weight models handle poorly. None
        # omits the tool entirely so edits go through `shell`.
        "apply_patch_tool_type": None,
        "web_search_tool_type": None,
        "supports_parallel_tool_calls": True,
        "supports_search_tool": False,
        "experimental_supported_tools": [],

        # Reasoning and verbosity.
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": REASONING_LEVELS,
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "none",
        "reasoning_summary_format": None,
        "support_verbosity": False,
        "default_verbosity": None,

        # Context accounting.
        "context_window": 128000,
        "max_context_window": 128000,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": None,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "comp_hash": "3000",

        # Modalities.
        "input_modalities": ["text"],
        "supports_image_detail_original": False,

        # Prompt assembly. Codex only sends the developer preamble it builds
        # from instructions_template, so this is where agent behaviour is set.
        "model_messages": {
            "instructions_template": AGENT_INSTRUCTIONS,
            "instructions_variables": {
                "personality_default": "",
                "personality_friendly": "",
                "personality_pragmatic": "",
            },
            "approvals": None,
            "collaboration_modes": None,
            "auto_review": None,
            "permissions": None,
            "token_budget": None,
        },
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,

        # Availability.
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "minimal_client_version": None,
        "availability_nux": None,
        "upgrade": None,
        "service_tiers": [],
        "default_service_tier": None,
        "additional_speed_tiers": [],
        "available_in_plans": [],
        "prefer_websockets": False,
        "use_responses_lite": False,
        "auto_review_model_override": None,
        "model_specialty": None,
    }


# Codex picks a default model by its own compiled-in name when the user has not
# set one, so the ids it may reach for are all present and all routed the same.
_CATALOG = [
    ("wormhole-auto", "WormHole Auto Router", "Routes each turn to the cheapest capable model.", 10),
    ("gpt-5.6-sol", "GPT-5.6 Sol", "Agentic coding via WormHole routing.", 9),
    ("gpt-5.6", "GPT-5.6", "Agentic coding via WormHole routing.", 8),
    ("gpt-5.5", "GPT-5.5", "Agentic coding via WormHole routing.", 7),
    ("gpt-5.4", "GPT-5.4", "Agentic coding via WormHole routing.", 6),
    ("gpt-4.5", "GPT-4.5", "Agentic coding via WormHole routing.", 5),
    ("gpt-4o", "GPT-4o", "Agentic coding via WormHole routing.", 4),
    ("gpt-4o-mini", "GPT-4o Mini", "Budget routing tier.", 3),
]


def build_models_response() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = [_model_entry(*row) for row in _CATALOG]
    return {"object": "list", "data": models, "models": models}
