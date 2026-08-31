import json
import pytest
import asyncio
from config import settings
from db.database import init_db
from services.enhancer import enhance_prompt
from services.router import route_prompt
from services.dispatcher import dispatch_inference, calculate_cost
from services.judge import evaluate_completion
from services.dataset import export_router_dataset, export_enhancer_dataset

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    # Availability state is process-global. Without clearing it, a rate limit
    # hit by one test takes models out of routing for the ones that follow,
    # and they fail for reasons unrelated to what they are testing.
    from services import dispatcher
    dispatcher.MODEL_COOLDOWN.clear()
    dispatcher.PROVIDER_FAILURE_COUNTS.clear()
    dispatcher.UNAUTHENTICATED_PROVIDERS.clear()

@pytest.mark.asyncio
async def test_cost_calculation():
    # Test GPT-4o Mini calculation: 1000 input ($0.00015) + 1000 output ($0.0006) = $0.00075
    cost = calculate_cost("gpt-4o-mini", 1000, 1000)
    assert cost == 0.00075
    
    # Test GPT-4o calculation: 1000 input ($0.0025) + 1000 output ($0.0100) = $0.0125
    baseline = calculate_cost("gpt-4o", 1000, 1000)
    assert baseline == 0.0125
    assert baseline > cost

@pytest.mark.asyncio
async def test_enhancer_service():
    original = "Write a function to compute fibonacci."
    enhanced = await enhance_prompt(original)
    assert len(enhanced) > len(original)
    assert "fibonacci" in enhanced.lower() or "objective" in enhanced.lower()

@pytest.mark.asyncio
async def test_router_service():
    prompt = "Write a simple Python script to sort an array."
    selected_model, reasoning = await route_prompt(prompt)
    assert selected_model in [m.id for m in settings.CANDIDATE_MODELS]
    assert len(reasoning) > 0

@pytest.mark.asyncio
async def test_dispatch_and_judge_flow():
    original = "Summarize meeting notes."
    enhanced = await enhance_prompt(original)
    selected_model = "gpt-4o-mini"
    reasoning = "Test low cost model selection"
    
    result = await dispatch_inference(
        original_prompt=original,
        enhanced_prompt=enhanced,
        enhancer_model=settings.ENHANCER_MODEL,
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=reasoning,
        original_messages=[{"role": "user", "content": original}]
    )
    
    assert "request_id" in result
    # Assert on observed usage rather than a savings figure, which is priced
    # against a model that never ran.
    assert result["metrics"]["actual_cost_usd"] >= 0.0
    assert result["metrics"]["total_tokens"] > 0
    
    # Run Judge evaluation
    req_id = result["request_id"]
    score, feedback = await evaluate_completion(req_id, enhanced, result["completion"])
    # The judge returns None when it could not evaluate, rather than inventing
    # a score. That is a provider availability outcome, not a flow failure.
    if score is not None:
        assert 1.0 <= score <= 10.0
    assert isinstance(feedback, str)

@pytest.mark.asyncio
async def test_dataset_exporter():
    router_data = export_router_dataset()
    enhancer_data = export_enhancer_dataset()
    assert isinstance(router_data, list)
    assert isinstance(enhancer_data, list)

@pytest.mark.asyncio
async def test_extract_tool_calls_from_text():
    from services.dispatcher import extract_tool_calls_from_text
    sample = "<exec> touch app.py </exec>\n<exec> echo \"hello\" > app.py </exec>"
    tool_calls = extract_tool_calls_from_text(sample)
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "exec"
    assert "touch app.py" in tool_calls[0]["arguments"]
    assert tool_calls[1]["name"] == "exec"
    assert "echo" in tool_calls[1]["arguments"]

    sample_md = "**index.html:**\n\n```html\n<h1>Hello</h1>\n```\n\n**script.js:**\n\n```javascript\nconsole.log(1);\n```"
    tool_calls_md = extract_tool_calls_from_text(sample_md)
    assert len(tool_calls_md) == 2
    assert "index.html" in tool_calls_md[0]["arguments"]
    assert "script.js" in tool_calls_md[1]["arguments"]

    sample_paren = "(exec) flask new app"
    tool_calls_paren = extract_tool_calls_from_text(sample_paren)
    assert len(tool_calls_paren) == 1
    assert "flask new app" in tool_calls_paren[0]["arguments"]

@pytest.mark.asyncio
async def test_extract_tool_calls_command_execution_and_file_creation():
    from services.dispatcher import extract_tool_calls_from_text
    sample_code = """I will create the Flask app now.

```python
# app.py
from flask import Flask
app = Flask(__name__)
```

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html><body><h1>Image Viewer</h1></body></html>
```
"""
    tool_calls = extract_tool_calls_from_text(sample_code)
    assert len(tool_calls) == 2
    cmds = [json.loads(tc["arguments"])["command"] for tc in tool_calls]
    assert any("app.py" in c for c in cmds)
    assert any("templates/index.html" in c for c in cmds)

@pytest.mark.asyncio
async def test_slm_model_suggestion_and_routing():
    import json
    from services.router import route_prompt
    
    # 1. Test simple prompt routes to low cost model
    simple_prompt = "Write a basic Python function to check if a number is prime."
    model1, reasoning1 = await route_prompt(simple_prompt)
    assert model1 in [m.id for m in settings.CANDIDATE_MODELS]
    assert len(reasoning1) > 0

    # 2. Test complex reasoning prompt routes to high intelligence model
    complex_prompt = "Given an array of integers, find all unique quad tuples that sum to target using O(N^3) optimization and formal proof of quantum correctness for autonomous enterprise system architecture."
    model2, reasoning2 = await route_prompt(complex_prompt)
    assert model2 in [m.id for m in settings.CANDIDATE_MODELS]
    # Assert the capability tier rather than specific ids, so the test states
    # the actual requirement and survives changes to the fleet.
    tier = settings.model_config_for(model2).intelligence_tier
    assert tier in ("high", "frontier"), f"complex prompt routed to {model2} (tier={tier})"

@pytest.mark.asyncio
async def test_min_routing_tier_excludes_weaker_models():
    """The floor must hold regardless of what the router would prefer.

    A model can call tools correctly and still be unable to drive an agentic
    harness, so supports_tools is not a sufficient guard on its own. This is
    the setting that keeps those tiers out of the pool entirely.
    """
    from services.dispatcher import is_model_routable

    original = settings.MIN_ROUTING_TIER
    # This test is about the tier floor, so the other filters that can also
    # make a model unroutable have to be out of the way. Without this the
    # result depends on whatever ROUTING_MODELS happens to be in the
    # developer's own .env, and the test fails for a reason that has nothing
    # to do with what it is checking.
    original_models = settings.ROUTING_MODELS
    original_providers = settings.ROUTING_PROVIDERS
    try:
        settings.ROUTING_MODELS = []
        settings.ROUTING_PROVIDERS = []
        settings.MIN_ROUTING_TIER = ""
        assert is_model_routable("gpt-5-nano"), "basic tier should be routable with no floor"

        settings.MIN_ROUTING_TIER = "medium"
        assert not is_model_routable("gpt-5-nano"), "basic tier must be excluded by a medium floor"
        assert is_model_routable("gpt-4o-mini"), "medium tier must survive its own floor"
        assert is_model_routable("gpt-4o"), "frontier tier must survive a medium floor"

        settings.MIN_ROUTING_TIER = "frontier"
        assert not is_model_routable("gpt-4o-mini")
        assert is_model_routable("gpt-4o")

        # An unusable floor must not empty the fleet: a typo that silently
        # blocked every model would take the gateway down rather than degrade.
        settings.MIN_ROUTING_TIER = "enormous"
        assert is_model_routable("gpt-4o-mini"), "an unrecognised floor must be ignored, not fatal"
    finally:
        settings.MIN_ROUTING_TIER = original
        settings.ROUTING_MODELS = original_models
        settings.ROUTING_PROVIDERS = original_providers
