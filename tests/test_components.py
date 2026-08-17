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
    assert result["metrics"]["cost_savings_usd"] >= 0.0
    
    # Run Judge evaluation
    req_id = result["request_id"]
    score, feedback = await evaluate_completion(req_id, enhanced, result["completion"])
    assert 1.0 <= score <= 10.0
    assert len(feedback) > 0

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
