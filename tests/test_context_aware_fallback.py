import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from ai_council.core.models import (
    Subtask, TaskType, ModelCapabilities, RiskLevel, Priority, AgentResponse
)
from ai_council.routing.context_protocol import ModelContextProtocolImpl
from ai_council.routing.registry import ModelRegistryImpl
from ai_council.core.interfaces import AIModel

class MockModel(AIModel):
    def __init__(self, model_id):
        self.model_id = model_id
    def get_model_id(self):
        return self.model_id
    async def generate_response(self, prompt, **kwargs):
        return f"Response from {self.model_id}"

@pytest.fixture
def model_registry():
    registry = ModelRegistryImpl()
    
    # Model 1: High reasoning, strict safety
    model1 = MockModel("model-high-reasoning-strict-safety")
    caps1 = ModelCapabilities(
        task_types=[TaskType.REASONING, TaskType.CODE_GENERATION],
        reliability_score=0.9,
        average_latency=2.0,
        tags=["high-reasoning", "strict-safety", "premium"]
    )
    registry.register_model(model1, caps1)
    
    # Model 2: Weak reasoning, relaxed safety
    model2 = MockModel("model-weak-reasoning-relaxed-safety")
    caps2 = ModelCapabilities(
        task_types=[TaskType.REASONING, TaskType.RESEARCH],
        reliability_score=0.8,
        average_latency=1.0,
        tags=["weak-reasoning", "relaxed-safety"]
    )
    registry.register_model(model2, caps2)
    
    # Model 3: Balanced
    model3 = MockModel("model-balanced")
    caps3 = ModelCapabilities(
        task_types=[TaskType.REASONING, TaskType.RESEARCH],
        reliability_score=0.85,
        average_latency=1.5,
        tags=["balanced"]
    )
    registry.register_model(model3, caps3)
    
    return registry

@pytest.mark.asyncio
async def test_fallback_reasoning_aware(model_registry):
    protocol = ModelContextProtocolImpl(model_registry)
    subtask = Subtask(content="Complex reasoning task", task_type=TaskType.REASONING)
    
    # Simulate a "reasoning failed" context
    failure_context = {
        "failure_type": "api_failure",
        "error_message": "insufficient reasoning depth"
    }
    
    # Primary model was model-balanced
    selection = await protocol.select_fallback("model-balanced", subtask, failure_context)
    
    # Should select high-reasoning model
    assert selection.model_id == "model-high-reasoning-strict-safety"
    assert "high-reasoning" in selection.reasoning.lower()

@pytest.mark.asyncio
async def test_fallback_safety_aware(model_registry):
    protocol = ModelContextProtocolImpl(model_registry)
    subtask = Subtask(content="Potential safety sensitive task", task_type=TaskType.RESEARCH)
    
    # Simulate a "content filter" context from a strict model
    failure_context = {
        "failure_type": "validation_error",
        "error_message": "content_filter triggered"
    }
    
    # Primary model was model-high-reasoning-strict-safety
    selection = await protocol.select_fallback("model-high-reasoning-strict-safety", subtask, failure_context)
    
    # Should potentially avoid strict-safety if it failed due to content filter and prefer relaxed-safety
    assert selection.model_id == "model-weak-reasoning-relaxed-safety"
    assert "chose" in selection.reasoning.lower()

@pytest.mark.asyncio
async def test_fallback_no_context(model_registry):
    protocol = ModelContextProtocolImpl(model_registry)
    subtask = Subtask(content="General task", task_type=TaskType.REASONING)
    
    # No context provided
    selection = await protocol.select_fallback("model-balanced", subtask, None)
    
    # Should default to highest reliable remaining model
    assert selection.model_id == "model-high-reasoning-strict-safety"
    assert "no failure context" in selection.reasoning.lower() or "score" in selection.reasoning.lower()

@pytest.mark.asyncio
async def test_general_scoring_with_tags(model_registry):
    protocol = ModelContextProtocolImpl(model_registry)
    subtask = Subtask(content="High accuracy requirement", task_type=TaskType.REASONING, accuracy_requirement=0.95)
    
    # Routing (not fallback)
    selection = await protocol.route_task(subtask)
    
    # Should prefer "premium" tag for high accuracy requirement if configured
    assert selection.model_id == "model-high-reasoning-strict-safety"
    assert "premium" in protocol.model_registry.get_model_capabilities(selection.model_id).tags
