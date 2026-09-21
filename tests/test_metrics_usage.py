"""Unit tests for LLM usage aggregation, incl. prompt-cache creation tokens."""

from __future__ import annotations

import pytest

from livekit.agents.metrics import (
    LLMMetrics,
    LLMModelUsage,
    ModelUsageCollector,
    RealtimeModelMetrics,
)
from livekit.agents.metrics.base import Metadata

pytestmark = pytest.mark.unit


def _llm_metrics(**overrides: object) -> LLMMetrics:
    base: dict[str, object] = {
        "label": "test.LLM",
        "request_id": "req-1",
        "timestamp": 0.0,
        "duration": 1.0,
        "ttft": 0.1,
        "cancelled": False,
        "completion_tokens": 10,
        "prompt_tokens": 100,
        "prompt_cached_tokens": 20,
        "total_tokens": 110,
        "tokens_per_second": 10.0,
        "metadata": Metadata(model_provider="anthropic", model_name="claude-sonnet-4"),
    }
    base.update(overrides)
    return LLMMetrics(**base)


def test_llm_metrics_defaults_cache_creation_to_zero() -> None:
    m = _llm_metrics()
    assert m.cache_creation_tokens == 0


def test_llm_metrics_carries_cache_creation_tokens() -> None:
    m = _llm_metrics(cache_creation_tokens=42)
    assert m.cache_creation_tokens == 42


def test_collector_aggregates_cache_creation_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(cache_creation_tokens=42))
    collector.collect(_llm_metrics(cache_creation_tokens=8))

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.input_cache_creation_tokens == 50


def test_llm_metrics_defaults_reasoning_to_zero() -> None:
    m = _llm_metrics()
    assert m.reasoning_tokens == 0


def test_llm_metrics_carries_reasoning_tokens() -> None:
    m = _llm_metrics(reasoning_tokens=64)
    assert m.reasoning_tokens == 64


def test_collector_aggregates_reasoning_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(completion_tokens=100, reasoning_tokens=64))
    collector.collect(_llm_metrics(completion_tokens=50, reasoning_tokens=8))

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.output_reasoning_tokens == 72
    # reasoning is a subset of the output tokens, never added on top of them
    assert llm_usage.output_tokens == 150


def _realtime_metrics(**overrides: object) -> RealtimeModelMetrics:
    base: dict[str, object] = {
        "label": "test.RealtimeModel",
        "request_id": "resp-1",
        "timestamp": 0.0,
        "duration": 1.0,
        "ttft": 0.1,
        "cancelled": False,
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "tokens_per_second": 30.0,
        "input_token_details": RealtimeModelMetrics.InputTokenDetails(text_tokens=100),
        "output_token_details": RealtimeModelMetrics.OutputTokenDetails(
            text_tokens=10, audio_tokens=20
        ),
        "metadata": Metadata(model_provider="google", model_name="gemini-live"),
    }
    base.update(overrides)
    return RealtimeModelMetrics(**base)


def test_realtime_output_token_details_default_reasoning_to_zero() -> None:
    details = RealtimeModelMetrics.OutputTokenDetails(text_tokens=10)
    assert details.reasoning_tokens == 0


def test_collector_aggregates_realtime_reasoning_tokens() -> None:
    """A thinking realtime model bills hidden reasoning like any other output.

    Same invariant the LLM path holds above: reasoning is a subset of
    ``output_tokens``, so a provider whose response count excludes it has to
    add it in before reporting.
    """
    collector = ModelUsageCollector()
    collector.collect(
        _realtime_metrics(
            output_tokens=830,
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=10, audio_tokens=20, reasoning_tokens=800
            ),
        )
    )
    collector.collect(
        _realtime_metrics(
            output_tokens=130,
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=10, audio_tokens=20, reasoning_tokens=100
            ),
        )
    )

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.output_reasoning_tokens == 900
    assert llm_usage.output_tokens == 960
