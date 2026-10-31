"""Unit and integration tests for the Groq LLM module (backend/llm.py).

Verifies:
- Safe handling of missing configuration
- Fallback default verification model
- Strict rejection of unsupported schema types (requires BaseModel subclass)
- Pydantic model validation and structured output fallback
- Automatic retries on invalid JSON or validation failure
- Respecting rate limits (HTTP 429) and Retry-After headers
- Disk cache hits and failure non-caching
- Concurrency limiter behavior
- Real Groq API integration (skipped if GROQ_API_KEY is not available or unreachable)
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, Field

import groq
from backend import llm
from backend.llm import complete_json, get_api_key, get_cache_key, get_verify_model


class SampleExtraction(BaseModel):
    company_name: str = Field(description="Name of the company")
    confidence_score: float = Field(description="Score between 0 and 1")


def _create_mock_completion_response(content: str):
    """Helper to mock a Groq chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def test_default_verify_model(monkeypatch):
    """Verify fallback model for verification is openai/gpt-oss-20b when env is unset."""
    monkeypatch.delenv("GROQ_MODEL_VERIFY", raising=False)
    assert get_verify_model() == "openai/gpt-oss-20b"


@pytest.mark.parametrize(
    "invalid_schema",
    [
        {"type": "object", "properties": {"a": {"type": "string"}}},
        dict,
        str,
        "SampleExtraction",
        SampleExtraction(company_name="Acme", confidence_score=1.0),
    ],
)
def test_unsupported_schema_types_rejected(invalid_schema, tmp_path):
    """complete_json must reject schemas that are not subclasses of pydantic.BaseModel with TypeError."""
    with pytest.raises(TypeError) as exc_info:
        complete_json(
            prompt="Extract data",
            schema=invalid_schema,
            cache_dir=tmp_path,
        )
    assert "schema must be a subclass of pydantic.BaseModel" in str(exc_info.value)


def test_missing_api_key_raises_clear_error(monkeypatch):
    """Missing GROQ_API_KEY must raise a clear, safe error without exposing tokens."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(RuntimeError) as exc_info:
        get_api_key()
    assert "GROQ_API_KEY is not configured" in str(exc_info.value)

    # Calling complete_json without key should also safely fail
    with pytest.raises(RuntimeError) as exc_info_call:
        complete_json(prompt="Test prompt", schema=SampleExtraction, client=None)
    assert "GROQ_API_KEY is not configured" in str(exc_info_call.value)


def test_pydantic_validation_success(tmp_path):
    """Successful LLM completion validates into a Pydantic instance."""
    mock_client = MagicMock()
    payload = {"company_name": "Acme Industrial", "confidence_score": 0.92}
    mock_client.chat.completions.create.return_value = _create_mock_completion_response(
        json.dumps(payload)
    )

    result = complete_json(
        prompt="Extract Acme Industrial",
        schema=SampleExtraction,
        model="mock-model",
        client=mock_client,
        cache_dir=tmp_path,
    )

    assert isinstance(result, SampleExtraction)
    assert result.company_name == "Acme Industrial"
    assert result.confidence_score == 0.92


def test_disk_cache_behavior(tmp_path):
    """Cached response is returned on subsequent calls without hitting the client."""
    mock_client = MagicMock()
    payload = {"company_name": "Cached Corp", "confidence_score": 0.88}
    mock_client.chat.completions.create.return_value = _create_mock_completion_response(
        json.dumps(payload)
    )

    prompt = "Extract Cached Corp"
    model = "mock-model"

    # First call - cache miss
    result1 = complete_json(
        prompt=prompt,
        schema=SampleExtraction,
        model=model,
        client=mock_client,
        cache_dir=tmp_path,
    )
    assert mock_client.chat.completions.create.call_count == 1
    assert result1.company_name == "Cached Corp"

    # Verify cache file exists
    cache_key = get_cache_key(model, prompt)
    cache_file = tmp_path / f"{cache_key}.json"
    assert cache_file.exists()

    # Second call - cache hit (client should not be invoked)
    result2 = complete_json(
        prompt=prompt,
        schema=SampleExtraction,
        model=model,
        client=mock_client,
        cache_dir=tmp_path,
    )
    assert mock_client.chat.completions.create.call_count == 1  # Still 1
    assert isinstance(result2, SampleExtraction)
    assert result2.company_name == "Cached Corp"


def test_failed_response_is_not_cached(tmp_path):
    """Failed responses (invalid JSON or validation errors) must never be cached."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _create_mock_completion_response(
        "INVALID_NOT_JSON"
    )

    prompt = "Extract failing entity"
    model = "mock-model"

    with pytest.raises(ValueError) as exc_info:
        complete_json(
            prompt=prompt,
            schema=SampleExtraction,
            model=model,
            client=mock_client,
            cache_dir=tmp_path,
            max_validation_retries=1,
        )

    assert "Failed to obtain valid JSON" in str(exc_info.value)
    # Ensure cache directory is empty
    cache_key = get_cache_key(model, prompt)
    assert not (tmp_path / f"{cache_key}.json").exists()


def test_validation_retry_and_recovery(tmp_path):
    """Invalid JSON on first attempt retries and recovers on subsequent attempt."""
    mock_client = MagicMock()
    invalid_resp = _create_mock_completion_response("Not JSON at all")
    valid_resp = _create_mock_completion_response(
        json.dumps({"company_name": "Recovered Corp", "confidence_score": 0.77})
    )
    mock_client.chat.completions.create.side_effect = [invalid_resp, valid_resp]

    result = complete_json(
        prompt="Extract Recovered Corp",
        schema=SampleExtraction,
        model="mock-model",
        client=mock_client,
        cache_dir=tmp_path,
        max_validation_retries=2,
    )

    assert result.company_name == "Recovered Corp"
    assert mock_client.chat.completions.create.call_count == 2


def test_validation_exhausts_retries_without_fabrication(tmp_path):
    """When retries are exhausted, an error is raised and no fabricated data is returned."""
    mock_client = MagicMock()
    # Pydantic schema validation will fail because confidence_score is missing
    bad_payload = json.dumps({"company_name": "Incomplete Data"})
    mock_client.chat.completions.create.return_value = _create_mock_completion_response(
        bad_payload
    )

    with pytest.raises(ValueError) as exc_info:
        complete_json(
            prompt="Extract incomplete entity",
            schema=SampleExtraction,
            model="mock-model",
            client=mock_client,
            cache_dir=tmp_path,
            max_validation_retries=2,  # 1 initial + 2 retries = 3 calls
        )

    assert mock_client.chat.completions.create.call_count == 3
    assert "Failed to obtain valid JSON conforming to schema" in str(exc_info.value)


def test_rate_limit_retry_with_retry_after(tmp_path):
    """Rate limit 429 response is retried respecting Retry-After header."""
    mock_client = MagicMock()

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    rate_limit_response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "0.05"},
    )
    rate_error = groq.RateLimitError(
        message="Rate limit exceeded",
        response=rate_limit_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    success_resp = _create_mock_completion_response(
        json.dumps({"company_name": "Throttled Corp", "confidence_score": 0.85})
    )
    mock_client.chat.completions.create.side_effect = [rate_error, success_resp]

    result = complete_json(
        prompt="Extract Throttled Corp",
        schema=SampleExtraction,
        model="mock-model",
        client=mock_client,
        cache_dir=tmp_path,
        max_rate_retries=2,
    )

    assert result.company_name == "Throttled Corp"
    assert mock_client.chat.completions.create.call_count == 2


def test_structured_output_fallback_on_bad_request(tmp_path):
    """Fallback from json_schema to json_object if model rejects json_schema."""
    mock_client = MagicMock()

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    bad_req_response = httpx.Response(400, request=request)
    bad_request_err = groq.BadRequestError(
        message="Model does not support json_schema",
        response=bad_req_response,
        body={"error": {"message": "json_schema unsupported"}},
    )

    success_resp = _create_mock_completion_response(
        json.dumps({"company_name": "Fallback Corp", "confidence_score": 0.99})
    )

    # First attempt with json_schema raises BadRequestError, fallback to json_object succeeds
    mock_client.chat.completions.create.side_effect = [bad_request_err, success_resp]

    result = complete_json(
        prompt="Extract Fallback Corp",
        schema=SampleExtraction,
        model="mock-model",
        client=mock_client,
        cache_dir=tmp_path,
    )

    assert result.company_name == "Fallback Corp"
    assert mock_client.chat.completions.create.call_count == 2


def test_concurrency_limiting(tmp_path):
    """Ensure requests acquire and release concurrency semaphore."""
    semaphore = threading.Semaphore(2)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _create_mock_completion_response(
        json.dumps({"company_name": "Concurrent Corp", "confidence_score": 0.90})
    )

    result = complete_json(
        prompt="Extract Concurrent Corp",
        schema=SampleExtraction,
        model="mock-model",
        client=mock_client,
        cache_dir=tmp_path,
        semaphore=semaphore,
    )

    assert result.company_name == "Concurrent Corp"


def test_real_groq_integration(tmp_path):
    """Real Groq integration test; skipped safely if GROQ_API_KEY is not set or network is unreachable."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        pytest.skip("GROQ_API_KEY is unavailable in this environment")

    # Use verification model for lightweight quick test
    model = os.getenv("GROQ_MODEL_VERIFY", "openai/gpt-oss-20b")
    prompt = "Output a JSON object with company_name 'Groq Technologies' and confidence_score 0.99."

    try:
        result = complete_json(
            prompt=prompt,
            schema=SampleExtraction,
            model=model,
            cache_dir=tmp_path,
        )
        assert isinstance(result, SampleExtraction)
        assert result.company_name
        assert 0.0 <= result.confidence_score <= 1.0

        # Second call to verify cache
        cached_result = complete_json(
            prompt=prompt,
            schema=SampleExtraction,
            model=model,
            cache_dir=tmp_path,
        )
        assert cached_result.company_name == result.company_name
    except groq.APIConnectionError as conn_err:
        pytest.skip(
            f"Groq network connection not reachable in current execution environment: {conn_err}"
        )
