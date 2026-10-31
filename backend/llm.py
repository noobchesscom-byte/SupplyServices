"""Groq LLM client module for SupplyGraph.

Provides structured JSON completion with:
- Configuration loading from repository root .env
- Schema enforcement via Groq JSON Schema structured output or json_object mode
- Pydantic model validation (requires a BaseModel subclass)
- Automatic retry on invalid JSON or validation failure (up to 2 retries)
- Rate-limit (HTTP 429) retry with exponential backoff & Retry-After support
- Concurrency limiting (default semaphore: 3)
- Disk caching for successful responses keyed by model and prompt hash
- Safe error handling without leaking secrets or fabricating data
"""

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
import groq
from groq import APIStatusError, BadRequestError, Groq, RateLimitError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Repository root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load repository-root .env without overriding already-set environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

# Disk cache directory for LLM completions
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "llm"

# Default concurrency limit
DEFAULT_CONCURRENCY = 3
_GLOBAL_SEMAPHORE = threading.Semaphore(DEFAULT_CONCURRENCY)

T = TypeVar("T", bound=BaseModel)


def get_api_key() -> str:
    """Retrieve GROQ_API_KEY safely.

    Raises RuntimeError with a clear message if not configured.
    Never prints, logs, or exposes the key value.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in the repository-root .env file."
        )
    return key


def get_extract_model() -> str:
    """Return the model to use for extraction tasks."""
    return os.getenv("GROQ_MODEL_EXTRACT", "openai/gpt-oss-120b").strip()


def get_verify_model() -> str:
    """Return the model to use for verification tasks."""
    return os.getenv("GROQ_MODEL_VERIFY", "openai/gpt-oss-20b").strip()


def get_client(api_key: Optional[str] = None) -> Groq:
    """Instantiate a Groq client with the resolved API key."""
    resolved_key = api_key if api_key is not None else get_api_key()
    return Groq(api_key=resolved_key)


def get_cache_key(model: str, prompt: str) -> str:
    """Derive disk cache key from model name and prompt hash."""
    material = f"{model}:{prompt}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_cache(cache_file: Path) -> Optional[Any]:
    """Read cached response from disk."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("response")
    except Exception as e:
        logger.warning("Failed to read disk cache at %s: %s", cache_file, e)
        return None


def _write_cache(cache_file: Path, model: str, prompt: str, response_data: Any) -> None:
    """Write successful response to disk atomically."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".tmp")
        payload = {
            "model": model,
            "created_at": time.time(),
            "response": response_data,
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_file.replace(cache_file)
    except Exception as e:
        logger.warning("Failed to write disk cache at %s: %s", cache_file, e)


def _extract_schema_details(schema: Type[T]) -> Tuple[str, Dict[str, Any]]:
    """Extract schema name and JSON schema dict from a Pydantic model class."""
    schema_dict = schema.model_json_schema()
    schema_name = getattr(schema, "__name__", "ResponseModel")
    return schema_name, schema_dict


def _ensure_json_in_prompt(prompt: str, schema_dict: Dict[str, Any]) -> str:
    """Ensure the prompt explicitly requests JSON, appending schema instructions if needed."""
    text = prompt.strip()
    if "json" not in text.lower():
        text += "\n\nRespond with valid JSON."
    if "schema" not in text.lower():
        text += f"\n\nJSON Schema:\n{json.dumps(schema_dict, indent=2)}"
    return text


def _clean_json_text(raw_text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    return text


def _parse_and_validate(raw_text: str, schema: Type[T]) -> T:
    """Parse JSON string and validate against Pydantic schema model.

    Never invents fallback data or returns unvalidated JSON.
    """
    cleaned = _clean_json_text(raw_text)
    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON returned: {exc}") from exc

    if not isinstance(parsed_json, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed_json).__name__}")

    try:
        return schema.model_validate(parsed_json)
    except ValidationError as val_err:
        raise ValueError(f"Schema validation error: {val_err}") from val_err


def _execute_groq_call(
    client: Groq,
    model: str,
    prompt: str,
    schema_name: str,
    schema_dict: Dict[str, Any],
) -> str:
    """Call Groq API trying JSON Schema structured output first, falling back to json_object."""
    messages = [
        {
            "role": "system",
            "content": "You are a precise data system. Always output valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]

    # Attempt structured output via json_schema if supported
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_dict,
                },
            },
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except BadRequestError as err:
        logger.info(
            "json_schema structured output rejected for model %s; falling back to json_object: %s",
            model,
            err,
        )
    except Exception:
        raise

    # Fallback to json_object mode
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _handle_rate_limit(
    exc: Exception,
    attempt: int,
    max_retries: int = 5,
    base_backoff: float = 1.0,
) -> None:
    """Handle HTTP 429 rate limit with exponential backoff and Retry-After header."""
    if attempt >= max_retries:
        raise exc

    sleep_seconds: Optional[float] = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        retry_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_header:
            try:
                sleep_seconds = float(retry_header)
            except (ValueError, TypeError):
                pass

    if sleep_seconds is None:
        # Exponential backoff with jitter
        sleep_seconds = (base_backoff * (2**attempt)) + random.uniform(0.1, 0.5)

    time.sleep(sleep_seconds)


def complete_json(
    prompt: str,
    schema: Type[T],
    model: Optional[str] = None,
    client: Optional[Groq] = None,
    cache_dir: Optional[Path] = None,
    semaphore: Optional[threading.Semaphore] = None,
    max_validation_retries: int = 2,
    max_rate_retries: int = 5,
) -> T:
    """Execute structured JSON completion using Groq with caching, retries, and Pydantic validation.

    Parameters:
        prompt: User prompt text.
        schema: Pydantic BaseModel subclass defining the output structure.
        model: Groq model name; defaults to GROQ_MODEL_EXTRACT from environment.
        client: Optional pre-configured Groq client (useful for testing/injection).
        cache_dir: Optional custom cache directory.
        semaphore: Concurrency limiter; defaults to global semaphore of 3.
        max_validation_retries: Number of additional retries on invalid JSON or validation failure (default 2).
        max_rate_retries: Maximum retries for HTTP 429 rate limit responses (default 5).

    Returns:
        Validated instance of the Pydantic schema model.

    Raises:
        TypeError: If schema is not a subclass of pydantic.BaseModel.
        RuntimeError: If GROQ_API_KEY is missing or the API response cannot be obtained.
        ValueError: If valid schema conforming JSON cannot be obtained within allowed retries.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"schema must be a subclass of pydantic.BaseModel, got {schema!r} (type: {type(schema).__name__})"
        )

    resolved_model = model or get_extract_model()
    schema_name, schema_dict = _extract_schema_details(schema)

    # 1. Check disk cache
    target_cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_key = get_cache_key(resolved_model, prompt)
    cache_file = target_cache_dir / f"{cache_key}.json"

    cached_response = _read_cache(cache_file)
    if cached_response is not None:
        try:
            if isinstance(cached_response, dict):
                return schema.model_validate(cached_response)
        except Exception as cache_val_err:
            logger.warning("Cached data failed schema validation; refetching: %s", cache_val_err)

    # 2. Prepare client and concurrency limiter
    groq_client = client or get_client()
    limiter = semaphore or _GLOBAL_SEMAPHORE

    # 3. Validation and retry loop
    # Total attempts = 1 initial + max_validation_retries
    total_validation_attempts = 1 + max_validation_retries
    current_prompt = _ensure_json_in_prompt(prompt, schema_dict)
    last_exception: Optional[Exception] = None

    for validation_attempt in range(total_validation_attempts):
        # Handle 429 rate limiting with backoff
        raw_text: Optional[str] = None
        for rate_attempt in range(total_rate_retries := max_rate_retries + 1):
            try:
                with limiter:
                    raw_text = _execute_groq_call(
                        client=groq_client,
                        model=resolved_model,
                        prompt=current_prompt,
                        schema_name=schema_name,
                        schema_dict=schema_dict,
                    )
                break
            except (RateLimitError, APIStatusError) as exc:
                status_code = getattr(exc, "status_code", None)
                if isinstance(exc, RateLimitError) or status_code == 429:
                    _handle_rate_limit(exc, rate_attempt, max_retries=max_rate_retries)
                    continue
                raise

        if raw_text is None:
            raise RuntimeError("Failed to receive a response from Groq API.")

        # Parse and validate response
        try:
            validated_result = _parse_and_validate(raw_text, schema)

            # 4. Cache only successful, validated responses on disk
            cache_payload = validated_result.model_dump()
            _write_cache(cache_file, resolved_model, prompt, cache_payload)

            return validated_result
        except ValueError as parse_err:
            last_exception = parse_err
            logger.warning(
                "Attempt %d/%d failed validation: %s",
                validation_attempt + 1,
                total_validation_attempts,
                parse_err,
            )
            # Adjust prompt for the next attempt with error details
            current_prompt = (
                f"{_ensure_json_in_prompt(prompt, schema_dict)}\n\n"
                f"[Feedback]: The previous attempt was invalid: {parse_err}. "
                f"Please output strictly valid JSON conforming to the schema."
            )

    # If all attempts failed, never invent fallback data
    raise ValueError(
        f"Failed to obtain valid JSON conforming to schema after {total_validation_attempts} attempts. "
        f"Last error: {last_exception}"
    )
