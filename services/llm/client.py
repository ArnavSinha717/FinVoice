"""LLM Client — Instructor + Ollama with VRAM-aware lifecycle management."""

import os
import requests
import instructor
from openai import OpenAI
from pydantic import BaseModel
from loguru import logger


# Default Ollama URL (can be overridden via .env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Default extraction model. Must be a NON-REASONING model: via Ollama's
# OpenAI-compatible endpoint, reasoning models (qwen3:*, qwen3.5:*) put their
# chain-of-thought in a separate `reasoning` field, leave `content` empty, and
# spend the whole token budget doing it. See pipeline/orchestrator.py.
DEFAULT_MODEL = os.getenv("FINVOICE_LLM_MODEL", "qwen2.5:3b")


def get_instructor_client(
    base_url: str | None = None,
    mode: instructor.Mode = instructor.Mode.JSON,
) -> instructor.Instructor:
    """Create an Instructor client connected to Ollama.

    Args:
        base_url: Ollama API URL (defaults to OLLAMA_URL env var)
        mode: Instructor output mode (JSON or JSON_SCHEMA)

    Returns:
        Instructor-wrapped OpenAI client
    """
    url = base_url or f"{OLLAMA_URL}/v1"
    return instructor.from_openai(
        OpenAI(base_url=url, api_key="ollama"),
        mode=mode,
    )


def extract_structured(
    prompt: str,
    response_model: type[BaseModel],
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    max_retries: int = 2,
    base_url: str | None = None,
    timeout: float = 120,
) -> BaseModel:
    """Extract structured data from text using Qwen3 via Instructor.

    On 6GB VRAM: Ollama auto-loads qwen3:8b (~5GB) on first call.
    Call unload_ollama_model() after all extractions to free VRAM.

    Args:
        prompt: The user prompt (transcript text, utterance, etc.)
        response_model: Pydantic model class for structured output
        model: Ollama model name
        system_prompt: Optional system prompt for context
        max_retries: Instructor retry count on schema validation failure
        timeout: Request timeout in seconds (default 120s)

    Returns:
        Instance of response_model
    """
    url = base_url or f"{OLLAMA_URL}/v1"
    client = instructor.from_openai(
        OpenAI(base_url=url, api_key="ollama", timeout=timeout),
        mode=instructor.Mode.JSON,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        max_retries=max_retries,
    )
    return result


def extract_structured_batch(
    prompts: list[str],
    response_model: type[BaseModel],
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
) -> list[BaseModel]:
    """Extract structured data from multiple prompts sequentially.

    Keeps the model loaded across calls (efficient on Ollama).
    Call unload_ollama_model() once after the full batch.
    """
    client = get_instructor_client()
    results = []

    for i, prompt in enumerate(prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            result = client.chat.completions.create(
                model=model,
                response_model=response_model,
                messages=messages,
                max_retries=2,
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Extraction failed for prompt {i}: {e}")
            results.append(None)

    return results


def extract_raw(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    timeout: float = 60,
    temperature: float | None = None,
) -> str:
    """Raw LLM completion — no Instructor, no schema validation.

    Much faster than extract_structured because:
    1. No JSON mode constraint (model generates freely)
    2. No Instructor retry loop on validation failure
    3. Single HTTP request, no wrapping

    Returns:
        Raw text response from the model
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{OLLAMA_URL}/v1"
    client = OpenAI(base_url=url, api_key="ollama", timeout=timeout)

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    msg = response.choices[0].message
    content = msg.content or ""
    if not content and getattr(msg, "reasoning", None):
        logger.warning(
            f"Model '{model}' returned only reasoning and no content — it is a "
            f"reasoning model. Set FINVOICE_LLM_MODEL to a non-reasoning model "
            f"(e.g. qwen2.5:3b)."
        )
    return content


def preload_ollama_model(model: str = DEFAULT_MODEL, keep_alive: str = "5m") -> bool:
    """Pre-load an Ollama model into VRAM so first inference is fast.

    Sends a minimal generate request with keep_alive to warm the model.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": keep_alive},
            timeout=120,
        )
        if resp.status_code == 200:
            logger.info(f"Ollama model '{model}' pre-loaded (keep_alive={keep_alive})")
            return True
        logger.warning(f"Ollama preload returned {resp.status_code}")
        return False
    except Exception as e:
        logger.warning(f"Failed to preload Ollama model: {e}")
        return False


def unload_ollama_model(model: str = DEFAULT_MODEL) -> None:
    """Force Ollama to release GPU memory for a model.

    CRITICAL on 6GB VRAM: Must call this before loading WhisperX.
    Sends keep_alive=0 which tells Ollama to immediately unload the model.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Ollama model '{model}' unloaded (VRAM freed)")
        else:
            logger.warning(f"Ollama unload returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Failed to unload Ollama model: {e}")


def check_ollama_health() -> dict:
    """Check if Ollama is running and which models are available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "healthy", "models": models}
        return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except requests.ConnectionError:
        return {"status": "unreachable", "detail": f"Cannot connect to {OLLAMA_URL}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
