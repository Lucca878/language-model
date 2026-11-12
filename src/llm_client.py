import json
import os
from typing import Any

from openrouter import OpenRouter


def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def call_llm(
    model_id: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "response",
    response_format: str = "json_schema",
) -> Any:
    """Call OpenRouter using the task's configured model and response format."""
    if response_format not in {"json_schema", "json_object", "text", "none"}:
        raise ValueError(f"Unsupported LLM response format '{response_format}'.")
    if response_format == "json_schema" and schema is None:
        raise ValueError("A JSON schema is required when response_format is 'json_schema'.")

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if response_format == "json_schema":
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    elif response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    client = _build_client()
    max_attempts = 3
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.send(**kwargs)
            content = response.choices[0].message.content

            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not isinstance(content, str):
                raise ValueError("OpenRouter returned non-text content.")

            if response_format in {"json_schema", "json_object"}:
                return json.loads(content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break

    raise RuntimeError(
        f"Failed to get output after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error
