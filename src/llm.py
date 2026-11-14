"""Minimal OpenRouter chat-completions client with tool calling, retries and token/cost accounting."""
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def chat(model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None,
         temperature: float = 0.0, retries: int = 3) -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 1024, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            if "choices" not in out:
                raise LLMError(f"unexpected response: {str(out)[:300]}")
            return out
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:300]}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise LLMError(last) from e
        except (urllib.error.URLError, TimeoutError, LLMError) as e:
            last = str(e)
        time.sleep(2 ** attempt)
    raise LLMError(f"failed after {retries} attempts: {last}")
