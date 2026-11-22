"""Local LLM inference through Ollama's REST API (standard library only)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import config


class LLMError(Exception):
    """LLM failure with a user-readable message."""


class OllamaUnavailableError(LLMError):
    """The Ollama server cannot be reached."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


class OllamaEngine:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        base_url: str = config.OLLAMA_URL,
        temperature: float = config.LLM_TEMPERATURE,
        seed: int = config.LLM_SEED,
        num_ctx: int = config.LLM_NUM_CTX,
        max_new_tokens: int = config.LLM_MAX_NEW_TOKENS,
        timeout: float = config.LLM_TIMEOUT_S,
        keep_alive: str = config.LLM_KEEP_ALIVE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.keep_alive = keep_alive

    def _call(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # must precede URLError (its parent)
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = parsed.get("error", detail)
            except ValueError:
                pass
            if exc.code == 404:
                raise LLMError(
                    f"Ollama could not find model '{self.model}' ({detail}). "
                    f"Install it with: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app and retry. ({exc.reason})"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama request timed out after {self.timeout}s. "
                "CPU inference is slow; try a smaller model or raise LLM_TIMEOUT_S."
            ) from exc

    def is_ready(self) -> bool:
        """True if the configured model is installed. Raises OllamaUnavailableError if the server is down."""
        data = self._call("/api/tags")
        names = {m.get("name") for m in data.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_new_tokens,
            },
        }
        start = time.perf_counter()
        data = self._call("/api/chat", payload)
        latency = time.perf_counter() - start
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            latency_s=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )