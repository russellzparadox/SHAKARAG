from __future__ import annotations

import json
import re
from typing import Any

import httpx

JSON_INSTRUCTIONS = "Respond with a single JSON object and nothing else."


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    brace = text.find("{")
    if brace != -1:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace : i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _decode_first_json(body: str) -> dict[str, Any]:
    body = body.strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(body)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    raise LLMError(f"Unexpected LLM response shape: {body[:400]}")


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        eff_timeout = timeout if timeout else self.timeout
        try:
            response = httpx.post(
                self.endpoint, json=payload, headers=headers, timeout=eff_timeout
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if response.status_code == 400:
            payload.pop("max_tokens", None)
            response = httpx.post(
                self.endpoint, json=payload, headers=headers, timeout=eff_timeout
            )
        if response.status_code != 200:
            raise LLMError(f"LLM returned HTTP {response.status_code}: {response.text[:400]}")
        data = _decode_first_json(response.text)
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = (message.get("reasoning_content") or "").strip()
            return f"{content}\n{reasoning}".strip() if reasoning and not content.strip() else content
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {response.text[:400]}") from exc

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 8192,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        messages = [*messages]
        raw = self.chat(messages, max_tokens=max_tokens, timeout=timeout)
        parsed = extract_json(raw)
        if parsed is None:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"That was not valid JSON. {JSON_INSTRUCTIONS}",
                }
            )
            raw = self.chat(messages, max_tokens=max_tokens, timeout=timeout)
            parsed = extract_json(raw)
        if parsed is None:
            raise LLMError("Model did not return parseable JSON.")
        return parsed
