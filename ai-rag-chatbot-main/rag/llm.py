from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.helpers import env


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class LLM:
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text, provider="openai", model=self.model)


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = _resolve_gemini_model_name(genai, requested=model)
        self._model = genai.GenerativeModel(model_name=self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        # Gemini API supports "system_instruction" on the model and/or separate parts.
        # We'll prepend system instructions to keep behavior consistent.
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
        )
        text = getattr(resp, "text", "") or ""
        return LLMResult(text=text, provider="gemini", model=self.model)


def _resolve_gemini_model_name(genai, requested: str) -> str:
    """
    The google-generativeai SDK sometimes expects names like `models/gemini-1.5-flash`
    and the set of available models varies by account / API version.
    We resolve a working model by calling ListModels and choosing one that supports
    `generateContent`.
    """
    req = (requested or "").strip()
    if not req:
        req = "gemini-1.5-flash"

    # Normalize requested into both forms for matching
    req_no_prefix = req.removeprefix("models/")
    candidates_preferred = [
        req_no_prefix,
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    try:
        models = list(genai.list_models())
    except Exception:
        # If listing fails, try the requested name as-is (SDK may still accept it)
        return req if req.startswith("models/") else f"models/{req}"

    def supports_generate(m) -> bool:
        methods = getattr(m, "supported_generation_methods", None) or []
        return any(str(x).lower().endswith("generatecontent") or str(x) == "generateContent" for x in methods)

    available = [m for m in models if supports_generate(m)]
    names = [getattr(m, "name", "") for m in available]

    # Try preferred candidate list first
    for cand in candidates_preferred:
        for name in names:
            if name == cand or name == f"models/{cand}" or name.endswith(f"/{cand}"):
                return name

    # Fallback: pick first available generateContent model
    if names:
        return names[0]

    # Final fallback: requested with prefix
    return req if req.startswith("models/") else f"models/{req}"


class RuleBasedFallbackLLM(LLM):
    """
    If no API key is configured, we avoid hallucination by refusing to answer.
    The RAG pipeline will still show retrieved chunks and citations in the UI.
    """

    def __init__(self):
        self.model = "fallback"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        return LLMResult(
            text="The answer is not available in the uploaded DSA material.",
            provider="fallback",
            model=self.model,
        )


def get_llm() -> LLM:
    gemini_key = env("GEMINI_API_KEY")
    gemini_model = env("GEMINI_MODEL", "gemini-1.5-flash-latest") or "gemini-1.5-flash-latest"
    if gemini_key:
        return GeminiLLM(api_key=gemini_key, model=gemini_model)

    key = env("OPENAI_API_KEY")
    model = env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    if key:
        return OpenAILLM(api_key=key, model=model)
    return RuleBasedFallbackLLM()

