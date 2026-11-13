"""Optional LLM assistance: candidate reranking and phrase polish.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint, which covers
OpenAI, Groq, Together, LM Studio, llama.cpp and Ollama (all of which expose
that path). Every entry point is best-effort: if no endpoint is configured or
the call fails, the caller keeps its heuristic result, so the pipeline never
depends on network access.

Configuration comes from the environment or ``data/settings.json``:

    CLIPFORGE_LLM_BASE_URL   e.g. http://localhost:11434/v1
    CLIPFORGE_LLM_API_KEY    optional for local servers
    CLIPFORGE_LLM_MODEL      e.g. llama3.1, gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

ENV_BASE = "CLIPFORGE_LLM_BASE_URL"
ENV_KEY = "CLIPFORGE_LLM_API_KEY"
ENV_MODEL = "CLIPFORGE_LLM_MODEL"

RERANK_SYSTEM = (
    "You are a short-form video editor who decides which moments of a long "
    "recording will perform best as vertical short clips. You value: a hook in "
    "the first seconds, emotional or funny peaks, concrete vivid detail, a "
    "satisfying ending, and self-contained moments that need no prior context. "
    "You penalise: rambling, repetition, setup that is never paid off, "
    "references to earlier context, and anything slow or low-energy."
)

PHRASE_SYSTEM = (
    "You write the big bold caption that sits at the top of a vertical short "
    "video. Rules: at most 8 words, at most 40 characters, ALL CAPS, no "
    "trailing punctuation, no hashtags, no emoji, no quotation marks, no scene "
    "directions. It must capture the single most arresting idea in the clip and "
    "read as punchy and self-contained. Return only the caption text."
)


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 45.0
    temperature: float = 0.4

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"


def config_from_env() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ.get(ENV_BASE, "").strip(),
        api_key=os.environ.get(ENV_KEY, "").strip(),
        model=os.environ.get(ENV_MODEL, "").strip(),
    )


def available() -> bool:
    return config_from_env().enabled


def describe() -> dict[str, Any]:
    cfg = config_from_env()
    return {
        "enabled": cfg.enabled,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "hint": (
            f"set {ENV_BASE} and {ENV_MODEL} to enable AI scoring"
            if not cfg.enabled else "AI scoring active"
        ),
    }


def _chat(messages: list[dict[str, str]], cfg: LLMConfig, max_tokens: int = 700) -> str | None:
    if not cfg.enabled:
        return None
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            response = client.post(cfg.endpoint(), json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        return None
    try:
        return str(data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None


def _extract_json(raw: str) -> Any:
    """Pull JSON out of a model response, fences and prose included."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # fall back to the first balanced array/object
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for index in range(start, len(text)):
            if text[index] == opener:
                depth += 1
            elif text[index] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: index + 1])
                    except ValueError:
                        break
    return None


def rerank(candidates: list[dict[str, Any]], context: str = "", cfg: LLMConfig | None = None) -> dict[int, dict[str, Any]]:
    """Score candidates with the model. Returns ``{index: {score, reason}}``.

    An empty dict means "no opinion" and the caller keeps heuristic ordering.
    """
    cfg = cfg or config_from_env()
    if not cfg.enabled or not candidates:
        return {}

    listing = []
    for i, cand in enumerate(candidates):
        listing.append(
            f"[{i}] ({cand.get('duration', 0):.0f}s) {cand.get('text', '')[:420]}"
        )
    prompt = (
        "Rank these candidate clips from a long video by likely viral performance.\n\n"
        + "\n\n".join(listing)
        + "\n\nReturn JSON only, as an array of objects "
          '{"index": <int>, "score": <0-10 number>, "reason": "<max 12 words>"}. '
          "Include every index exactly once, ordered best first."
    )
    if context:
        prompt = f"Overall video context: {context[:600]}\n\n" + prompt

    raw = _chat(
        [{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": prompt}],
        cfg,
    )
    parsed = _extract_json(raw or "")
    if not isinstance(parsed, list):
        return {}

    out: dict[int, dict[str, Any]] = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(candidates)):
            continue
        out[index] = {
            "score": max(0.0, min(10.0, score)) / 10.0,
            "reason": str(row.get("reason", ""))[:80],
        }
    return out


def polish_phrase(text: str, fallback: str, cfg: LLMConfig | None = None) -> str:
    """Rewrite a heuristic phrase into something punchier. Falls back silently."""
    cfg = cfg or config_from_env()
    if not cfg.enabled or not text.strip():
        return fallback

    raw = _chat(
        [
            {"role": "system", "content": PHRASE_SYSTEM},
            {"role": "user", "content": f"Clip transcript:\n{text[:1200]}\n\nCaption:"},
        ],
        cfg,
        max_tokens=60,
    )
    if not raw:
        return fallback
    candidate = raw.splitlines()[0].strip().strip('"').strip("'")
    candidate = re.sub(r"[#*_`]", "", candidate).strip()
    if not candidate or len(candidate) > 46 or len(candidate) < 6:
        return fallback
    if candidate.upper() == candidate:
        return candidate
    return candidate.upper()
