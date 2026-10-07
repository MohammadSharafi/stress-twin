"""Nebius Token Factory client (OpenAI-compatible API).

Roles and the models they prefer (first match in the live /v1/models listing wins):
- planner: NVIDIA Nemotron 3 Ultra, then Super. Designs hypothesis-driven stress scenarios.
- fast:    Nemotron 3.5 Lightning, then Nemotron 3 Nano. Short summaries.
- vision:  Nemotron 3 Nano Omni, then Cosmos3-Super-Reasoner. Reads failure keyframes.

Model ids are discovered at runtime, never hard-coded, because Token Factory rotates
endpoints. Override any role with STRESS_TWIN_<ROLE>_MODEL.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

PREFERENCES: dict[str, tuple[str, ...]] = {
    "planner": ("nemotron-3-ultra", "nemotron-3-super", "nemotron-super", "nemotron"),
    "fast": ("nemotron-3.5-lightning", "nemotron-3-5-lightning", "lightning", "nemotron-3-nano", "nemotron-nano", "nemotron"),
    "vision": ("nemotron-3-nano-omni", "nano-omni", "omni", "cosmos3-super-reasoner", "cosmos3", "cosmos-reason", "cosmos"),
}
EXCLUDE = ("embed", "reward", "guard", "safety", "parse", "rerank")


class LLMUnavailable(RuntimeError):
    pass


def load_dotenv(paths: list[Path] | None = None) -> None:
    """Read KEY=VALUE lines from a .env file into the environment without overriding values
    that are already set. Looks in the current directory, then the repository root."""
    paths = paths or [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value


def _norm(s: str) -> str:
    return s.lower().replace("_", "-").replace(" ", "-")


def pick_model(available: list[str], prefs: tuple[str, ...]) -> str | None:
    cands = [m for m in available if not any(x in _norm(m) for x in EXCLUDE)]
    for pref in prefs:
        hits = [m for m in cands if pref in _norm(m)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating think-blocks and code fences."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def image_part(img) -> dict:
    """OpenAI-format image content part from a numpy array or PIL image (base64 PNG data URL)."""
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


class TokenFactory:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, log_path: Path | None = None,
                 timeout: float = 120.0) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)
        self.base_url = base_url
        self.log_path = Path(log_path) if log_path else None
        self._models: dict[str, str] | None = None
        self.calls: list[dict] = []

    @classmethod
    def from_env(cls, log_path: Path | None = None) -> "TokenFactory":
        load_dotenv()
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise LLMUnavailable("NEBIUS_API_KEY is not set (environment or .env file); Token Factory features need a key "
                                 "(https://tokenfactory.nebius.com). Use --strategy boundary --analyst offline to run offline.")
        return cls(key, os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL), log_path=log_path)

    def models(self) -> dict[str, str]:
        if self._models is None:
            available = [m.id for m in self.client.models.list().data]
            chosen = {}
            for role, prefs in PREFERENCES.items():
                override = os.environ.get(f"STRESS_TWIN_{role.upper()}_MODEL")
                model = override or pick_model(available, prefs)
                if model:
                    chosen[role] = model
            if "planner" not in chosen:
                raise LLMUnavailable(f"no Nemotron model found among {len(available)} Token Factory models")
            chosen.setdefault("fast", chosen["planner"])
            self._models = chosen
        return self._models

    def has(self, role: str) -> bool:
        return role in self.models()

    def chat_json(self, role: str, system: str, user: str, schema: dict, images: list | None = None,
                  max_tokens: int = 4096, temperature: float = 0.4) -> dict:
        model = self.models()[role]
        content: list | str = user
        if images:
            content = [{"type": "text", "text": user}] + [image_part(i) for i in images]
        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single JSON object matching this JSON schema:\n"
                                         + json.dumps(schema)},
            {"role": "user", "content": content},
        ]
        attempts = [{"type": "json_schema", "json_schema": schema}, {"type": "json_object"}]
        last_err = None
        for fmt in attempts:
            t0 = time.time()
            entry = {"ts": round(t0, 3), "role": role, "model": model, "response_format": fmt["type"],
                     "images": len(images or [])}
            try:
                resp = self.client.chat.completions.create(model=model, messages=messages, response_format=fmt,
                                                           max_tokens=max_tokens, temperature=temperature)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                entry.update({"latency_s": round(time.time() - t0, 3),
                              "prompt_tokens": getattr(usage, "prompt_tokens", None),
                              "completion_tokens": getattr(usage, "completion_tokens", None)})
                data = extract_json(text)
                entry["ok"] = True
                self._log(entry)
                return data
            except Exception as e:  # noqa: BLE001 - logged and retried with a looser format
                last_err = e
                entry.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                              "latency_s": round(time.time() - t0, 3)})
                self._log(entry)
        raise LLMUnavailable(f"{role} call failed: {last_err}")

    def _log(self, entry: dict) -> None:
        self.calls.append(entry)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
