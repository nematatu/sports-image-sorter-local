"""Optional cloud engines (OFF unless explicitly approved and enabled).

Sending personal photos to a third party is a decision the tool must never make
on its own. Every engine in this module is therefore gated twice:

1. ``SPORTS_SORTER_ALLOW_CLOUD=1`` must be set in the environment, and
2. the provider's API key must be present.

If either gate is missing, ``availability()`` returns ``(False, reason)`` and
the CLI reports the engine as "not verified / not configured" instead of
silently falling back to something else.

The three implementations show how to add a cloud engine without touching the
pipeline. The reference evaluation did **not** run them (the user approved
local-only processing), so they are documented as *unverified* in
``docs/engines.md``. They are still useful the day someone decides to opt in.

None of them writes an API key or a photo to disk or to the database; only the
model's textual answer is stored (``raw``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..images import load_prepared
from .base import Engine, Prediction


def cloud_allowed() -> tuple[bool, str]:
    if os.environ.get("SPORTS_SORTER_ALLOW_CLOUD", "0") != "1":
        return False, "cloud engines disabled (requires SPORTS_SORTER_ALLOW_CLOUD=1 and explicit human approval)"
    return True, "allowed"


def _encode_jpeg(path: str | Path, max_side: int = 1024) -> bytes:
    image = load_prepared(path, max_side=max_side)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 120) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # surface the body, it explains quota/auth issues
        body = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


class _CloudEngine(Engine):
    sends_images_offsite = True
    requires_credentials = True
    requires_network = True
    env_key = ""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.env_key, "")

    def availability(self) -> tuple[bool, str]:
        ok, reason = cloud_allowed()
        if not ok:
            return False, reason
        if not self.api_key:
            return False, f"{self.env_key} is not set"
        return True, "configured (sending photos off-device: confirm you have approval)"

    def _sport_menu(self) -> tuple[list[str], list[str]]:
        ids = [lab.id for lab in self.config.labels] + ["unclassified"]
        descr = []
        for lab in self.config.labels:
            descr.append(f"{lab.id}: {lab.name_en or lab.id}")
        descr.append("unclassified: no sport can be identified (portrait, ceremony, casual clothes, ambiguous)")
        return ids, descr


class GeminiVisionEngine(_CloudEngine):
    name = "gemini"
    version = "gemini-flash-structured-v1"
    description = "Google Gemini Vision with JSON schema output (opt-in, paid, sends photos)"
    env_key = "GEMINI_API_KEY"

    def classify(self, path: str | Path) -> Prediction:
        try:
            model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            ids, descr = self._sport_menu()
            prompt = (
                "Classify the sport shown in the photo using visual evidence only: equipment, court/venue, "
                "uniform, body action, scoreboards. Do not use the filename. If the sport cannot be determined "
                "(portrait, medal ceremony, casual clothes, ambiguous), choose unclassified. "
                "Pick exactly one of: " + ", ".join(ids)
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_encode_jpeg(path)).decode()}},
                    ]
                }],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": ids},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["label", "confidence", "evidence"],
                    },
                },
            }
            out = _post_json(url, payload, {})
            text = out["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            label = parsed.get("label")
            if label not in ids:
                label = "unclassified"
            conf = float(parsed.get("confidence", 0.0))
            return Prediction(label_id=label, confidence=conf, scores={label: conf}, raw={"engine": self.name, **parsed, "model": model})
        except Exception as exc:
            return self._error(exc)


class OpenAIVisionEngine(_CloudEngine):
    name = "openai"
    version = "openai-vision-structured-v1"
    description = "OpenAI vision with JSON schema output (opt-in, paid, sends photos)"
    env_key = "OPENAI_API_KEY"

    def classify(self, path: str | Path) -> Prediction:
        try:
            model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
            ids, _ = self._sport_menu()
            b64 = base64.b64encode(_encode_jpeg(path)).decode()
            prompt = (
                "Classify the sport from visual evidence only (equipment, court, uniform, action, signage), "
                "never from the filename. Use 'unclassified' when the sport is not identifiable. "
                "Allowed labels: " + ", ".join(ids)
            )
            payload = {
                "model": model,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sport_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "enum": ids},
                                "confidence": {"type": "number"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["label", "confidence", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
            }
            out = _post_json("https://api.openai.com/v1/chat/completions", payload, {"Authorization": f"Bearer {self.api_key}"})
            text = out["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            label = parsed.get("label") if parsed.get("label") in ids else "unclassified"
            conf = float(parsed.get("confidence", 0.0))
            return Prediction(label_id=label, confidence=conf, scores={label: conf}, raw={"engine": self.name, **parsed, "model": model})
        except Exception as exc:
            return self._error(exc)


class TypeSafeJevEngine(_CloudEngine):
    """Jev 1.14 image input (experimental "danger zone" feature).

    Jev is a *decision* model: instead of free-form text it answers typed
    questions with calibrated probabilities. That maps nicely onto a Choice
    question whose criteria are the sport descriptions. Image input is
    explicitly experimental (``TypeSafe-Danger-Zone-Images: true``) and the
    request shape may change; treat the output as one signal, not as truth.

    Not executed in the reference run: no API key and no approval to upload
    photos. Implemented so that enabling it later is a config change only.
    """

    name = "jev"
    version = "jev-1.14-image-experimental"
    description = "TypeSafe Jev 1.14 image Choice question (experimental, opt-in, paid, sends photos)"
    env_key = "TYPESAFE_API_KEY"

    def availability(self) -> tuple[bool, str]:
        ok, reason = super().availability()
        if not ok:
            return ok, reason
        if os.environ.get("TYPESAFE_DANGER_ZONE_IMAGES", "0") != "1":
            return False, "set TYPESAFE_DANGER_ZONE_IMAGES=1 to opt into the experimental image input"
        return True, "configured (experimental image input; sends photos)"

    def classify(self, path: str | Path) -> Prediction:
        try:
            model = os.environ.get("TYPESAFE_MODEL", "jev-1.14.0")
            ref = f"img_{os.urandom(8).hex()}"
            criteria = {}
            for lab in self.config.labels:
                criteria[lab.id] = f"{lab.name_en or lab.id}: " + "; ".join(lab.evidence or lab.prompts[:1])
            criteria["unclassified"] = (
                "No sport can be identified: portrait, face close-up, medal ceremony, casual clothes, "
                "or no sport-specific equipment/venue is visible."
            )
            payload = {
                "model": model,
                "state": {"photo": ref},
                "assets": {
                    ref: {
                        "type": "image",
                        "format": "jpg",
                        "data_base64": base64.b64encode(_encode_jpeg(path, max_side=768)).decode(),
                    }
                },
                "questions": {
                    "sport": {
                        "type": "choice",
                        "instructions": (
                            f"Look at `{ref}`. Which sport is shown? Judge only from visual evidence: "
                            "equipment, court or venue, uniform, body action and signage. "
                            "Do not infer anything from people's appearance or identity. "
                            "Choose unclassified when the sport is not clearly identifiable."
                        ),
                        "criteria": criteria,
                    }
                },
            }
            out = _post_json(
                "https://api.typesafe.ai/v1/systemone",
                payload,
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "TypeSafe-Danger-Zone-Images": "true",
                },
            )
            answer = (out.get("answers") or {}).get("sport")
            label = None
            conf = None
            if isinstance(answer, dict):
                label = answer.get("choice") or answer.get("value") or answer.get("label")
                conf = answer.get("probability") or answer.get("confidence")
            elif isinstance(answer, str):
                label = answer
            if label not in [lab.id for lab in self.config.labels] + ["unclassified"]:
                label = "unclassified"
            return Prediction(
                label_id=label,
                confidence=float(conf) if conf is not None else None,
                scores={label: float(conf or 0.0)},
                raw={"engine": self.name, "model": model, "answer": answer},
            )
        except Exception as exc:
            return self._error(exc)
