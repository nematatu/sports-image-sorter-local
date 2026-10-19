"""Local, offline neural engines (optional ``local`` extra).

Three flavours, all running on CPU or Apple ``mps``:

``clip`` / ``siglip2``
    Zero-shot image classification. The taxonomy prompts in the config are
    embedded with the model's text tower; each image is compared against the
    whole prompt set and the per-label prompt scores are aggregated. Two
    negative/abstain prompt groups (":data:`abstain_prompts`") make it possible
    for the model itself to say "this is a portrait / ceremony, not a sport",
    which is what turns ambiguous photos into ``未分類`` instead of a coin flip.

``sports100_vit`` / ``sportsnet7``
    Small supervised classifiers downloaded from the Hub. They have a fixed
    label set; anything that does not map onto the configured taxonomy is
    folded into ``other`` (``その他の競技``). They are strong where their label
    set overlaps our taxonomy and blind elsewhere, which is exactly why the
    fusion engine keeps them around.

No image ever leaves the machine in these engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from ..images import load_prepared
from .base import Engine, Prediction

ABSTAIN = "__abstain__"


def pick_device(preference: str = "auto") -> str:
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    pref = (preference or "auto").lower()
    if pref in {"mps", "cpu"}:
        if pref == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover - not the target platform
        return "cuda"
    return "cpu"


class _TorchEngine(Engine):
    """Shared torch plumbing (lazy imports, device selection, model cache)."""

    model_id: str = ""
    requires_network = False  # weights are downloaded once, then offline

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            return False, f"needs the 'local' extra (pip install -e '.[local]'): {exc}"
        return True, f"device={self.options.get('device', 'auto')} model={self.model_id}"

    def prepare(self) -> None:
        import torch
        from transformers import AutoProcessor  # noqa: F401

        self.torch = torch
        self.device = pick_device(self.options.get("device", "auto"))
        self.max_side = int(self.options.get("max_side", self.config.preprocess["max_side"]))

    def _to_device(self, model):
        model.to(self.device)
        model.eval()
        return model


class ZeroShotEngine(_TorchEngine):
    """Prompt-based zero-shot classification (CLIP / SigLIP family)."""

    prompt_aggregation = "mean"

    def prepare(self) -> None:
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        super().prepare()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self._to_device(AutoModelForZeroShotImageClassification.from_pretrained(self.model_id))
        # SigLIP's text tower has no attention mask: it must be fed a fixed
        # length sequence (`padding="max_length"` + truncation) or the pooled
        # text features are garbage. CLIP works with dynamic padding.
        model_type = getattr(self.model.config, "model_type", "")
        self.text_kwargs = (
            {"padding": "max_length", "truncation": True}
            if "siglip" in str(model_type).lower()
            else {"padding": True}
        )

        self.prompt_specs: list[tuple[str, str]] = []  # (label_id, prompt)
        for lab in self.config.labels:
            prompts = list(lab.prompts) or [f"a photo of a {lab.name_en or lab.id} player"]
            for prompt in prompts:
                self.prompt_specs.append((lab.id, prompt))
        for prompt in self.config.abstain_prompts:
            self.prompt_specs.append((ABSTAIN, prompt))
        self.texts = [p for _, p in self.prompt_specs]
        self.label_ids = [l for l, _ in self.prompt_specs]

    # -- inference -----------------------------------------------------------
    def _crops(self, image):
        yield image
        mode = str(self.options.get("crops", "center")).lower()
        if mode in {"none", "off", "false"}:
            return
        w, h = image.size
        side = int(min(w, h) * 0.75)
        if side < 64:
            return
        left = (w - side) // 2
        top = (h - side) // 2
        yield image.crop((left, top, left + side, top + side))
        if mode == "five":
            yield image.crop((0, 0, side, side))
            yield image.crop((w - side, 0, w, side))
            yield image.crop((0, h - side, side, h))

    def _probs_for_batch(self, images: list) -> list[dict[str, float]]:
        torch = self.torch
        inputs = self.processor(text=self.texts, images=images, return_tensors="pt", **self.text_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image  # (n_images, n_texts)
            probs = logits.softmax(dim=-1)
        # Aggregate prompts -> labels for every image in the batch.
        out: list[dict[str, float]] = []
        for row in probs:
            per_label: dict[str, list[float]] = {}
            for label_id, value in zip(self.label_ids, row.tolist()):
                per_label.setdefault(label_id, []).append(value)
            agg = (
                {k: sum(v) / len(v) for k, v in per_label.items()}
                if self.prompt_aggregation == "mean"
                else {k: max(v) for k, v in per_label.items()}
            )
            out.append(agg)
        return out

    def _finalize(self, agg: dict[str, float], raw: dict[str, Any]) -> Prediction:
        abstain = float(agg.get(ABSTAIN, 0.0))
        sport = {k: float(v) for k, v in agg.items() if k != ABSTAIN}
        total = sum(sport.values()) + abstain
        if total <= 0:
            return Prediction(label_id=None, status="error", error="model returned no scores", raw=raw)
        norm = {k: v / total for k, v in sport.items()}
        # Report the abstain mass as a score for transparency.
        raw = dict(raw)
        raw["abstain"] = abstain / total
        if abstain >= max(norm.values(), default=0.0):
            return Prediction(label_id="unclassified", confidence=abstain / total, scores=norm, raw=raw)
        label = max(norm, key=norm.get)
        return Prediction(label_id=label, confidence=norm[label], scores=norm, raw=raw)

    def classify(self, path: str | Path) -> Prediction:
        try:
            return next(self.classify_many([path]))[1]
        except StopIteration:  # pragma: no cover
            return self._error(RuntimeError("no result"))

    def classify_many(self, paths: Iterable[str | Path]) -> Iterator[tuple[str, Prediction]]:
        torch = self.torch
        batch: list[Path] = []
        batch_crops: list[Any] = []
        batch_meta: list[tuple[Path, int]] = []
        batch_size = int(self.options.get("batch_size", 4))

        def flush() -> Iterator[tuple[str, Prediction]]:
            if not batch:
                return
            try:
                per_image = self._probs_for_batch(batch_crops)
            except Exception as exc:  # pragma: no cover - model/runtime errors
                for p, _ in batch_meta:
                    yield str(p), self._error(exc)
            else:
                grouped: dict[Path, list[dict[str, float]]] = {}
                for (path, _), agg in zip(batch_meta, per_image):
                    grouped.setdefault(path, []).append(agg)
                for path, aggs in grouped.items():
                    if len(aggs) == 1:
                        agg = aggs[0]
                    else:
                        keys = set().union(*[set(a) for a in aggs])
                        agg = {k: sum(a.get(k, 0.0) for a in aggs) / len(aggs) for k in keys}
                    yield str(path), self._finalize(agg, {"engine": self.name, "model": self.model_id, "crops": len(aggs)})
            batch.clear()
            batch_crops.clear()
            batch_meta.clear()

        for path in paths:
            try:
                image = load_prepared(path, max_side=self.max_side)
            except Exception as exc:
                yield str(path), self._error(exc)
                continue
            for crop in self._crops(image):
                batch.append(Path(path))
                batch_crops.append(crop)
                batch_meta.append((Path(path), len(batch_crops)))
            if len(batch_crops) >= batch_size:
                yield from flush()
        yield from flush()

    def close(self) -> None:
        self.model = None
        try:
            self.torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass


class ClipZeroShotEngine(ZeroShotEngine):
    name = "clip"
    version = "clip-vit-b-32-laion2b-s34b-b79k"
    description = "LAION-2B OpenCLIP ViT-B/32 zero-shot (offline after first download)"
    model_id = "laion/CLIP-ViT-B-32-laion2b-s34b-b79k"


class Siglip2ZeroShotEngine(ZeroShotEngine):
    name = "siglip2"
    version = "siglip2-base-patch16-224"
    description = "SigLIP 2 base zero-shot (offline after first download)"
    model_id = "google/siglip2-base-patch16-224"


class SupervisedEngine(_TorchEngine):
    """Wraps a supervised image classifier with a fixed label set."""

    def prepare(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        super().prepare()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = self._to_device(AutoModelForImageClassification.from_pretrained(self.model_id))
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.class_names = [str(id2label.get(i, i)) for i in range(len(id2label))]
        self.class_label_ids = [self.config.resolve(name) for name in self.class_names]
        self.matched = [i for i, lab in enumerate(self.class_label_ids) if lab]

    def classify(self, path: str | Path) -> Prediction:
        try:
            image = load_prepared(path, max_side=self.max_side)
        except Exception as exc:
            return self._error(exc)
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = self.model(**inputs).logits.softmax(dim=-1)[0].tolist()
        scores: dict[str, float] = {lab.id: 0.0 for lab in self.config.labels}
        for idx, prob in enumerate(probs):
            label_id = self.class_label_ids[idx] if idx < len(self.class_label_ids) else None
            if label_id:
                scores[label_id] = scores.get(label_id, 0.0) + float(prob)
        top_classes = sorted(
            ((self.class_names[i], probs[i]) for i in range(min(len(probs), len(self.class_names)))),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]
        raw = {"engine": self.name, "model": self.model_id, "top_classes": top_classes}
        if not scores or max(scores.values()) <= 0:
            return Prediction(label_id="other", confidence=None, scores=scores, raw=raw)
        # Everything the model knows but that is not in our taxonomy is "other".
        matched_mass = sum(scores.values())
        other_mass = max(0.0, 1.0 - matched_mass)
        scores["other"] = other_mass
        label = max(scores, key=scores.get)
        return Prediction(label_id=label, confidence=scores[label], scores=scores, raw=raw)


class Sports100VitEngine(SupervisedEngine):
    name = "sports100_vit"
    version = "vit-base-sports100"
    description = "ViT fine-tuned on 100 sports classes (includes volleyball, table tennis)"
    model_id = "vumin001/vit-base-sports-classification"


class SportsNet7Engine(SupervisedEngine):
    name = "sportsnet7"
    version = "siglip2-sportsnet7"
    description = "SigLIP2 fine-tuned on 7 sports (includes badminton, tennis)"
    model_id = "prithivMLmods/SportsNet-7"
