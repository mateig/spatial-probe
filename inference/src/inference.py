from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config import InferenceCfg, ModelSpec
from src.dataset import Scene

_DTYPES: dict[str, torch.dtype | str] = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Loaded:
    spec: ModelSpec
    processor: Any
    model: Any


def load_model(spec: ModelSpec, cfg: InferenceCfg) -> Loaded:
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        device_map=cfg.device_map,
        torch_dtype=_DTYPES[cfg.dtype],
        trust_remote_code=True,
    )
    return Loaded(spec=spec, processor=processor, model=model)


def free_model(loaded: Loaded) -> None:
    del loaded.model
    del loaded.processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def supports(spec: ModelSpec, modalities: tuple[str, ...]) -> bool:
    # PaliGemma 2 mix variants accept exactly one image per prompt.
    if spec.family == "paligemma" and "depth" in modalities:
        return False
    return True


def _question_text(
    scene: Scene,
    question_text: str,
    modalities: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if "description" in modalities:
        parts.append("Scene description:")
        parts.append(scene.description)
    parts.append("Question:")
    parts.append(question_text)
    return "\n\n".join(parts)


def _images(scene: Scene, modalities: tuple[str, ...]) -> list[Image.Image]:
    paths = [scene.rgb_path]
    if "depth" in modalities:
        paths.append(scene.depth_path)
    return [Image.open(p).convert("RGB") for p in paths]


def _prompt(
    loaded: Loaded,
    scene: Scene,
    question_text: str,
    modalities: tuple[str, ...],
    prompt: str,
) -> str:
    body = _question_text(scene, question_text, modalities)
    if loaded.spec.family == "paligemma":
        return "<image>answer en " + body

    content: list[dict[str, str]] = [{"type": "image", "image": str(scene.rgb_path)}]
    if "depth" in modalities:
        content.append({"type": "image", "image": str(scene.depth_path)})
    content.append({"type": "text", "text": body})
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    return loaded.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(
    loaded: Loaded,
    scene: Scene,
    question_texts: list[str],
    modalities: tuple[str, ...],
    prompt: str,
    cfg: InferenceCfg,
) -> list[str]:
    if not question_texts:
        return []
    prompts = [_prompt(loaded, scene, qt, modalities, prompt) for qt in question_texts]
    images = _images(scene, modalities)

    tok = loaded.processor.tokenizer
    saved_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = loaded.processor(
            text=prompts,
            images=[images] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
    finally:
        tok.padding_side = saved_side

    inputs = {
        k: v.to(loaded.model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
    }
    if cfg.do_sample:
        kwargs["temperature"] = cfg.temperature

    with torch.inference_mode():
        output_ids = loaded.model.generate(**inputs, **kwargs)

    new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    return [
        s.strip()
        for s in loaded.processor.batch_decode(new_ids, skip_special_tokens=True)
    ]
