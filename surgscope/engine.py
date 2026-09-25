"""Model backend: ms-swift ``TransformersEngine`` (bf16, SDPA, batch size 1, greedy).

The SurgScope adapter is merged into the base model at load time, the same way as
``swift export --merge_lora``, so the weights equal those of the challenge submission.

The video-sampling environment must be in place before ms-swift / qwen-vl-utils are
imported: ms-swift copies these variables into ``qwen_vl_utils.vision_process``.
Every request additionally carries its own frame count (``min_frames = max_frames = N``),
which takes precedence over the ``FPS_*_FRAMES`` fallback.
"""
from __future__ import annotations

import os
import sys

INFER_ENV = {
    "FPS_MIN_FRAMES": "1536",               # fallback only; requests carry min/max_frames
    "FPS_MAX_FRAMES": "1536",
    "VIDEO_MAX_TOKEN_NUM": "128",
    "VIDEO_MIN_TOKEN_NUM": "32",            # below the smallest baked grid (56 tokens/pair)
    "FORCE_QWENVL_VIDEO_READER": "torchvision",
}


def configure_env() -> None:
    """Set the inference sampling environment (call before importing ms-swift)."""
    loaded = [m for m in ("swift", "qwen_vl_utils") if m in sys.modules]
    if loaded:
        bad = {k: os.environ.get(k) for k, v in INFER_ENV.items() if os.environ.get(k) != v}
        if bad:
            raise RuntimeError(f"{loaded} imported before surgscope.engine.configure_env(); "
                               f"sampling environment differs: {bad}")
    os.environ.update(INFER_ENV)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("USE_HF", "1")     # ms-swift: resolve model ids on the HF Hub


class SwiftEngine:
    """Thin wrapper: one video + one system/user turn in, greedy text out."""

    def __init__(self, model, adapter=None, attn_impl: str = "sdpa", max_new_tokens: int = 64):
        configure_env()
        import torch
        from swift import InferRequest, RequestConfig, Swift, TransformersEngine

        self._InferRequest, self._RequestConfig = InferRequest, RequestConfig
        self.engine = TransformersEngine(str(model), model_type="qwen3_5",
                                         torch_dtype=torch.bfloat16, attn_impl=attn_impl,
                                         max_batch_size=1,
                                         adapters=[str(adapter)] if adapter else None)
        if adapter:
            Swift.merge_and_unload(self.engine.model)
            self.engine.model = self.engine.engine = self.engine.model.model
        self.cfg = RequestConfig(max_tokens=max_new_tokens, temperature=0.0)

    def request(self, system: str, user: str, video, nframes: int):
        return self._InferRequest(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            videos=[str(video)],
            chat_template_kwargs={"enable_thinking": False,
                                  "min_frames": nframes, "max_frames": nframes})

    def generate(self, system: str, user: str, video, nframes: int, max_tokens=None) -> str:
        cfg = self.cfg if max_tokens is None else self._RequestConfig(max_tokens=max_tokens,
                                                                      temperature=0.0)
        r = self.engine.infer([self.request(system, user, video, nframes)], cfg, use_tqdm=False)
        return r[0].choices[0].message.content
