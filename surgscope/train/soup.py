"""Uniform weight soup of LoRA adapters, and merging into the base model.

The released model averages three adapters trained under three sampling regimes. The
LoRA matrices A and B are averaged separately (accumulated in fp32, cast back to the
adapters' dtype); the soup adapter is then merged into Qwen3.5-9B with ``swift export``.
Tensors are streamed one at a time, so peak memory is ~2 GB.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-9B"


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def make_soup(adapters, out, names=None, base_model: str = BASE_MODEL) -> Path:
    """Average ``adapters`` (directories, order preserved) into ``out``."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    files = [Path(a) / "adapter_model.safetensors" for a in adapters]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    acc, dtypes, ref_keys = {}, {}, None
    for i, f in enumerate(files):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            if i == 0:
                ref_keys = set(keys)
            elif set(keys) != ref_keys:
                raise ValueError(f"tensor names of {f} differ from {files[0]}")
            for k in keys:
                t = sf.get_tensor(k)
                if i == 0:
                    dtypes[k] = t.dtype
                    acc[k] = t.to(torch.float32)
                else:
                    acc[k] += t.to(torch.float32)
    n = len(files)
    merged = {k: (v / n).to(dtypes[k]) for k, v in acc.items()}
    save_file(merged, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})

    # peft configuration: identical across members; point it at the public base model
    cfg = json.loads((Path(adapters[0]) / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = base_model
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    extra = Path(adapters[0]) / "additional_config.json"
    if extra.exists():
        shutil.copy2(extra, out / "additional_config.json")
    members = [{"name": (names[i] if names else Path(a).name),
                "sha256": sha256(files[i])} for i, a in enumerate(adapters)]
    (out / "SOUP_MEMBERS.json").write_text(json.dumps(
        {"method": "uniform mean of LoRA A and B (fp32 accumulation)", "members": members,
         "sha256": sha256(out / "adapter_model.safetensors")}, indent=2))
    return out


def merge(adapter, out, base_model: str = BASE_MODEL, dry_run: bool = False) -> list[str]:
    """Merge a LoRA adapter into the base model (bf16) with ``swift export``."""
    cmd = ["swift", "export", "--model", str(base_model), "--model_type", "qwen3_5",
           "--adapters", str(adapter), "--merge_lora", "true", "--output_dir", str(out)]
    if not dry_run:
        env = dict(os.environ, USE_HF=os.environ.get("USE_HF", "1"))
        subprocess.run(cmd, check=True, env=env)
    return cmd
