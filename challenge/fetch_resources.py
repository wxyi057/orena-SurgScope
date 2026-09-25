#!/usr/bin/env python
"""Build challenge/resources/SurgScope-9B (Qwen3.5-9B with the SurgScope adapter merged in) so
that the image can run offline (the evaluation platform has no network). Merged on a GPU, the
weights equal those of the submitted image."""
from pathlib import Path

from huggingface_hub import snapshot_download

from surgscope.train.soup import merge

out = Path(__file__).resolve().parent / "resources" / "SurgScope-9B"
merge(snapshot_download("wxyi088/orena-SurgScope", allow_patterns=["adapter_*"]), out)
print("weights ready:", sorted(p.name for p in out.iterdir()))
