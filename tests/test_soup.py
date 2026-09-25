import json

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from surgscope.train.soup import make_soup  # noqa: E402


def test_uniform_soup(tmp_path):
    dirs = []
    for i in range(3):
        d = tmp_path / f"m{i}"
        d.mkdir()
        st.save_file({"a.lora_A.weight": torch.full((2, 3), float(i), dtype=torch.bfloat16),
                      "a.lora_B.weight": torch.full((3, 2), float(2 * i), dtype=torch.bfloat16)},
                     str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/local/Qwen", "r": 2}))
        dirs.append(d)
    out = make_soup(dirs, tmp_path / "soup", names=["A", "B", "C"])
    t = st.load_file(str(out / "adapter_model.safetensors"))
    assert t["a.lora_A.weight"].dtype == torch.bfloat16
    assert torch.equal(t["a.lora_A.weight"], torch.full((2, 3), 1.0, dtype=torch.bfloat16))
    assert torch.equal(t["a.lora_B.weight"], torch.full((3, 2), 2.0, dtype=torch.bfloat16))
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen3.5-9B"
    assert [m["name"] for m in json.loads((out / "SOUP_MEMBERS.json").read_text())["members"]] == ["A", "B", "C"]
