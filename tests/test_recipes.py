import pytest

from surgscope.train.recipes import build_command, grad_accum, load_recipe


def test_grad_accum():
    assert grad_accum(64, 16, 1) == 4
    assert grad_accum(32, 8, 1) == 4
    with pytest.raises(ValueError):
        grad_accum(32, 6, 1)


def test_dense_mix_command():
    cfg = load_recipe("configs/train/dense-mix.yaml")
    argv, env = build_command(cfg, "d.jsonl", "out", nproc_per_node=8)
    s = " ".join(argv)
    assert "--learning_rate 0.0001" in s and "--gradient_accumulation_steps 4" in s
    assert "--lora_rank 64 --lora_alpha 128" in s and "--max_length 57344" in s
    assert env["FPS_MAX_FRAMES"] == "1536" and env["VIDEO_MIN_TOKEN_NUM"] == "32"
    assert env["NPROC_PER_NODE"] == "8"


def test_single_gpu_smoke():
    cfg = load_recipe("configs/train/quickstart.yaml")
    argv, env = build_command(cfg, "d.jsonl", "out", max_frames=256)
    assert "NPROC_PER_NODE" not in env and env["FPS_MAX_FRAMES"] == "256"
    assert argv[argv.index("--max_steps") + 1] == "2"


def test_released_recipes():
    lr = {n: load_recipe(f"configs/train/{n}.yaml")["learning_rate"] for n in ("r768", "dense-mix", "dense-mix-2")}
    assert lr == {"r768": 1.4e-4, "dense-mix": 1e-4, "dense-mix-2": 1e-4}
