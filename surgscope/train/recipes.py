"""Training recipes -> ``swift sft`` command line and environment.

Shared by all three fine-tunes (as run for the released weights):
LoRA r=64, alpha=128, dropout 0.05 on all linear layers of the language model, the vision
encoder and the merger; bf16; gradient checkpointing; FlashAttention-2; AdamW (fused),
weight decay 0.1, beta2 0.95, gradient clipping 1.0; cosine schedule with 3 % warmup;
per-device batch 1 without packing; sequences longer than ``max_length`` are dropped;
seed 42; a checkpoint per epoch; no validation split (checkpoints were selected on the
local test split). Only the regime, learning rate, global batch and ``max_length`` differ.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from surgscope.policy import REGIME_META


def load_recipe(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("regime") not in REGIME_META:
        raise ValueError(f"{path}: unknown regime {cfg.get('regime')!r}")
    return cfg


def grad_accum(global_batch: int, world: int, per_device: int) -> int:
    per_step = world * per_device
    if global_batch % per_step:
        raise ValueError(f"global batch {global_batch} is not divisible by {world} GPU(s) x "
                         f"{per_device} per device; change --nproc-per-node/--nnodes")
    return global_batch // per_step


def sampling_env(regime: str, max_frames: int | None = None) -> dict:
    """Processor environment of a regime (read by qwen-vl-utils via ms-swift)."""
    m = REGIME_META[regime]
    lo, hi = m["fps_min_frames"], m["fps_max_frames"]
    if max_frames:
        lo, hi = min(lo, max_frames), min(hi, max_frames)
    return {"FPS_MIN_FRAMES": str(lo), "FPS_MAX_FRAMES": str(hi),
            "VIDEO_MAX_TOKEN_NUM": str(m["vid_max_tok"]),
            "VIDEO_MIN_TOKEN_NUM": str(m["vid_min_tok"]),
            "FORCE_QWENVL_VIDEO_READER": "torchvision"}


def build_command(cfg: dict, dataset, output_dir, nproc_per_node: int = 1, nnodes: int = 1,
                  attn_impl: str = "flash_attn", max_steps: int | None = None,
                  max_frames: int | None = None, extra=()) -> tuple[list[str], dict]:
    """(argv, env) for ``swift sft``."""
    world = nproc_per_node * nnodes
    pdbs = int(cfg.get("per_device_train_batch_size", 1))
    gas = grad_accum(int(cfg["global_batch_size"]), world, pdbs)
    epochs = int(cfg["num_train_epochs"])
    max_steps = max_steps if max_steps is not None else cfg.get("max_steps")
    argv = [
        "swift", "sft",
        "--model", str(cfg.get("model", "Qwen/Qwen3.5-9B")), "--model_type", "qwen3_5",
        "--dataset", str(dataset), "--split_dataset_ratio", "0",
        "--tuner_type", "lora", "--lora_rank", "64", "--lora_alpha", "128",
        "--lora_dropout", "0.05", "--target_modules", "all-linear",
        "--freeze_vit", "false", "--freeze_aligner", "false", "--freeze_llm", "false",
        "--torch_dtype", "bfloat16", "--attn_impl", attn_impl,
        "--gradient_checkpointing", "true",
        "--num_train_epochs", str(epochs), "--learning_rate", str(cfg["learning_rate"]),
        "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.03",
        "--weight_decay", "0.1", "--adam_beta2", "0.95", "--max_grad_norm", "1.0",
        "--optim", "adamw_torch_fused",
        "--max_length", str(cfg.get("max_length", REGIME_META[cfg["regime"]]["max_length"])),
        "--truncation_strategy", "delete",
        "--per_device_train_batch_size", str(pdbs), "--gradient_accumulation_steps", str(gas),
        "--eval_strategy", "no",
        "--logging_steps", "5", "--logging_first_step", "true",
        "--dataloader_num_workers", "4", "--report_to", "tensorboard", "--seed", "42",
        "--output_dir", str(output_dir),
    ]
    if max_steps:       # smoke runs: stop early and save once at the end
        argv += ["--max_steps", str(max_steps), "--save_strategy", "steps",
                 "--save_steps", str(max_steps)]
    else:               # a checkpoint per epoch, all kept
        argv += ["--save_strategy", "epoch", "--save_total_limit", str(epochs + 1)]
    argv += list(extra)
    env = dict(sampling_env(cfg["regime"], max_frames))
    env.update(USE_HF="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if world > 1:       # ms-swift launches torchrun when NPROC_PER_NODE is set
        env.update(NPROC_PER_NODE=str(nproc_per_node), NNODES=str(nnodes))
    for k in ("NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        if k in os.environ:
            env[k] = os.environ[k]
    return argv, env
