# Training

Three LoRA fine-tunes of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) with
[ms-swift](https://github.com/modelscope/ms-swift) 4.3.2, one per sampling regime, averaged
into one model.

| | A · `r768` | B · `dense-mix` | C · `dense-mix-2` |
|---|---|---|---|
| Frames, time / count questions | 768 | 1536 @ 60 tok | 2880 @ 32 tok |
| Frames, other questions | 768 | 1152 @ 84 tok | 1512 @ 64 tok |
| Learning rate | 1.4e-4 | 1e-4 | 1e-4 |
| Global batch | 64 | 32 | 32 |
| Max. length | 57,344 | 57,344 | 65,536 |
| Epoch used | 4 | 14 | 6 |
| Time (GH200 GPUs) | 11.5 h (16) | 15.7 h (32) | 17 h (32) |

Shared: LoRA r = 64, α = 128, dropout 0.05 on all linear layers (vision encoder and merger
included); bf16; gradient checkpointing; FlashAttention-2; AdamW, weight decay 0.1, β₂ 0.95;
cosine schedule, 3 % warm-up; 15 epochs; batch 1 per GPU; seed 42. Peak memory 75.5 GB per GPU.

## Run

```bash
# one node, 8 GPUs (gradient accumulation is derived to keep the global batch)
surgscope train --config configs/train/dense-mix.yaml --dataset work/sft/dense-mix.jsonl \
    --output-dir runs/dense-mix --nproc-per-node 8

# multi-node: on every node, with NODE_RANK, MASTER_ADDR and MASTER_PORT set
surgscope train --config configs/train/dense-mix.yaml --dataset work/sft/dense-mix.jsonl \
    --output-dir runs/dense-mix --nproc-per-node 4 --nnodes 4
```

`--dry-run` prints the `swift sft` command; extra `swift sft` arguments go after `--`.
SLURM example: [`scripts/slurm/train_multinode.sbatch`](../scripts/slurm/train_multinode.sbatch).

## Soup

```bash
surgscope soup --adapters runs/r768/v0-*/checkpoint-432 runs/dense-mix/v0-*/checkpoint-2996 \
    runs/dense-mix-2/v0-*/checkpoint-1284 --out runs/soup
surgscope infer --model runs/soup --input examples/test --output answer.json
```

LoRA A and B are averaged separately (fp32). `surgscope infer` merges the adapter into the base
model (bf16) at load time; `surgscope merge --adapter runs/soup --out runs/SurgScope-9B` writes the
merged model, as used by the [challenge container](../challenge/).

## Smoke test

```bash
surgscope prepare --from-input examples/train --regime dense-mix --work runs/qs
surgscope train --config configs/train/quickstart.yaml --dataset runs/qs/sft/dense-mix.jsonl --output-dir runs/qs/run
```

Two steps: 73.5 GB peak on a GH200, or 29.6 GB with `--max-frames 256` on both commands.
