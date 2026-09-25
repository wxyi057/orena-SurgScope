# Reproduction

**Metric.** Accuracy per question inside each (capability group × in-/out-of-distribution) bucket,
averaged over the populated buckets. The official test set (1,000 questions) fills ten buckets;
the public test split (3,127 questions) is in-distribution only and fills five.
`surgscope evaluate` wraps the official [`orena-focus`](https://github.com/IMSY-DKFZ/orena-focus)
evaluator.

**Submitted container.** `surgscope infer` and [`challenge/inference.py`](../challenge/inference.py)
reproduce the submission's answers byte for byte (20 / 20 on a test batch), with the pinned
`imageio-ffmpeg`. Peak GPU memory 29.5 GB.

**Weights.** `soup(r768-ep4, dense-mix-ep14, dense-mix-2-ep6)` equals the released adapter bit for
bit; merged into Qwen3.5-9B, at load time or with `surgscope merge`, it gives the weights of the
submitted image tensor for tensor ([TRAINING.md](TRAINING.md)).

**Routing.** Windows, frame budgets and prompts of all 6,000 HeiCo-FOCUS questions are pinned in
`tests/golden/`:

```bash
SURGSCOPE_DATA_ROOTS="heico=data/heico-focus-vqa" pytest -m golden
```

**Public test split.** `infer-dataset` reads the mezzanines of `surgscope transcode` directly
(one process per GPU):

```bash
DATA="--data heico=data/heico-focus-vqa --data lapchole=data/lapchole-focus-vqa"
surgscope transcode $DATA --work work --splits test
for K in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$K surgscope infer-dataset $DATA --work work --split test \
      --shard $K/4 --output answers.$K.json &
done; wait
surgscope evaluate --pred answers.*.json --reference data/*/data/procedure/test.parquet
```
