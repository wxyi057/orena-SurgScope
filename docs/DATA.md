# Data

SurgScope uses only the PROCEDURE training split (6,873 questions) of
[HeiCo-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and
[LapChole-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) (gated; accept
the terms on the dataset pages).

```bash
hf download orena-dkfz/heico-focus-vqa --repo-type dataset --local-dir data/heico-focus-vqa
hf download orena-dkfz/lapchole-focus-vqa --repo-type dataset --local-dir data/lapchole-focus-vqa
DATA="--data heico=data/heico-focus-vqa --data lapchole=data/lapchole-focus-vqa"

surgscope transcode $DATA --work work --procs 32                       # mezzanines + dead-frame stats
for R in r768 dense-mix dense-mix-2; do
  surgscope prepare $DATA --work work --regime $R --procs 32            # clips + training rows
done
```

Everything derived goes to `work/`:

```
mezz/<ds>/{plain,overlayed}/<video>.mp4    5 fps, height ≤ 576, H.264, HH:MM:SS clock burned in
stats/<ds>/<video>.npz                     per-frame dead-frame statistics
clips/frames<N>/<ds>/<split>/<id>.mp4      training mini-clips (+ .json sidecar)
sft/<regime>.jsonl                         ms-swift training rows
```

- **Clock.** `transcode` needs an ffmpeg with `drawtext` (libfreetype), e.g. from conda-forge; it
  is picked from `PATH`, `--ffmpeg` or `SURGSCOPE_FFMPEG`. Inference and training use the pinned
  `imageio-ffmpeg`.
- **Multi-node.** Add `--shard K/N` to both steps, then run `prepare ... --jsonl-only` once
  (see `scripts/slurm/prepare_array.sbatch`).
- **Challenge layout.** `surgscope make-input --data heico=data/heico-focus-vqa --work work --split test
  --ids 2101639 --out my_input` writes `request.json`, `FO_definitions.json`, `plain/` and
  `overlayed/` clips and `reference.parquet`.

## Examples

[`scripts/make_examples.sh`](../scripts/make_examples.sh) downloads two HeiCo videos and cuts
two cases in the challenge input layout into `examples/{train,test}`:

| Split | Id | Prefix | Question | Answer |
|---|---|---|---|---|
| train | 1488678 | 00:10:49 | How many Clip(s) are inserted in the abdomen in this video? | 2 |
| test | 2101639 | 00:31:09 | At what time point was the final and last retrieval of a Sponge in the video? | 00:24:40 |
