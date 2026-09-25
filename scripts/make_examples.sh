#!/usr/bin/env bash
# Cut two cases in the challenge input layout from the official HeiCo-FOCUS-VQA data:
#   examples/test   2101639  video 0020, first 31 min  "final retrieval of a Sponge"
#   examples/train  1488678  video 0013, first 11 min  "how many Clips are inserted"
# Request access on https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa and log in
# (`hf auth login`) first. The clock overlay needs an ffmpeg with drawtext on PATH.
set -euo pipefail
OUT=${1:-examples}
DATA=${DATA:-data/heico-focus-vqa}
WORK=${WORK:-work}
REV=4ee0e4b39ee59006b773beec501bb47e251827eb
TEST_VIDEO="0020 - Heico - Sigma - 1"
TRAIN_VIDEO="0013 - Heico - Rektum - 4"

hf download orena-dkfz/heico-focus-vqa --repo-type dataset --revision $REV --local-dir "$DATA" \
    data/procedure/train.parquet data/procedure/test.parquet \
    "videos/$TEST_VIDEO.avi" "videos/$TRAIN_VIDEO.avi"
surgscope transcode --data heico="$DATA" --work "$WORK" --videos "$TEST_VIDEO" "$TRAIN_VIDEO"
surgscope make-input --data heico="$DATA" --work "$WORK" --split test --ids 2101639 --out "$OUT/test"
surgscope make-input --data heico="$DATA" --work "$WORK" --split train --ids 1488678 --out "$OUT/train"
