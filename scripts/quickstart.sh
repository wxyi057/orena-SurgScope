#!/usr/bin/env bash
# End-to-end check on one GPU: build the examples, answer the test case, score it, then run
# two training steps on the train case.
set -euo pipefail
bash scripts/make_examples.sh examples
surgscope infer --input examples/test --output runs/quickstart/answer.json --trace runs/quickstart/trace.json
surgscope evaluate --pred runs/quickstart/answer.json --reference examples/test/reference.parquet --judge none
surgscope prepare --from-input examples/train --regime dense-mix --work runs/quickstart
surgscope train --config configs/train/quickstart.yaml --dataset runs/quickstart/sft/dense-mix.jsonl \
    --output-dir runs/quickstart/run
