# Challenge container

`inference.py` is the script of our submitted image (pre-evaluation score 0.6592), with comments
and names rewritten. It reads `/input/request.json`, `/input/FO_definitions.json` and
`/input/overlayed/<qID>.mp4`, and writes `/output/answer.json`.

```bash
python challenge/fetch_resources.py        # base model + adapter, merged
docker build --platform linux/amd64 -t surgscope-procedure challenge/
docker run --rm --gpus all --network none -v $PWD/examples/test:/input:ro -v $PWD/out:/output surgscope-procedure
```
