# SurgScope challenge container (ORena SAVE FOCUS 2026, PROCEDURE track).
# Build on an amd64 host with network access:
#   python challenge/fetch_resources.py              # base model + adapter, merged -> challenge/resources/SurgScope-9B
#   docker build --platform linux/amd64 -t surgscope-procedure challenge/
# Run like the evaluation platform (offline, read-only input):
#   docker run --gpus all --network none -v $PWD/examples/test:/input:ro -v $PWD/out:/output surgscope-procedure
# cu12.8 kernels cover the L40S (sm_89), H100 (sm_90) and RTX PRO 6000 (sm_120) evaluation GPUs.
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    FPS_MIN_FRAMES=1536 \
    FPS_MAX_FRAMES=1536 \
    VIDEO_MAX_TOKEN_NUM=128 \
    VIDEO_MIN_TOKEN_NUM=32 \
    FORCE_QWENVL_VIDEO_READER=torchvision \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HOME=/tmp/hf MODELSCOPE_CACHE=/tmp/ms MPLCONFIGDIR=/tmp/mpl TRITON_CACHE_DIR=/tmp/triton HOME=/tmp

RUN groupadd -r user && useradd -m --no-log-init -r -g user -u 1000 user
WORKDIR /opt/app
COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt
# Files must be readable by uid 1000: unreadable model files surface as a misleading
# "Unrecognized model ... should have a model_type key" error in transformers.
COPY --chown=user:user resources/ /opt/app/resources/
COPY --chown=user:user proc_policy.py inference.py /opt/app/
USER user
ENTRYPOINT ["python", "inference.py"]
