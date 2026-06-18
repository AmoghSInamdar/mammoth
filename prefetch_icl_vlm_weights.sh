#!/bin/bash
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# Pre-download the open-source VLM weights ONCE into the shared HF cache, as a
# SLURM job, before launching the per-task ICL grid. Running this first avoids
# hundreds of fanned-out jobs racing to download the same ~5GB of weights
# simultaneously (cache thrash / HF rate limits / partial-download corruption).
#
# Submit:   sbatch prefetch_icl_vlm_weights.sh
# Then wait for it to finish (check: squeue -u $USER) before submitting the grid.
#
# Edit the SBATCH headers below to match Insomnia (account / partition / gres).
# Needs internet egress on the compute node; if compute nodes are offline, run
# the python block on a node that has internet instead.
#SBATCH --job-name=icl_prefetch
#SBATCH --output=logs/icl/prefetch_%j.out
#SBATCH --error=logs/icl/prefetch_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p logs/icl
source .mammoth/bin/activate

python - <<'PY'
from transformers import (
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
)

MODELS = [
    "Qwen/Qwen2-VL-2B-Instruct",
    "llava-hf/llava-interleave-qwen-0.5b-hf",
]
for m in MODELS:
    print(f"caching processor: {m}", flush=True)
    AutoProcessor.from_pretrained(m)

print("caching Qwen2-VL-2B weights...", flush=True)
Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("caching LLaVA-Interleave-0.5B weights...", flush=True)
LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
print("cached all VLM weights -> ~/.cache/huggingface", flush=True)
PY
