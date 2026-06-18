#!/bin/bash
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# Merge all per-task in-context SAUCE CSVs produced by the per-task SLURM
# fan-out (submit_icl_sauce_jobs.sh ... per-task) into the single
# evaluation_results_icl-<tag>_<dataset>.csv files that SAUCE/plotting expect.
#
# Run this AFTER all task jobs for a sweep have finished. It loops every
# (backend, mode, dataset) group, finds its per-task subdirs, and calls
# merge_icl_task_csvs.py (which concatenates the disjoint-task rows and
# recomputes the four plasticity columns on the union).
#
# Usage:
#   ./merge_all_icl.sh [BACKENDS] [DATASETS] [MODES]
#     BACKENDS : space-separated (default: "qwen2vl llava")
#     DATASETS : space-separated, or "all" (default: all)
#     MODES    : space-separated subset of "named anon" (default: "named anon")
#
# Env:
#   OUT_DIR (default results/k_shot_evaluation_multirun) -- must match the
#           --multirun_temp_csv_dir / OUT_DIR used at submit time.
#
# Examples:
#   ./merge_all_icl.sh                              # qwen2vl+llava, named+anon, all datasets
#   ./merge_all_icl.sh qwen2vl seq-cifar100 named   # one group only

set -u

BACKENDS="${1:-qwen2vl llava}"
DATASETS_ARG="${2:-all}"
MODES="${3:-named anon}"
OUT_DIR="${OUT_DIR:-results/k_shot_evaluation_multirun}"
ROOT="${OUT_DIR}/_tasks"

if [ "$DATASETS_ARG" = "all" ]; then
    DATASETS="seq-mnist rot-mnist smooth-mnist seq-cifar100 seq-cifar100-20task struct-cifar100 seq-tinyimg"
else
    DATASETS="$DATASETS_ARG"
fi

PY="$(command -v python || echo python)"

merged=0; missing=0
for BACKEND in $BACKENDS; do
    for MODE in $MODES; do
        SUFFIX=""; [ "$MODE" = "anon" ] && SUFFIX="-anon"
        for DATASET in $DATASETS; do
            TAG="icl-${BACKEND}${SUFFIX}_${DATASET}"
            if [ -d "${ROOT}/${TAG}" ]; then
                echo "=== merging ${TAG} ==="
                "$PY" merge_icl_task_csvs.py --tag "$TAG" \
                    --tasks_root "$ROOT" --out_dir "$OUT_DIR" && merged=$((merged+1))
            else
                echo "--- skip ${TAG}: no per-task dir at ${ROOT}/${TAG} ---"
                missing=$((missing+1))
            fi
        done
    done
done

echo
echo "Done: ${merged} groups merged, ${missing} groups missing/skipped."
echo "Merged files: ${OUT_DIR}/aggregated/evaluation_results_icl-*.csv"
