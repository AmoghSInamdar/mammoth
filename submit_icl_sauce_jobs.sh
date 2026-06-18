#!/bin/bash
# Copyright 2026-present, Amogh Inamdar, Vici Milenia, Richard Zemel.
# Submit in-context SAUCE evaluation jobs with per-(backend,mode,dataset,task)
# fan-out, so the full grid (all tasks x all datasets) parallelizes across the
# cluster instead of running serially in one job.
#
# Each job runs eval_checkpoints.py for ONE task and writes its CSV into a
# per-task temp dir; after all of a (backend,mode,dataset)'s task jobs finish,
# run merge_icl_task_csvs.py to concatenate them into the single
# evaluation_results_icl-<tag>_<dataset>.csv that SAUCE/plotting expect (SAUCE
# groups by (checkpoint_id, eval_task_id), so every task must land in one file).
#
# Usage:
#   ./submit_icl_sauce_jobs.sh [BACKEND] [DATASETS] [N_SEEDS] [GRANULARITY]
#     BACKEND      : clip|dinov2|vit|vlm|qwen2vl|llava|all   (default: vit)
#     DATASETS     : space-separated list, or "all"          (default: all)
#     N_SEEDS      : number of support-sampling seeds         (default: 5)
#     GRANULARITY  : per-task | per-dataset                   (default: per-task)
#
# Env knobs:
#   K_VALUES (default 0,1,2,5,10), VLM_MAX_QUERIES (50), VLM_BATCH_SIZE (8 on GPU),
#   ANON_MODES ("named anon" for generative VLMs; "named" otherwise),
#   OUT_DIR (results/k_shot_evaluation_multirun)
#
# Examples:
#   ./submit_icl_sauce_jobs.sh qwen2vl all 5 per-task   # full grid, fanned out per task
#   ./submit_icl_sauce_jobs.sh dinov2 "seq-cifar100 seq-tinyimg" 5 per-dataset

BACKEND_ARG="${1:-vit}"
DATASETS_ARG="${2:-all}"
N_SEEDS="${3:-5}"
GRANULARITY="${4:-per-task}"
K_VALUES="${K_VALUES:-0,1,2,5,10}"
VLM_MAX_QUERIES="${VLM_MAX_QUERIES:-50}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-results/k_shot_evaluation_multirun}"

# N_TASKS per dataset (fixed dataset constants; see datasets/*.py).
declare -A NTASKS=(
  [seq-mnist]=5 [rot-mnist]=20 [smooth-mnist]=20 [seq-cifar100]=10
  [seq-cifar100-20task]=20 [struct-cifar100]=20 [seq-tinyimg]=10
)

if [ "$BACKEND_ARG" = "all" ]; then
    BACKENDS="clip dinov2 vit qwen2vl llava"   # excludes paid 'vlm' by default
else
    BACKENDS="$BACKEND_ARG"
fi

if [ "$DATASETS_ARG" = "all" ]; then
    DATASETS="seq-mnist rot-mnist smooth-mnist seq-cifar100 seq-cifar100-20task struct-cifar100 seq-tinyimg"
else
    DATASETS="$DATASETS_ARG"
fi

is_generative_vlm() { case "$1" in qwen2vl|llava|vlm) return 0;; *) return 1;; esac; }

submit_cell() {  # backend mode dataset task_spec out_subdir
    local BACKEND="$1" MODE="$2" DATASET="$3" TASKS="$4" SUBDIR="$5"
    local ANON_FLAG=""; [ "$MODE" = "anon" ] && ANON_FLAG="--icl_anonymize"
    local TASK_FLAG=""; [ "$TASKS" != "all" ] && TASK_FLAG="--eval_tasks ${TASKS}"

    # Resources: feature backends + local VLMs need a GPU; Claude 'vlm' is API/CPU.
    local GRES EXTRA=""
    if [ "$BACKEND" = "vlm" ]; then
        GRES="#SBATCH --cpus-per-task=4"
        EXTRA="# Requires ANTHROPIC_API_KEY and 'pip install anthropic'."
    else
        GRES="#SBATCH --gres=gpu:A6000:1"$'\n'"#SBATCH --cpus-per-task=4"
    fi
    # Batch size only matters for generative VLMs; harmless otherwise.
    local BS_FLAG=""
    is_generative_vlm "$BACKEND" && BS_FLAG="--icl_vlm_batch_size ${VLM_BATCH_SIZE}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=icl_${BACKEND}_${MODE}_${DATASET}_${SUBDIR##*/}
#SBATCH --output=logs/icl/${BACKEND}_${MODE}_${DATASET}_${SUBDIR##*/}_%j.out
#SBATCH --error=logs/icl/${BACKEND}_${MODE}_${DATASET}_${SUBDIR##*/}_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --mem=64G
${GRES}
mkdir -p logs/icl ${SUBDIR}
source .mammoth/bin/activate
${EXTRA}
python eval_checkpoints.py \
    --icl_backend ${BACKEND} ${ANON_FLAG} ${BS_FLAG} \
    --dataset ${DATASET} --eval_dataset ${DATASET} \
    --lr 0.0 --k_values ${K_VALUES} --n_seeds ${N_SEEDS} \
    --num_workers 0 --vlm_max_queries ${VLM_MAX_QUERIES} ${TASK_FLAG} \
    --multirun_temp_csv_dir ${SUBDIR}
EOF
    echo "Submitted: ${BACKEND}/${MODE} ${DATASET} tasks=${TASKS} -> ${SUBDIR}"
}

for BACKEND in $BACKENDS; do
    # Anonymized ablation only applies to generative VLMs.
    if is_generative_vlm "$BACKEND"; then
        MODES="${ANON_MODES:-named anon}"
    else
        MODES="named"
    fi
    [ "$BACKEND" = "vlm" ] && echo "NOTE: vlm is API-billed — run 'python estimate_vlm_cost.py' first."

    for MODE in $MODES; do
        for DATASET in $DATASETS; do
            if [ "$GRANULARITY" = "per-dataset" ]; then
                # One job per (backend,mode,dataset): writes directly to OUT_DIR.
                submit_cell "$BACKEND" "$MODE" "$DATASET" "all" "$OUT_DIR"
            else
                # One job PER TASK -> per-task subdir; merge afterwards.
                NT="${NTASKS[$DATASET]:-1}"
                TAG="${BACKEND}"; [ "$MODE" = "anon" ] && TAG="${BACKEND}-anon"
                for ((t=0; t<NT; t++)); do
                    submit_cell "$BACKEND" "$MODE" "$DATASET" "$t" \
                        "${OUT_DIR}/_tasks/icl-${TAG}_${DATASET}/task${t}"
                done
                echo "  -> after these ${NT} task jobs finish, merge with:"
                echo "     python merge_icl_task_csvs.py --tag icl-${TAG}_${DATASET} \\"
                echo "        --tasks_root ${OUT_DIR}/_tasks --out_dir ${OUT_DIR}"
            fi
        done
    done
done
