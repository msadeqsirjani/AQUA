#!/usr/bin/env bash
# =============================================================================
# Run AQUA v2 for all (dataset, model, bit-target) combinations.
#
# Usage:
#   bash scripts/run_all_aqua.sh              # run all 54 experiments sequentially
#   bash scripts/run_all_aqua.sh --gpu 0      # pin to GPU 0
#   bash scripts/run_all_aqua.sh --parallel 4 # run 4 jobs in parallel across GPUs 0-3
#   bash scripts/run_all_aqua.sh --dry-run    # print commands without executing
# =============================================================================
set -euo pipefail

DATASETS=("cifar10" "cifar100" "tiny_imagenet")
MODELS=("resnet18" "resnet34" "vgg11" "vgg16" "mobilenetv2" "vit_tiny")
BITS=(4 6 8)

GPU=-1          # -1 = auto-distribute, >=0 = pin to that GPU
PARALLEL=1      # number of parallel jobs
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "=== AQUA v2 Full Sweep ==="
echo "  Datasets : ${DATASETS[*]}"
echo "  Models   : ${MODELS[*]}"
echo "  Bits     : ${BITS[*]}"
echo "  GPUs     : ${NUM_GPUS}"
echo "  Parallel : ${PARALLEL}"
echo "  Total    : $(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#BITS[@]} )) runs"
echo "=========================="

JOB_IDX=0

run_one() {
    local dataset="$1" model="$2" bits="$3"
    local config="configs/${dataset}_${model}_aqua.yaml"
    local tag="${bits}bit"

    if [[ $GPU -ge 0 ]]; then
        local gpu_id=$GPU
    else
        local gpu_id=$(( JOB_IDX % NUM_GPUS ))
    fi

    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m scripts.aqua.train --config ${config} --target-avg-bits ${bits} --run-tag ${tag}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] ${cmd}"
        return 0
    fi

    echo ""
    echo ">>> [Job $((JOB_IDX+1))] ${dataset} / ${model} / ${bits}-bit  (GPU ${gpu_id})"
    echo ">>> ${cmd}"
    eval "${cmd}"
    echo ">>> [Job $((JOB_IDX+1))] DONE: ${dataset} / ${model} / ${bits}-bit"
}

if [[ $PARALLEL -le 1 ]]; then
    # Sequential execution
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for bits in "${BITS[@]}"; do
                run_one "$dataset" "$model" "$bits"
                JOB_IDX=$((JOB_IDX + 1))
            done
        done
    done
else
    # Parallel execution with GNU parallel or background jobs
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for bits in "${BITS[@]}"; do
                if [[ $GPU -ge 0 ]]; then
                    gpu_id=$GPU
                else
                    gpu_id=$(( JOB_IDX % NUM_GPUS ))
                fi

                config="configs/${dataset}_${model}_aqua.yaml"
                tag="${bits}bit"
                cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m scripts.aqua.train --config ${config} --target-avg-bits ${bits} --run-tag ${tag}"

                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[DRY-RUN] ${cmd}"
                else
                    echo ">>> [Job $((JOB_IDX+1))] ${dataset} / ${model} / ${bits}-bit  (GPU ${gpu_id})"
                    eval "${cmd}" &
                fi

                JOB_IDX=$((JOB_IDX + 1))

                # Throttle: wait if we've hit the parallel limit
                if (( JOB_IDX % PARALLEL == 0 )); then
                    wait
                fi
            done
        done
    done
    wait
fi

echo ""
echo "=== All ${JOB_IDX} AQUA v2 runs complete ==="
