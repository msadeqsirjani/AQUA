#!/bin/bash
# Run CIFAR-100 baseline for both models across all dtypes.
# Distributes 12 jobs round-robin across 8 GPUs.

set -e

DTYPES=(fp32 fp16 int8 int4 int2 int1)
MODELS=(simplecnn5 resnet18)
NUM_GPUS=8

gpu=0
pids=()

for model in "${MODELS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
        echo "[GPU $gpu] Starting: $model / cifar100 / $dtype"
        CUDA_VISIBLE_DEVICES=$gpu python -m scripts.train \
            --dataset cifar100 \
            --model "$model" \
            --mode baseline \
            --dtype "$dtype" \
            > "results/baseline_${model}_cifar100_${dtype}.log" 2>&1 &
        pids+=($!)
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done
done

echo ""
echo "Launched ${#pids[@]} jobs across $NUM_GPUS GPUs. Waiting..."
echo ""

# Wait for all and report
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

if [ $failed -eq 0 ]; then
    echo "All ${#pids[@]} jobs completed successfully."
else
    echo "$failed / ${#pids[@]} jobs failed. Check logs in results/"
fi
