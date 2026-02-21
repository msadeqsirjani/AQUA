#!/bin/bash
# Run all AQUA pretrain configs distributed across 8 GPUs.
# Uses a job queue: as soon as a GPU finishes, it immediately picks up the next job.
# Failed jobs are logged but don't block other jobs.

# Ignore SIGHUP so the scheduler survives terminal disconnection
trap '' HUP

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

COMMANDS=(
  "python -m scripts.pretrain --config configs/cifar10_resnet18_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_resnet18_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_resnet18_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar10_resnet34_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_resnet34_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_resnet34_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar10_vgg11_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_vgg11_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_vgg11_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar10_vgg16_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_vgg16_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_vgg16_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar10_mobilenetv2_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_mobilenetv2_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_mobilenetv2_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar10_vit_tiny_aqua.yaml"
  "python -m scripts.pretrain --config configs/cifar100_vit_tiny_aqua.yaml"
  "python -m scripts.pretrain --config configs/tiny_imagenet_vit_tiny_aqua.yaml"
)

total=${#COMMANDS[@]}
next_job=0       # index of next job to dispatch
running=0        # count of currently running jobs
failed=0         # count of failed jobs
declare -A gpu_pid   # gpu -> pid
declare -A gpu_cmd   # gpu -> command description
declare -A pid_gpu   # pid -> gpu

mkdir -p logs

launch_on_gpu() {
  local gpu=$1
  if (( next_job >= total )); then
    return 1
  fi
  local cmd="${COMMANDS[$next_job]}"
  local config
  config=$(echo "$cmd" | sed -n 's/.*--config[= ]\([^ ]*\).*/\1/p')
  local log_suffix
  log_suffix=$(basename "$config" .yaml)
  local log_file="logs/pretrain_${log_suffix}.log"

  nohup env CUDA_VISIBLE_DEVICES=$gpu $cmd >> "$log_file" 2>&1 &
  local pid=$!
  gpu_pid[$gpu]=$pid
  gpu_cmd[$gpu]="$cmd"
  pid_gpu[$pid]=$gpu
  ((next_job++))
  ((running++))
  echo "[$(date '+%H:%M:%S')] GPU $gpu: started job $next_job/$total — $cmd (log: $log_file)"
  return 0
}

echo "Running $total pretrain jobs on $NUM_GPUS GPUs (queue mode — immediate replacement)."
echo ""

# Fill all GPUs initially
for ((gpu = 0; gpu < NUM_GPUS && next_job < total; gpu++)); do
  launch_on_gpu $gpu
done

# Wait for any job to finish, then immediately backfill that GPU
while (( running > 0 )); do
  # Wait for any child to exit
  wait -n -p finished_pid 2>/dev/null
  exit_code=$?

  # Find which GPU finished
  if [[ -n "${pid_gpu[$finished_pid]+x}" ]]; then
    gpu=${pid_gpu[$finished_pid]}
    cmd_desc="${gpu_cmd[$gpu]}"
    unset "pid_gpu[$finished_pid]"
    unset "gpu_pid[$gpu]"
    unset "gpu_cmd[$gpu]"
    ((running--))

    if (( exit_code != 0 )); then
      echo "[$(date '+%H:%M:%S')] GPU $gpu: FAILED (exit $exit_code) — $cmd_desc"
      ((failed++))
    else
      echo "[$(date '+%H:%M:%S')] GPU $gpu: completed — $cmd_desc"
    fi

    # Immediately launch next job on the freed GPU
    launch_on_gpu $gpu || true
  else
    # Fallback: if wait -n doesn't give us the pid (older bash), poll
    for gpu in "${!gpu_pid[@]}"; do
      pid=${gpu_pid[$gpu]}
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        exit_code=$?
        cmd_desc="${gpu_cmd[$gpu]}"
        unset "pid_gpu[$pid]"
        unset "gpu_pid[$gpu]"
        unset "gpu_cmd[$gpu]"
        ((running--))

        if (( exit_code != 0 )); then
          echo "[$(date '+%H:%M:%S')] GPU $gpu: FAILED (exit $exit_code) — $cmd_desc"
          ((failed++))
        else
          echo "[$(date '+%H:%M:%S')] GPU $gpu: completed — $cmd_desc"
        fi

        launch_on_gpu $gpu || true
        break
      fi
    done
    # Small sleep to avoid busy-waiting in fallback path
    sleep 2
  fi
done

echo ""
echo "All $total jobs dispatched. Failures: $failed."
if (( failed > 0 )); then
  echo "Check logs/ for details on failed jobs."
  exit 1
fi
echo "All pretrain jobs finished successfully."
