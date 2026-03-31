#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TIMESTEPS="${TIMESTEPS:-2000000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints}"
LOG_ROOT="${LOG_ROOT:-logs}"
BENCH_ROOT="${BENCH_ROOT:-benchmark_results/phase4a}"

COMMON_ARGS=(
  --total-timesteps "$TIMESTEPS"
  --num-train-envs "$TRAIN_ENVS"
  --num-eval-envs "$EVAL_ENVS"
  --train-map-split train
  --eval-map-split validation
  --opponent-pool aggressive baseline example selfplay
  --selfplay-snapshot-freq 100000
  --selfplay-max-snapshots 3
  --save-dir "$SAVE_ROOT"
  --log-dir "$LOG_ROOT"
)

run_experiment() {
  local run_name="$1"
  shift

  local agent_dir="agents/agent_${run_name}"

  echo "=== Training ${run_name} ==="
  python train.py \
    --run-name "$run_name" \
    --export-agent-dir "$agent_dir" \
    "${COMMON_ARGS[@]}" \
    "$@"

  echo "=== Benchmarking ${run_name} (learned) ==="
  python benchmark.py \
    --agent-dir "$agent_dir" \
    --map-split validation \
    --opponent-preset learned \
    --output-dir "$BENCH_ROOT/${run_name}/learned"

  echo "=== Benchmarking ${run_name} (full) ==="
  python benchmark.py \
    --agent-dir "$agent_dir" \
    --map-split validation \
    --opponent-preset full \
    --output-dir "$BENCH_ROOT/${run_name}/full"
}

run_experiment \
  p4a_ref \
  --pi-layers 256 256 \
  --vf-layers 256 256 \
  --lr 3e-4 \
  --lr-schedule constant \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 10 \
  --clip-range 0.2 \
  --ent-coef 0.01 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5

run_experiment \
  p4a_big \
  --pi-layers 512 256 \
  --vf-layers 512 256 \
  --lr 3e-4 \
  --lr-schedule constant \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 10 \
  --clip-range 0.2 \
  --ent-coef 0.01 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5

run_experiment \
  p4a_rollout \
  --pi-layers 256 256 \
  --vf-layers 256 256 \
  --lr 3e-4 \
  --lr-schedule constant \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 10 \
  --clip-range 0.2 \
  --ent-coef 0.01 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5

run_experiment \
  p4a_big_linear \
  --pi-layers 512 256 \
  --vf-layers 512 256 \
  --lr 3e-4 \
  --lr-schedule linear \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 10 \
  --clip-range 0.2 \
  --ent-coef 0.01 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --target-kl 0.03
