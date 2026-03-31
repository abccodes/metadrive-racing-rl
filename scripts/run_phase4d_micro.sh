#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-2000000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase4d_micro}"

run_experiment() {
  local run_name="$1"
  shift

  echo
  echo "============================================================"
  echo "Running ${run_name}"
  echo "============================================================"

  python train.py \
    --run-name "${run_name}" \
    --export-agent-dir "agents/agent_${run_name}" \
    --total-timesteps "${TIMESTEPS}" \
    --num-train-envs "${TRAIN_ENVS}" \
    --num-eval-envs "${EVAL_ENVS}" \
    --train-map-split server \
    --eval-map-split server \
    --opponent-pool still random \
    --pi-layers 512 256 \
    --vf-layers 512 256 \
    --lr 3e-4 \
    --lr-schedule linear \
    --n-steps 512 \
    --batch-size 256 \
    --n-epochs 10 \
    --clip-range 0.2 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --target-kl 0.03 \
    "$@"

  python benchmark.py \
    --agent-dir "agents/agent_${run_name}" \
    --map-split server \
    --opponent-preset learned \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${run_name}/learned"

  python benchmark.py \
    --agent-dir "agents/agent_${run_name}" \
    --map-split server \
    --opponent-preset full \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${run_name}/full"
}

run_experiment p4d_progress60_sustain_lowent \
  --ent-coef 0.005 \
  --progress-reward-weight 10.0 \
  --early-reward-horizon 300 \
  --early-progress-weight 60.0

run_experiment p4d_progress70_sustain \
  --ent-coef 0.0075 \
  --progress-reward-weight 10.0 \
  --early-reward-horizon 300 \
  --early-progress-weight 70.0

run_experiment p4d_progress60_sustain_moreglobal \
  --ent-coef 0.0075 \
  --progress-reward-weight 15.0 \
  --early-reward-horizon 300 \
  --early-progress-weight 60.0

run_experiment p4d_progress70_sustain_lowent \
  --ent-coef 0.005 \
  --progress-reward-weight 10.0 \
  --early-reward-horizon 300 \
  --early-progress-weight 70.0
