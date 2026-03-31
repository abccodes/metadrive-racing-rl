#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-2000000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase10_guidance}"

run_experiment() {
  local run_name="$1"
  shift

  echo
  echo "============================================================"
  echo "Running ${run_name}"
  echo "============================================================"

  python train.py \
    --run-name "${run_name}" \
    --export-agent-dir "agents/agent_${run_name}_raw" \
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
    --ent-coef 0.005 \
    --progress-reward-weight 10.0 \
    --early-reward-horizon 300 \
    --early-progress-weight 70.0 \
    "$@"

  python build_hybrid_launch_submission.py \
    --base-agent-dir "agents/agent_${run_name}_raw" \
    --output-dir "agents/agent_${run_name}" \
    --launch-steps 32 \
    --launch-min-throttle 1.0 \
    --launch-final-min-throttle 0.25 \
    --launch-steer-scale 0.70

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

run_experiment p10_guidance_obs \
  --use-track-guidance

run_experiment p10_guidance_line_speed \
  --use-track-guidance \
  --line-progress-reward-weight 2.0 \
  --line-speed-reward-weight 0.5

run_experiment p10_guidance_line_speed_early \
  --use-track-guidance \
  --line-progress-reward-weight 2.0 \
  --line-speed-reward-weight 0.75 \
  --line-center-penalty-weight 0.1
