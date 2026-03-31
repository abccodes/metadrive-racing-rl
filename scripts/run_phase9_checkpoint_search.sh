#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-3000000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase9_checkpoint_search}"
RUN_NAME="p9_checkpoint_search"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

train_base() {
  python train.py \
    --run-name "${RUN_NAME}" \
    --export-agent-dir "agents/agent_${RUN_NAME}_final_raw" \
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
    --early-progress-weight 70.0
}

export_wrap_and_bench() {
  local ckpt_label="$1"
  local ckpt_path="$2"
  local raw_agent_dir="agents/agent_${RUN_NAME}_${ckpt_label}_raw"
  local launch_agent_dir="agents/agent_${RUN_NAME}_${ckpt_label}"

  python export_sb3_checkpoint.py \
    --checkpoint "${ckpt_path}" \
    --output-dir "${raw_agent_dir}"

  python build_hybrid_launch_submission.py \
    --base-agent-dir "${raw_agent_dir}" \
    --output-dir "${launch_agent_dir}" \
    --launch-steps 32 \
    --launch-min-throttle 1.0 \
    --launch-final-min-throttle 0.25 \
    --launch-steer-scale 0.70

  python benchmark.py \
    --agent-dir "${launch_agent_dir}" \
    --map-split server \
    --opponent-preset learned \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${ckpt_label}/learned"

  python benchmark.py \
    --agent-dir "${launch_agent_dir}" \
    --map-split server \
    --opponent-preset full \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${ckpt_label}/full"
}

train_base

SELECTED_STEPS=(
  2000000
  2250000
  2500000
  2750000
  3000000
)

for step in "${SELECTED_STEPS[@]}"; do
  ckpt="${CHECKPOINT_DIR}/racing_ppo_${step}_steps.zip"
  if [[ -f "${ckpt}" ]]; then
    export_wrap_and_bench "ckpt_${step}" "${ckpt}"
  else
    echo "Skipping missing checkpoint ${ckpt}"
  fi
done

if [[ -f "${CHECKPOINT_DIR}/best/best_model.zip" ]]; then
  export_wrap_and_bench "best_model" "${CHECKPOINT_DIR}/best/best_model.zip"
fi

if [[ -f "${CHECKPOINT_DIR}/racing_ppo_final.zip" ]]; then
  export_wrap_and_bench "final" "${CHECKPOINT_DIR}/racing_ppo_final.zip"
fi
