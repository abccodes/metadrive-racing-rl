#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-300000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"

RUN_NAME="p8b_league_launch"
RAW_EXPORT_DIR="agents/agent_${RUN_NAME}_raw"
LAUNCH_EXPORT_DIR="agents/agent_${RUN_NAME}"
INIT_MODEL="checkpoints/p4d_progress70_sustain_lowent/best/best_model.zip"

LEAGUE_AGENTS=(
  "agents/agent_phase6_launch_long"
  "agents/agent_p4d_progress70_sustain_lowent"
  "agents/agent_p4c_pace_progress60_sustain"
  "agents/agent_p4b_server_progress"
)

for agent_dir in "${LEAGUE_AGENTS[@]}"; do
  if [[ ! -d "${agent_dir}" ]]; then
    echo "Missing league agent directory: ${agent_dir}"
    exit 1
  fi
done

if [[ ! -f "${INIT_MODEL}" ]]; then
  echo "Missing init model checkpoint: ${INIT_MODEL}"
  exit 1
fi

python train.py \
  --run-name "${RUN_NAME}" \
  --init-model "${INIT_MODEL}" \
  --export-agent-dir "${RAW_EXPORT_DIR}" \
  --total-timesteps "${TIMESTEPS}" \
  --num-train-envs "${TRAIN_ENVS}" \
  --num-eval-envs "${EVAL_ENVS}" \
  --train-map-split server \
  --eval-map-split server \
  --opponent-pool still random aggressive baseline example \
  --opponent-agent-dirs "${LEAGUE_AGENTS[@]}" \
  --pi-layers 512 256 \
  --vf-layers 512 256 \
  --lr 3e-5 \
  --lr-schedule constant \
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

python build_hybrid_launch_submission.py \
  --base-agent-dir "${RAW_EXPORT_DIR}" \
  --output-dir "${LAUNCH_EXPORT_DIR}" \
  --launch-steps 32 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.25 \
  --launch-steer-scale 0.70

bash run_server_benchmark.sh "${LAUNCH_EXPORT_DIR}" "${BENCH_EPISODES}"
