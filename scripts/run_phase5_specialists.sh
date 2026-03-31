#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-2000000}"
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-2}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase5_specialists}"
COMPOSITE_DIR="${COMPOSITE_DIR:-agents/agent_phase5_specialist_combo}"

train_specialist() {
  local map_name="$1"
  local run_name="$2"

  echo
  echo "============================================================"
  echo "Training ${run_name} on ${map_name}"
  echo "============================================================"

  python train.py \
    --run-name "${run_name}" \
    --export-agent-dir "agents/agent_${run_name}" \
    --total-timesteps "${TIMESTEPS}" \
    --num-train-envs "${TRAIN_ENVS}" \
    --num-eval-envs "${EVAL_ENVS}" \
    --train-maps "${map_name}" \
    --eval-maps "${map_name}" \
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

  python benchmark.py \
    --agent-dir "agents/agent_${run_name}" \
    --maps "${map_name}" \
    --opponent-preset learned \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${run_name}/learned"
}

train_specialist server_map1 phase5_server_map1
train_specialist server_map2 phase5_server_map2
train_specialist server_map3 phase5_server_map3
train_specialist server_map4 phase5_server_map4

python build_specialist_submission.py \
  --output-dir "${COMPOSITE_DIR}" \
  --specialist "server_map1=agents/agent_phase5_server_map1" \
  --specialist "server_map2=agents/agent_phase5_server_map2" \
  --specialist "server_map3=agents/agent_phase5_server_map3" \
  --specialist "server_map4=agents/agent_phase5_server_map4"

python benchmark.py \
  --agent-dir "${COMPOSITE_DIR}" \
  --map-split server \
  --opponent-preset learned \
  --num-episodes "${BENCH_EPISODES}" \
  --output-dir "${RESULT_ROOT}/composite/learned"

python benchmark.py \
  --agent-dir "${COMPOSITE_DIR}" \
  --map-split server \
  --opponent-preset full \
  --num-episodes "${BENCH_EPISODES}" \
  --output-dir "${RESULT_ROOT}/composite/full"
