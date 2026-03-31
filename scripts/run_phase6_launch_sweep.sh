#!/usr/bin/env bash
set -euo pipefail

BASE_AGENT="${BASE_AGENT:-agents/agent_p4d_progress70_sustain_lowent}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase6_launch}"

build_and_bench() {
  local name="$1"
  shift

  python build_hybrid_launch_submission.py \
    --base-agent-dir "${BASE_AGENT}" \
    --output-dir "agents/${name}" \
    "$@"

  python benchmark.py \
    --agent-dir "agents/${name}" \
    --map-split server \
    --opponent-preset learned \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${name}/learned"

  python benchmark.py \
    --agent-dir "agents/${name}" \
    --map-split server \
    --opponent-preset full \
    --num-episodes "${BENCH_EPISODES}" \
    --output-dir "${RESULT_ROOT}/${name}/full"
}

build_and_bench agent_phase6_launch_default \
  --launch-steps 24 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.35 \
  --launch-steer-scale 0.75

build_and_bench agent_phase6_launch_short \
  --launch-steps 16 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.45 \
  --launch-steer-scale 0.80

build_and_bench agent_phase6_launch_long \
  --launch-steps 32 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.25 \
  --launch-steer-scale 0.70

build_and_bench agent_phase6_launch_aggressive \
  --launch-steps 24 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.55 \
  --launch-steer-scale 0.70
