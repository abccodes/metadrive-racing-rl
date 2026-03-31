#!/usr/bin/env bash
set -euo pipefail

BASE_AGENT="${BASE_AGENT:-agents/agent_p4d_progress70_sustain_lowent}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase7_action_wrapper}"

build_and_bench() {
  local name="$1"
  shift

  python build_hybrid_launch_submission.py \
    --base-agent-dir "${BASE_AGENT}" \
    --output-dir "agents/${name}" \
    --launch-steps 32 \
    --launch-min-throttle 1.0 \
    --launch-final-min-throttle 0.25 \
    --launch-steer-scale 0.70 \
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

# Keep the phase6 launch-long behavior as the baseline inside this wrapper family.
build_and_bench agent_phase7_wrapper_base

# More smoothing and stronger throttle-drop protection.
build_and_bench agent_phase7_wrapper_smooth \
  --steer-smoothing 0.50 \
  --max-throttle-drop 0.12 \
  --straight-min-throttle 0.40 \
  --mild-turn-min-throttle 0.15 \
  --sharp-turn-min-throttle 0.00 \
  --sharp-turn-steer-scale 0.90

# Slightly looser steering with a bit more corner throttle.
build_and_bench agent_phase7_wrapper_corner \
  --steer-smoothing 0.30 \
  --max-throttle-drop 0.16 \
  --straight-min-throttle 0.38 \
  --mild-turn-min-throttle 0.22 \
  --sharp-turn-min-throttle 0.05 \
  --sharp-turn-steer-scale 0.96

# Heavier steering damping for preserving speed through technical sections.
build_and_bench agent_phase7_wrapper_damped \
  --steer-smoothing 0.55 \
  --max-throttle-drop 0.14 \
  --straight-min-throttle 0.35 \
  --mild-turn-min-throttle 0.18 \
  --sharp-turn-min-throttle -0.02 \
  --sharp-turn-steer-scale 0.86
