#!/usr/bin/env bash
set -euo pipefail

BASE_AGENT="${BASE_AGENT:-agents/agent_p4d_progress70_sustain_lowent}"
BENCH_EPISODES="${BENCH_EPISODES:-10}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/phase6b_launch_micro}"

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

# Baseline from the previous best launch wrapper.
build_and_bench agent_phase6b_launch_long_base \
  --launch-steps 32 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.25 \
  --launch-steer-scale 0.70

# Slightly longer hold with the same smooth taper.
build_and_bench agent_phase6b_launch_longer \
  --launch-steps 40 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.25 \
  --launch-steer-scale 0.70

# Keep the long launch, but release throttle less aggressively.
build_and_bench agent_phase6b_launch_long_higher_floor \
  --launch-steps 32 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.35 \
  --launch-steer-scale 0.70

# Keep the long launch, but allow a bit more steering authority.
build_and_bench agent_phase6b_launch_long_more_steer \
  --launch-steps 32 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.25 \
  --launch-steer-scale 0.80

# Combine the two most plausible "slightly more aggressive" tweaks.
build_and_bench agent_phase6b_launch_long_combo \
  --launch-steps 40 \
  --launch-min-throttle 1.0 \
  --launch-final-min-throttle 0.35 \
  --launch-steer-scale 0.80
