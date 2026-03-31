#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_server_benchmark.sh <agent_dir> [num_episodes]"
  exit 1
fi

AGENT_DIR="$1"
NUM_EPISODES="${2:-10}"
AGENT_LABEL="$(basename "$AGENT_DIR")"
RESULT_ROOT="benchmark_results/${AGENT_LABEL}"

python benchmark.py \
  --agent-dir "${AGENT_DIR}" \
  --map-split server \
  --opponent-preset learned \
  --num-episodes "${NUM_EPISODES}" \
  --output-dir "${RESULT_ROOT}/learned"

python benchmark.py \
  --agent-dir "${AGENT_DIR}" \
  --map-split server \
  --opponent-preset full \
  --num-episodes "${NUM_EPISODES}" \
  --output-dir "${RESULT_ROOT}/full"
