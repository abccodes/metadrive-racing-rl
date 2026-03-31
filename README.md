# MetaDrive Racing RL

Reinforcement learning experiments for multi-agent racing in the MetaDrive environment, focused on training a competitive continuous-control racing policy and evaluating it against scripted and learned opponents.

This project is built around the MetaDrive / MetaDrive Arena ecosystem:

- MetaDrive Arena: https://github.com/VAIL-UCLA/MetaDrive-Arena

## Project Summary

This repository contains a full experimental pipeline for training and evaluating racing agents in MetaDrive. The final approach in this repository placed 5th out of 70 students in the course competition. The core direction that worked best was:

- PPO as the base algorithm
- server-map specialization instead of broad map generalization
- pace-oriented reward shaping
- a short launch wrapper to improve the opening phase of each race
- track-guidance reward shaping during training to improve line quality and corner-speed preservation

The repository includes:

- local racing environments and map definitions
- benchmark and evaluation tooling
- opponent-pool and self-play infrastructure
- experiment scripts for major training branches
- utilities to export trained checkpoints into self-contained submission agents

## Repository Structure

Main files:

- `train.py`
  - PPO training entrypoint
  - supports map splits, reward shaping, frame stacking, self-play, and resumed training
- `env.py`
  - single-agent wrapper around MetaDrive multi-agent racing
  - contains reward shaping and optional track-guidance shaping
- `racing_maps.py`
  - local track definitions
  - includes default maps and reconstructed server-style maps
- `map_splits.py`
  - train / validation / test / server map groupings
- `opponents.py`
  - scripted and learned opponent loading helpers
- `eval_local.py`
  - direct local evaluation against scripted or learned agents
- `benchmark.py`
  - repeatable benchmark runner with JSON summaries
- `track_guidance.py`
  - reference-line progress and line-alignment helpers used for training-time shaping

Utility scripts:

- `build_hybrid_launch_submission.py`
  - wraps a trained submission agent with a short scripted launch phase
- `build_specialist_submission.py`
  - utility for packaging specialist variants
- `export_sb3_checkpoint.py`
  - exports an SB3 PPO checkpoint to a self-contained submission agent

Experiment scripts:

- `scripts/run_phase4a.sh`
- `scripts/run_phase4b_server.sh`
- `scripts/run_phase4c_pace.sh`
- `scripts/run_phase4d_micro.sh`
- `scripts/run_phase5_specialists.sh`
- `scripts/run_phase6_launch_sweep.sh`
- `scripts/run_phase6b_launch_micro.sh`
- `scripts/run_phase7_action_wrapper.sh`
- `scripts/run_phase8_league_finetune.sh`
- `scripts/run_phase8b_league_launch.sh`
- `scripts/run_phase9_checkpoint_search.sh`
- `scripts/run_phase10_guidance.sh`
- `scripts/run_server_benchmark.sh`

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cs260r_miniproject
```

If needed, install directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Training

Basic training run:

```bash
python train.py --total-timesteps 2000000
```

Example of a server-specialized PPO run:

```bash
python train.py \
  --run-name example_server_run \
  --export-agent-dir agents/agent_example_server_run \
  --total-timesteps 2000000 \
  --num-train-envs 8 \
  --num-eval-envs 2 \
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
```

## Benchmarking

Run the full server benchmark for a trained agent:

```bash
bash scripts/run_server_benchmark.sh agents/agent_example_server_run
```

Direct benchmark examples:

```bash
python benchmark.py \
  --agent-dir agents/agent_example_server_run \
  --map-split server \
  --opponent-preset learned

python benchmark.py \
  --agent-dir agents/agent_example_server_run \
  --map-split server \
  --opponent-preset full
```

The benchmark summaries include metrics such as:

- `win_rate`
- `arrival_rate`
- `route_completion`
- `speed`
- `route@100`
- `speed@100`
- `arrival_step`

## Local Evaluation

Evaluate a trained agent directly:

```bash
python eval_local.py --agent-dirs agents/agent_example_server_run
python eval_local.py --agent-dirs agents/agent_example_server_run agents/example_agent --mode versus
python eval_local.py --agent-dirs agents/agent_example_server_run agents/example_agent --mode versus --map server_map1
```

## Best Experimental Direction

The strongest direction in this repository was:

1. Train PPO on the reconstructed server-style maps
2. Use pace-oriented reward shaping with stronger early progress pressure
3. Keep entropy low enough to reduce hesitation
4. Wrap the final policy with a short scripted launch phase
5. Add track-guidance reward shaping during training to improve line quality

That direction outperformed:

- broader map-generalization branches
- frame-stacking branches
- specialist-map routing
- league fine-tuning against frozen local agents
- pure checkpoint-search approaches
- more invasive action postprocessing

## Notes on Repository Contents

This repository is structured as a clean public code release.

Excluded from version control:

- local checkpoints
- benchmark outputs
- logs
- local submission artifacts
- report drafts
- large model files

The code is intended to document the training and evaluation workflow, not to archive every local artifact generated during experimentation.
