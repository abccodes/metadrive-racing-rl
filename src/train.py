"""
Full training script for the multi-agent racing environment.

Trains a PPO agent with:
- Curriculum learning (start with fewer opponents, scale up)
- Periodic evaluation with detailed metrics
- Self-play support (optional)
- TensorBoard logging

Usage:
    python train.py
    python train.py --total-timesteps 2000000 --num-agents 6
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from .env import RacingEnv, make_racing_env
from .map_splits import ALL_MAPS, MAP_SPLITS
from .opponents import REFERENCE_AGENT_DIRS, SCRIPTED_OPPONENTS, SELFPLAY_OPPONENT

UID = "000000000"  # Replace with your unique UID for submission
NAME = "Your Agent Name"  # Replace with your agent's name

assert UID != "000000000", "Please update the UID"
if NAME == "Your Agent Name":
    print("Consider updating the agent name from the default placeholder.")


class RacingMetricsCallback(BaseCallback):
    """Logs additional racing-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._route_completions = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
            if "route_completion" in info:
                self._route_completions.append(info["route_completion"])

        if len(self._episode_rewards) >= 10:
            self.logger.record("racing/mean_reward", np.mean(self._episode_rewards))
            self.logger.record("racing/mean_length", np.mean(self._episode_lengths))
            if self._route_completions:
                self.logger.record(
                    "racing/mean_route_completion", np.mean(self._route_completions)
                )
            self._episode_rewards.clear()
            self._episode_lengths.clear()
            self._route_completions.clear()

        return True


class SelfPlaySnapshotCallback(BaseCallback):
    """Periodically export a frozen snapshot and push it to self-play workers."""

    def __init__(
        self,
        train_envs,
        selfplay_indices,
        snapshot_freq,
        snapshot_dir,
        max_snapshots=3,
        frame_stack=1,
        verbose=0,
    ):
        super().__init__(verbose)
        self.train_envs = train_envs
        self.selfplay_indices = selfplay_indices
        self.snapshot_freq = snapshot_freq
        self.snapshot_dir = snapshot_dir
        self.max_snapshots = max_snapshots
        self.frame_stack = frame_stack
        self._next_snapshot = snapshot_freq
        self._snapshots = []

    def _on_step(self) -> bool:
        if not self.selfplay_indices or self.num_timesteps < self._next_snapshot:
            return True

        snapshot_name = f"snapshot_{self.num_timesteps:09d}"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        convert_to_submission(self.model, snapshot_path, frame_stack=self.frame_stack)
        self._snapshots.append(snapshot_path)

        if len(self._snapshots) > self.max_snapshots:
            stale_path = self._snapshots.pop(0)
            shutil.rmtree(stale_path)

        self.train_envs.env_method(
            "set_opponent_policy",
            f"dir:{snapshot_path}",
            indices=self.selfplay_indices,
        )

        if self.verbose:
            print(
                f"Updated {len(self.selfplay_indices)} self-play workers "
                f"with {snapshot_name}"
            )

        self._next_snapshot += self.snapshot_freq
        return True


def parse_args():
    available_opponents = (
        sorted(SCRIPTED_OPPONENTS) + sorted(REFERENCE_AGENT_DIRS) + [SELFPLAY_OPPONENT]
    )
    parser = argparse.ArgumentParser(description="Train a racing agent (full example)")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-train-envs", type=int, default=8)
    parser.add_argument("--num-eval-envs", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="aggressive",
        choices=available_opponents,
    )
    parser.add_argument(
        "--opponent-pool",
        nargs="+",
        choices=available_opponents,
        help="Optional opponent pool. Workers cycle through this pool by rank.",
    )
    parser.add_argument(
        "--opponent-agent-dirs",
        nargs="+",
        default=[],
        help="Optional additional frozen submission agents to add to the opponent pool.",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--init-model",
        type=str,
        default="",
        help="Optional SB3 PPO checkpoint (.zip) to load and continue training from.",
    )
    parser.add_argument(
        "--export-agent-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the exported standalone agent.",
    )
    parser.add_argument("--selfplay-snapshot-freq", type=int, default=100_000)
    parser.add_argument("--selfplay-max-snapshots", type=int, default=3)
    parser.add_argument(
        "--train-map-split",
        type=str,
        default="train",
        choices=sorted(MAP_SPLITS.keys()),
    )
    parser.add_argument(
        "--eval-map-split",
        type=str,
        default="validation",
        choices=sorted(MAP_SPLITS.keys()),
    )
    parser.add_argument(
        "--train-maps",
        nargs="+",
        choices=ALL_MAPS,
        help="Optional explicit training maps. Overrides --train-map-split.",
    )
    parser.add_argument(
        "--eval-maps",
        nargs="+",
        choices=ALL_MAPS,
        help="Optional explicit evaluation maps. Overrides --eval-map-split.",
    )
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--pi-layers", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--vf-layers", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--progress-reward-weight", type=float, default=0.0)
    parser.add_argument("--speed-reward-weight", type=float, default=0.0)
    parser.add_argument("--early-reward-horizon", type=int, default=0)
    parser.add_argument("--early-progress-weight", type=float, default=0.0)
    parser.add_argument("--early-speed-weight", type=float, default=0.0)
    parser.add_argument("--speed-target-kmh", type=float, default=80.0)
    parser.add_argument("--use-track-guidance", action="store_true")
    parser.add_argument(
        "--guidance-lookahead-steps", type=int, nargs="+", default=[10, 25, 45]
    )
    parser.add_argument("--line-progress-reward-weight", type=float, default=0.0)
    parser.add_argument("--line-speed-reward-weight", type=float, default=0.0)
    parser.add_argument("--line-center-penalty-weight", type=float, default=0.0)
    parser.add_argument("--line-speed-lateral-threshold", type=float, default=3.0)
    parser.add_argument("--line-speed-heading-threshold", type=float, default=0.25)
    return parser.parse_args()


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def configure_loaded_model(model, args, train_envs, log_dir, learning_rate):
    """Apply fine-tune hyperparameters to a loaded PPO model."""
    model.set_env(train_envs)
    model.tensorboard_log = log_dir
    model.n_steps = args.n_steps
    model.batch_size = args.batch_size
    model.n_epochs = args.n_epochs
    model.gamma = args.gamma
    model.gae_lambda = args.gae_lambda
    model.vf_coef = args.vf_coef
    model.ent_coef = args.ent_coef
    model.max_grad_norm = args.max_grad_norm
    model.target_kl = args.target_kl
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)
    model.clip_range = get_schedule_fn(args.clip_range)
    model._setup_lr_schedule()
    model.policy.optimizer.param_groups[0]["lr"] = model.lr_schedule(1.0)


def main():
    args = parse_args()
    train_maps = (
        args.train_maps if args.train_maps else MAP_SPLITS[args.train_map_split]
    )
    eval_maps = args.eval_maps if args.eval_maps else MAP_SPLITS[args.eval_map_split]
    save_dir = (
        os.path.join(args.save_dir, args.run_name) if args.run_name else args.save_dir
    )
    log_dir = (
        os.path.join(args.log_dir, args.run_name) if args.run_name else args.log_dir
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Racing - Full Training Example")
    print("=" * 60)
    if args.run_name:
        print(f"  Run name: {args.run_name}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Train envs: {args.num_train_envs}")
    print(f"  Agents per race: {args.num_agents}")
    print(f"  Opponent: {args.opponent_policy}")
    if args.opponent_pool:
        print(f"  Opponent pool: {', '.join(args.opponent_pool)}")
    if args.opponent_agent_dirs:
        print(f"  Frozen opponent dirs: {', '.join(args.opponent_agent_dirs)}")
    if args.opponent_pool and SELFPLAY_OPPONENT in args.opponent_pool:
        print(f"  Self-play snapshot freq: {args.selfplay_snapshot_freq}")
    if args.init_model:
        print(f"  Init model: {args.init_model}")
    print(f"  Train maps: {', '.join(train_maps)}")
    print(f"  Eval maps: {', '.join(eval_maps)}")
    print(f"  LR: {args.lr} ({args.lr_schedule}), Batch: {args.batch_size}")
    print(f"  Policy layers: {args.pi_layers}")
    print(f"  Value layers: {args.vf_layers}")
    print(f"  Frame stack: {args.frame_stack}")
    if args.progress_reward_weight or args.speed_reward_weight:
        print(
            "  Pace shaping: "
            f"progress_w={args.progress_reward_weight}, "
            f"speed_w={args.speed_reward_weight}, "
            f"speed_target={args.speed_target_kmh}"
        )
    if args.early_reward_horizon > 0:
        print(
            "  Early shaping: "
            f"horizon={args.early_reward_horizon}, "
            f"progress_w={args.early_progress_weight}, "
            f"speed_w={args.early_speed_weight}, "
            f"speed_target={args.speed_target_kmh}"
        )
    if args.use_track_guidance:
        print(
            "  Track guidance: "
            f"lookahead={args.guidance_lookahead_steps}, "
            f"line_progress_w={args.line_progress_reward_weight}, "
            f"line_speed_w={args.line_speed_reward_weight}, "
            f"line_center_penalty_w={args.line_center_penalty_weight}"
        )
    if args.export_agent_dir:
        print(f"  Export agent dir: {args.export_agent_dir}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    named_opponent_pool = (
        args.opponent_pool if args.opponent_pool else [args.opponent_policy]
    )
    dir_opponent_pool = [f"dir:{agent_dir}" for agent_dir in args.opponent_agent_dirs]
    opponent_pool = named_opponent_pool + dir_opponent_pool
    train_worker_opponents = [
        "aggressive" if spec == SELFPLAY_OPPONENT else spec for spec in opponent_pool
    ]
    eval_opponent_pool = [spec for spec in opponent_pool if spec != SELFPLAY_OPPONENT]
    if not eval_opponent_pool:
        eval_opponent_pool = ["aggressive"]
    selfplay_indices = [
        i
        for i in range(args.num_train_envs)
        if opponent_pool[i % len(opponent_pool)] == SELFPLAY_OPPONENT
    ]

    # Create environments
    train_envs = SubprocVecEnv(
        [
            make_racing_env(
                rank=i,
                num_agents=args.num_agents,
                opponent_policy=train_worker_opponents[i % len(train_worker_opponents)],
                map_names=train_maps,
                progress_reward_weight=args.progress_reward_weight,
                speed_reward_weight=args.speed_reward_weight,
                early_reward_horizon=args.early_reward_horizon,
                early_progress_weight=args.early_progress_weight,
                early_speed_weight=args.early_speed_weight,
                speed_target_kmh=args.speed_target_kmh,
                use_track_guidance=args.use_track_guidance,
                guidance_lookahead_steps=args.guidance_lookahead_steps,
                line_progress_reward_weight=args.line_progress_reward_weight,
                line_speed_reward_weight=args.line_speed_reward_weight,
                line_center_penalty_weight=args.line_center_penalty_weight,
                line_speed_lateral_threshold=args.line_speed_lateral_threshold,
                line_speed_heading_threshold=args.line_speed_heading_threshold,
            )
            for i in range(args.num_train_envs)
        ]
    )

    eval_envs = SubprocVecEnv(
        [
            make_racing_env(
                rank=100 + i,
                num_agents=args.num_agents,
                opponent_policy=eval_opponent_pool[i % len(eval_opponent_pool)],
                map_names=eval_maps,
                progress_reward_weight=args.progress_reward_weight,
                speed_reward_weight=args.speed_reward_weight,
                early_reward_horizon=args.early_reward_horizon,
                early_progress_weight=args.early_progress_weight,
                early_speed_weight=args.early_speed_weight,
                speed_target_kmh=args.speed_target_kmh,
                use_track_guidance=args.use_track_guidance,
                guidance_lookahead_steps=args.guidance_lookahead_steps,
                line_progress_reward_weight=args.line_progress_reward_weight,
                line_speed_reward_weight=args.line_speed_reward_weight,
                line_center_penalty_weight=args.line_center_penalty_weight,
                line_speed_lateral_threshold=args.line_speed_lateral_threshold,
                line_speed_heading_threshold=args.line_speed_heading_threshold,
            )
            for i in range(args.num_eval_envs)
        ]
    )

    if args.frame_stack > 1:
        train_envs = VecFrameStack(train_envs, n_stack=args.frame_stack)
        eval_envs = VecFrameStack(eval_envs, n_stack=args.frame_stack)

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.save_freq // args.num_train_envs, 1),
            save_path=save_dir,
            name_prefix="racing_ppo",
        ),
        EvalCallback(
            eval_envs,
            best_model_save_path=os.path.join(save_dir, "best"),
            log_path=log_dir,
            eval_freq=max(args.eval_freq // args.num_train_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        RacingMetricsCallback(),
    ]
    if selfplay_indices:
        callbacks.append(
            SelfPlaySnapshotCallback(
                train_envs=train_envs,
                selfplay_indices=selfplay_indices,
                snapshot_freq=args.selfplay_snapshot_freq,
                snapshot_dir=os.path.join(save_dir, "selfplay_snapshots"),
                max_snapshots=args.selfplay_max_snapshots,
                frame_stack=args.frame_stack,
                verbose=1,
            )
        )

    learning_rate = (
        args.lr if args.lr_schedule == "constant" else linear_schedule(args.lr)
    )

    # Create or resume PPO agent.
    if args.init_model:
        model = PPO.load(
            args.init_model,
            env=train_envs,
            device="auto",
            tensorboard_log=log_dir,
        )
        configure_loaded_model(model, args, train_envs, log_dir, learning_rate)
    else:
        model = PPO(
            "MlpPolicy",
            train_envs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=log_dir,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=dict(
                net_arch=dict(pi=args.pi_layers, vf=args.vf_layers),
            ),
        )

    print(f"\nPolicy architecture: {model.policy}")
    print(f"Observation space: {train_envs.observation_space}")
    print(f"Action space: {train_envs.action_space}")
    print()

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(args.init_model),
    )
    elapsed = time.time() - t0

    # Save final model
    final_path = os.path.join(save_dir, "racing_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final model saved to {final_path}")

    # Auto-convert to submission format
    print("\nConverting to submission format...")
    export_agent_dir = args.export_agent_dir or os.path.join("agents", f"agent_{UID}")
    convert_to_submission(model, export_agent_dir, frame_stack=args.frame_stack)
    print(f"Done! Example agent saved to {export_agent_dir}/")

    train_envs.close()
    eval_envs.close()


def convert_to_submission(model, output_dir, frame_stack=1):
    """Extract policy from SB3 model and save as standalone agent."""
    os.makedirs(output_dir, exist_ok=True)
    policy = model.policy

    obs_dim = policy.observation_space.shape[0]
    action_dim = policy.action_space.shape[0]
    base_obs_dim = obs_dim // max(int(frame_stack), 1)

    # Extract MLP extractor layers
    pi_layers = policy.mlp_extractor.policy_net
    hidden_sizes = []
    state_dict = {}

    for i, layer in enumerate(pi_layers):
        if isinstance(layer, torch.nn.Linear):
            hidden_sizes.append(layer.out_features)
            state_dict[f"features.{i}.weight"] = layer.weight.data.clone()
            state_dict[f"features.{i}.bias"] = layer.bias.data.clone()

    state_dict["action_mean.weight"] = policy.action_net.weight.data.clone()
    state_dict["action_mean.bias"] = policy.action_net.bias.data.clone()

    checkpoint = {
        "obs_dim": obs_dim,
        "base_obs_dim": base_obs_dim,
        "frame_stack": frame_stack,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "state_dict": state_dict,
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pt"))

    agent_code = '''"""Example trained racing agent."""

import os
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        return self.action_mean(x)


class Policy:
    CREATOR_NAME = "__CREATOR_NAME__"
    CREATOR_UID = "__CREATOR_UID__"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.base_obs_dim = checkpoint.get("base_obs_dim", self.obs_dim)
        self.frame_stack = checkpoint.get("frame_stack", 1)
        self.action_dim = checkpoint["action_dim"]
        hidden_sizes = checkpoint["hidden_sizes"]

        self.model = PolicyNetwork(self.obs_dim, self.action_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.reset()

    def reset(self):
        self.obs_history = [np.zeros(self.base_obs_dim, dtype=np.float32) for _ in range(self.frame_stack)]

    @torch.no_grad()
    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.obs_history.pop(0)
        self.obs_history.append(obs)
        stacked_obs = np.concatenate(self.obs_history, axis=0)
        obs_tensor = torch.FloatTensor(stacked_obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)
'''
    agent_code = agent_code.replace("__CREATOR_NAME__", NAME).replace(
        "__CREATOR_UID__", UID
    )
    with open(os.path.join(output_dir, "agent.py"), "w") as f:
        f.write(agent_code)


if __name__ == "__main__":
    main()
