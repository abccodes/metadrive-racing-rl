"""Build a submission agent with a short scripted launch phase.

The hybrid agent uses the exported PPO policy every step, but can:
- boost throttle and dampen steering for the first N timesteps
- smooth steering over time
- limit sudden throttle drops
- enforce throttle floors on straights / mild turns / sharp turns
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


AGENT_TEMPLATE = '''"""Hybrid launch racing agent."""

import json
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
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        model_path = os.path.join(base_dir, manifest["model_path"])
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = PolicyNetwork(
            checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_sizes"]
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.launch_steps = int(manifest["launch_steps"])
        self.launch_min_throttle = float(manifest["launch_min_throttle"])
        self.launch_final_min_throttle = float(manifest["launch_final_min_throttle"])
        self.launch_steer_scale = float(manifest["launch_steer_scale"])
        self.steer_smoothing = float(manifest["steer_smoothing"])
        self.max_throttle_drop = float(manifest["max_throttle_drop"])
        self.straight_steer_threshold = float(manifest["straight_steer_threshold"])
        self.mild_steer_threshold = float(manifest["mild_steer_threshold"])
        self.straight_min_throttle = float(manifest["straight_min_throttle"])
        self.mild_turn_min_throttle = float(manifest["mild_turn_min_throttle"])
        self.sharp_turn_min_throttle = float(manifest["sharp_turn_min_throttle"])
        self.sharp_turn_steer_scale = float(manifest["sharp_turn_steer_scale"])
        self.launch_counter = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.launch_counter = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

    def _launch_floor(self):
        if self.launch_steps <= 1:
            return self.launch_final_min_throttle
        progress = min(self.launch_counter, self.launch_steps - 1) / (self.launch_steps - 1)
        return (
            self.launch_min_throttle
            + progress * (self.launch_final_min_throttle - self.launch_min_throttle)
        )

    @torch.no_grad()
    def __call__(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()

        # Smooth steering to avoid twitchy mid-corner corrections.
        if self.steer_smoothing > 0.0:
            action[0] = (
                self.steer_smoothing * self.prev_action[0]
                + (1.0 - self.steer_smoothing) * action[0]
            )

        # Scale very sharp steering slightly down to preserve speed through turns.
        if self.sharp_turn_steer_scale < 1.0 and abs(action[0]) >= self.mild_steer_threshold:
            action[0] *= self.sharp_turn_steer_scale

        if self.launch_counter < self.launch_steps:
            throttle_floor = self._launch_floor()
            action[0] = np.clip(action[0] * self.launch_steer_scale, -1.0, 1.0)
            action[1] = max(action[1], throttle_floor)

        steer_mag = abs(action[0])
        if self.straight_min_throttle > -1.0 and steer_mag <= self.straight_steer_threshold:
            action[1] = max(action[1], self.straight_min_throttle)
        elif self.mild_turn_min_throttle > -1.0 and steer_mag <= self.mild_steer_threshold:
            action[1] = max(action[1], self.mild_turn_min_throttle)
        elif self.sharp_turn_min_throttle > -1.0:
            action[1] = max(action[1], self.sharp_turn_min_throttle)

        # Avoid abrupt throttle drops that kill opening and corner-exit speed.
        if self.max_throttle_drop < 2.0:
            action[1] = max(action[1], self.prev_action[1] - self.max_throttle_drop)

        self.launch_counter += 1
        action = np.clip(action, -1.0, 1.0)
        self.prev_action = action.copy()
        return action
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hybrid launch submission agent.")
    parser.add_argument("--base-agent-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--launch-steps", type=int, default=24)
    parser.add_argument("--launch-min-throttle", type=float, default=1.0)
    parser.add_argument("--launch-final-min-throttle", type=float, default=0.35)
    parser.add_argument("--launch-steer-scale", type=float, default=0.75)
    parser.add_argument("--steer-smoothing", type=float, default=0.0)
    parser.add_argument("--max-throttle-drop", type=float, default=2.0)
    parser.add_argument("--straight-steer-threshold", type=float, default=0.08)
    parser.add_argument("--mild-steer-threshold", type=float, default=0.35)
    parser.add_argument("--straight-min-throttle", type=float, default=-1.0)
    parser.add_argument("--mild-turn-min-throttle", type=float, default=-1.0)
    parser.add_argument("--sharp-turn-min-throttle", type=float, default=-1.0)
    parser.add_argument("--sharp-turn-steer-scale", type=float, default=1.0)
    parser.add_argument("--creator-name", default="Your Agent Name")
    parser.add_argument("--creator-uid", default="000000000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_model = os.path.join(args.base_agent_dir, "model.pt")
    if not os.path.exists(src_model):
        raise FileNotFoundError(f"Missing model.pt in {args.base_agent_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy2(src_model, os.path.join(args.output_dir, "model.pt"))

    manifest = {
        "model_path": "model.pt",
        "launch_steps": args.launch_steps,
        "launch_min_throttle": args.launch_min_throttle,
        "launch_final_min_throttle": args.launch_final_min_throttle,
        "launch_steer_scale": args.launch_steer_scale,
        "steer_smoothing": args.steer_smoothing,
        "max_throttle_drop": args.max_throttle_drop,
        "straight_steer_threshold": args.straight_steer_threshold,
        "mild_steer_threshold": args.mild_steer_threshold,
        "straight_min_throttle": args.straight_min_throttle,
        "mild_turn_min_throttle": args.mild_turn_min_throttle,
        "sharp_turn_min_throttle": args.sharp_turn_min_throttle,
        "sharp_turn_steer_scale": args.sharp_turn_steer_scale,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    agent_code = (
        AGENT_TEMPLATE.replace("__CREATOR_NAME__", args.creator_name)
        .replace("__CREATOR_UID__", args.creator_uid)
    )
    with open(os.path.join(args.output_dir, "agent.py"), "w", encoding="utf-8") as f:
        f.write(agent_code)

    print(f"Hybrid launch agent written to {args.output_dir}")


if __name__ == "__main__":
    main()
