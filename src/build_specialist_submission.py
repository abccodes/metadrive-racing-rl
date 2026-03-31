"""Build a composite submission agent from per-map specialist agents.

The generated agent uses a short fixed warmup policy to collect an opening
observation signature, classifies the server map via nearest prototype, then
routes control to the matching specialist network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import numpy as np

from .env import RacingEnv


DEFAULT_MAP_ORDER = ["server_map1", "server_map2", "server_map3", "server_map4"]


AGENT_TEMPLATE = '''"""Composite specialist racing agent."""

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


class SpecialistPolicy:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = PolicyNetwork(
            checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_sizes"]
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)


class Policy:
    CREATOR_NAME = "__CREATOR_NAME__"
    CREATOR_UID = "__CREATOR_UID__"

    def __init__(self):
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.classify_steps = int(manifest["classify_steps"])
        self.warmup_action = np.asarray(manifest["warmup_action"], dtype=np.float32)
        self.prototype_keys = manifest["prototype_keys"]
        self.map_order = manifest["map_order"]
        self.prototypes = {
            name: np.asarray(proto, dtype=np.float32)
            for name, proto in manifest["prototypes"].items()
        }
        self.specialists = {
            name: SpecialistPolicy(os.path.join(base_dir, rel_path))
            for name, rel_path in manifest["specialist_models"].items()
        }
        self.reset()

    def reset(self):
        self.step = 0
        self.map_name = None
        self.obs_history = []

    def _signature(self):
        frames = self.obs_history[: self.classify_steps]
        if len(frames) < self.classify_steps:
            pad = [frames[-1]] * (self.classify_steps - len(frames)) if frames else []
            frames = frames + pad
        return np.concatenate([frame[self.prototype_keys] for frame in frames], axis=0)

    def _classify_map(self):
        signature = self._signature()
        best_name = None
        best_dist = None
        for name in self.map_order:
            dist = float(np.mean((signature - self.prototypes[name]) ** 2))
            if best_dist is None or dist < best_dist:
                best_name = name
                best_dist = dist
        self.map_name = best_name

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.step += 1
        self.obs_history.append(obs.copy())

        if self.map_name is None and self.step >= self.classify_steps:
            self._classify_map()

        if self.map_name is None:
            return self.warmup_action.copy()

        return self.specialists[self.map_name](obs)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a composite specialist agent.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--specialist",
        action="append",
        required=True,
        help="Map-to-agent mapping in the form server_map1=agents/agent_dir",
    )
    parser.add_argument("--classify-steps", type=int, default=12)
    parser.add_argument(
        "--warmup-action",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("STEER", "THROTTLE"),
    )
    parser.add_argument("--prototype-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--prototype-keys",
        type=int,
        nargs="+",
        default=list(range(80)) + list(range(120, 161)),
        help="Observation indices used for map prototypes.",
    )
    parser.add_argument("--creator-name", default="Your Agent Name")
    parser.add_argument("--creator-uid", default="000000000")
    return parser.parse_args()


def parse_specialists(entries: list[str]) -> dict[str, str]:
    mapping = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid specialist mapping: {entry}")
        map_name, agent_dir = entry.split("=", 1)
        mapping[map_name] = agent_dir
    missing = [name for name in DEFAULT_MAP_ORDER if name not in mapping]
    if missing:
        raise ValueError(f"Missing specialist mappings for: {', '.join(missing)}")
    return mapping


def collect_map_prototype(
    map_name: str,
    classify_steps: int,
    warmup_action: np.ndarray,
    prototype_keys: list[int],
    seeds: list[int],
) -> list[float]:
    signatures = []
    env = RacingEnv(num_agents=2, opponent_policy="still", map_name=map_name)
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            frames = []
            for _ in range(classify_steps):
                frames.append(np.asarray(obs, dtype=np.float32).copy())
                obs, _, terminated, truncated, _ = env.step(warmup_action)
                if terminated or truncated:
                    break
            while len(frames) < classify_steps:
                frames.append(frames[-1].copy())
            signature = np.concatenate([frame[prototype_keys] for frame in frames], axis=0)
            signatures.append(signature)
    finally:
        env.close()
    return np.mean(np.stack(signatures, axis=0), axis=0).astype(np.float32).tolist()


def main() -> None:
    args = parse_args()
    specialist_map = parse_specialists(args.specialist)
    os.makedirs(args.output_dir, exist_ok=True)

    specialist_models = {}
    for map_name in DEFAULT_MAP_ORDER:
        src_dir = specialist_map[map_name]
        src_model = os.path.join(src_dir, "model.pt")
        if not os.path.exists(src_model):
            raise FileNotFoundError(f"Missing model.pt for {map_name}: {src_model}")
        rel_model = os.path.join("specialists", f"{map_name}.pt")
        dst_model = os.path.join(args.output_dir, rel_model)
        os.makedirs(os.path.dirname(dst_model), exist_ok=True)
        shutil.copy2(src_model, dst_model)
        specialist_models[map_name] = rel_model

    warmup_action = np.asarray(args.warmup_action, dtype=np.float32)
    prototypes = {
        map_name: collect_map_prototype(
            map_name,
            args.classify_steps,
            warmup_action,
            args.prototype_keys,
            args.prototype_seeds,
        )
        for map_name in DEFAULT_MAP_ORDER
    }

    manifest = {
        "classify_steps": args.classify_steps,
        "warmup_action": warmup_action.tolist(),
        "prototype_keys": args.prototype_keys,
        "map_order": DEFAULT_MAP_ORDER,
        "prototypes": prototypes,
        "specialist_models": specialist_models,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    agent_code = (
        AGENT_TEMPLATE.replace("__CREATOR_NAME__", args.creator_name)
        .replace("__CREATOR_UID__", args.creator_uid)
    )
    with open(os.path.join(args.output_dir, "agent.py"), "w", encoding="utf-8") as f:
        f.write(agent_code)

    print(f"Composite specialist agent written to {args.output_dir}")


if __name__ == "__main__":
    main()
