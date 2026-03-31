"""Opponent policy helpers for training and evaluation."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Callable

import numpy as np


def random_opponent(obs, agent_id):
    return np.random.uniform(-1, 1, size=(2,)).astype(np.float32)


def aggressive_opponent(obs, agent_id):
    return np.array([0.0, 1.0], dtype=np.float32)


def still_opponent(obs, agent_id):
    return np.array([0.0, 0.0], dtype=np.float32)


SCRIPTED_OPPONENTS = {
    "random": random_opponent,
    "aggressive": aggressive_opponent,
    "still": still_opponent,
}

REFERENCE_AGENT_DIRS = {
    "baseline": os.path.join("agents", "baseline_agent"),
    "example": os.path.join("agents", "example_agent"),
}
SELFPLAY_OPPONENT = "selfplay"


def load_policy(agent_dir: str):
    """Load a submission policy from an agent directory."""
    agent_py = os.path.join(agent_dir, "agent.py")
    if not os.path.exists(agent_py):
        raise FileNotFoundError(f"No agent.py found in {agent_dir}")

    spec = importlib.util.spec_from_file_location(
        f"agent_{os.path.basename(agent_dir)}", agent_py
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(agent_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.Policy()


class LoadedPolicyOpponent:
    """Wrap a loaded agent policy into the env opponent callable interface."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def __call__(self, obs, agent_id):
        action = self.policy(obs)
        return np.asarray(action, dtype=np.float32)


def build_opponent(opponent_spec: str | Callable):
    """Build a training opponent from a scripted name or reference agent key."""
    if callable(opponent_spec):
        return opponent_spec
    if isinstance(opponent_spec, str) and opponent_spec.startswith("dir:"):
        return LoadedPolicyOpponent(load_policy(opponent_spec[4:]))
    if opponent_spec in SCRIPTED_OPPONENTS:
        return SCRIPTED_OPPONENTS[opponent_spec]
    if opponent_spec in REFERENCE_AGENT_DIRS:
        return LoadedPolicyOpponent(load_policy(REFERENCE_AGENT_DIRS[opponent_spec]))
    raise ValueError(
        f"Unknown opponent '{opponent_spec}'. "
        f"Available: {sorted(SCRIPTED_OPPONENTS)} + {sorted(REFERENCE_AGENT_DIRS)}"
    )
