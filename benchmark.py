"""Benchmark runner for local racing agent evaluation.

This script wraps eval_local.py's evaluation functions with a fixed benchmark
matrix, multiple seeds, and structured result export so candidate agents can be
compared consistently over time.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

from eval_local import evaluate_single, evaluate_versus
from map_splits import ALL_MAPS, MAP_SPLITS
from racing_maps import set_racing_map


DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_SINGLE_OPPONENT = "aggressive"
REFERENCE_AGENTS = {
    "baseline": os.path.join("agents", "baseline_agent"),
    "example": os.path.join("agents", "example_agent"),
}
OPPONENT_PRESETS = {
    "scripted": {
        "single_opponents": ["aggressive", "random", "still"],
        "reference_agents": [],
    },
    "learned": {
        "single_opponents": [],
        "reference_agents": ["baseline", "example"],
    },
    "full": {
        "single_opponents": ["aggressive", "random", "still"],
        "reference_agents": ["baseline", "example"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a repeatable local benchmark suite."
    )
    parser.add_argument(
        "--agent-dir", required=True, help="Candidate agent directory to benchmark."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to evaluate."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Episodes per benchmark configuration.",
    )
    parser.add_argument(
        "--maps", nargs="+", choices=ALL_MAPS, help="Explicit maps to evaluate."
    )
    parser.add_argument(
        "--map-split",
        choices=sorted(MAP_SPLITS.keys()),
        default="validation",
        help="Named map split to evaluate when --maps is not provided.",
    )
    parser.add_argument(
        "--single-opponent-policy",
        default=DEFAULT_SINGLE_OPPONENT,
        choices=["random", "aggressive", "still"],
        help="Built-in opponent used for single-agent benchmarks.",
    )
    parser.add_argument(
        "--reference-agents",
        nargs="+",
        default=["baseline", "example"],
        choices=sorted(REFERENCE_AGENTS.keys()),
        help="Reference agents to include in versus benchmarks.",
    )
    parser.add_argument(
        "--opponent-preset",
        choices=sorted(OPPONENT_PRESETS.keys()),
        help="Optional opponent benchmark preset. Overrides default opponent lists unless explicitly set.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for raw and summary benchmark outputs.",
    )
    return parser.parse_args()


def build_benchmark_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    selected_maps = args.maps if args.maps else MAP_SPLITS[args.map_split]
    single_opponents = [args.single_opponent_policy]
    reference_agents = list(args.reference_agents)

    if args.opponent_preset:
        preset = OPPONENT_PRESETS[args.opponent_preset]
        single_opponents = preset["single_opponents"]
        reference_agents = preset["reference_agents"]

    for map_name in selected_maps:
        for seed in args.seeds:
            for single_opponent in single_opponents:
                configs.append(
                    {
                        "name": f"single_{map_name}_{single_opponent}_seed{seed}",
                        "group": "single",
                        "mode": "single",
                        "map": map_name,
                        "seed": seed,
                        "num_episodes": args.num_episodes,
                        "agent_dirs": [args.agent_dir],
                        "opponent_policy": single_opponent,
                        "opponent_label": single_opponent,
                    }
                )

            for ref_name in reference_agents:
                configs.append(
                    {
                        "name": f"versus_{ref_name}_{map_name}_seed{seed}",
                        "group": "versus",
                        "mode": "versus",
                        "map": map_name,
                        "seed": seed,
                        "num_episodes": args.num_episodes,
                        "agent_dirs": [args.agent_dir, REFERENCE_AGENTS[ref_name]],
                        "reference_agent_key": ref_name,
                        "opponent_label": ref_name,
                    }
                )

    return configs


def normalize_single_result(
    config: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    details = result.get("details", {})
    arrive_count = int(details.get("arrive_count", 0))
    num_episodes = int(config["num_episodes"])
    win_count = int(result.get("win_count", 0))
    lose_count = int(result.get("lose_count", 0))

    return {
        "config_name": config["name"],
        "group": config["group"],
        "mode": config["mode"],
        "map": config["map"],
        "seed": config["seed"],
        "num_episodes": num_episodes,
        "agent_dir": config["agent_dirs"][0],
        "agent_label": os.path.basename(config["agent_dirs"][0]),
        "opponent_label": config["opponent_label"],
        "avg_reward": float(result["avg_reward"]),
        "avg_route_completion": float(result["avg_route_completion"]),
        "avg_speed": float(result["avg_speed"]),
        "avg_route_step_100": float(result.get("avg_route_step_100", 0.0)),
        "avg_route_step_200": float(result.get("avg_route_step_200", 0.0)),
        "avg_speed_step_100": float(result.get("avg_speed_step_100", 0.0)),
        "avg_speed_step_200": float(result.get("avg_speed_step_200", 0.0)),
        "avg_arrival_step": (
            float(result["avg_arrival_step"])
            if result.get("avg_arrival_step") is not None
            else None
        ),
        "arrive_count": arrive_count,
        "arrival_rate": arrive_count / num_episodes,
        "win_count": win_count,
        "win_rate": win_count / num_episodes,
        "lose_count": lose_count,
        "lose_rate": lose_count / num_episodes,
        "rank": int(result.get("rank", 1)),
    }


def normalize_versus_result(
    config: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_dir = config["agent_dirs"][0]
    candidate = None
    for item in results:
        if item["agent_dir"] == candidate_dir:
            candidate = item
            break
    if candidate is None:
        raise ValueError(f"Candidate agent result missing for config {config['name']}")

    details = candidate.get("details", {})
    num_episodes = int(config["num_episodes"])
    arrive_count = int(details.get("arrive_count", 0))
    win_count = int(candidate.get("win_count", 0))
    lose_count = int(candidate.get("lose_count", 0))

    return {
        "config_name": config["name"],
        "group": config["group"],
        "mode": config["mode"],
        "map": config["map"],
        "seed": config["seed"],
        "num_episodes": num_episodes,
        "agent_dir": candidate_dir,
        "agent_label": os.path.basename(candidate_dir),
        "opponent_label": config["opponent_label"],
        "avg_reward": float(candidate["avg_reward"]),
        "avg_route_completion": float(candidate["avg_route_completion"]),
        "avg_speed": float(candidate["avg_speed"]),
        "avg_route_step_100": float(candidate.get("avg_route_step_100", 0.0)),
        "avg_route_step_200": float(candidate.get("avg_route_step_200", 0.0)),
        "avg_speed_step_100": float(candidate.get("avg_speed_step_100", 0.0)),
        "avg_speed_step_200": float(candidate.get("avg_speed_step_200", 0.0)),
        "avg_arrival_step": (
            float(candidate["avg_arrival_step"])
            if candidate.get("avg_arrival_step") is not None
            else None
        ),
        "arrive_count": arrive_count,
        "arrival_rate": arrive_count / num_episodes,
        "win_count": win_count,
        "win_rate": win_count / num_episodes,
        "lose_count": lose_count,
        "lose_rate": lose_count / num_episodes,
        "rank": int(candidate["rank"]),
    }


def run_config(config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    restore = set_racing_map(config["map"])
    try:
        if config["mode"] == "single":
            raw_result = evaluate_single(
                config["agent_dirs"][0],
                num_episodes=config["num_episodes"],
                opponent_policy=config["opponent_policy"],
                seed=config["seed"],
            )
            normalized = normalize_single_result(config, raw_result)
        else:
            raw_result = evaluate_versus(
                config["agent_dirs"],
                num_episodes=config["num_episodes"],
                seed=config["seed"],
            )
            normalized = normalize_versus_result(config, raw_result)
    finally:
        restore()

    return normalized, raw_result


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (run["mode"], run["map"], run["opponent_label"])
        grouped[key].append(run)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        mode, map_name, opponent_label = key
        items = grouped[key]

        def values(field: str) -> list[float]:
            return [float(item[field]) for item in items]

        arrival_rates = values("arrival_rate")
        win_rates = values("win_rate")
        arrival_steps = [
            float(item["avg_arrival_step"])
            for item in items
            if item.get("avg_arrival_step") is not None
        ]

        summary.append(
            {
                "mode": mode,
                "map": map_name,
                "opponent_label": opponent_label,
                "num_runs": len(items),
                "num_episodes_per_run": items[0]["num_episodes"],
                "seeds": [item["seed"] for item in items],
                "mean_arrival_rate": mean(arrival_rates),
                "std_arrival_rate": pstdev(arrival_rates)
                if len(arrival_rates) > 1
                else 0.0,
                "mean_win_rate": mean(win_rates),
                "std_win_rate": pstdev(win_rates) if len(win_rates) > 1 else 0.0,
                "mean_avg_route_completion": mean(values("avg_route_completion")),
                "mean_avg_speed": mean(values("avg_speed")),
                "mean_avg_reward": mean(values("avg_reward")),
                "mean_route_step_100": mean(values("avg_route_step_100")),
                "mean_route_step_200": mean(values("avg_route_step_200")),
                "mean_speed_step_100": mean(values("avg_speed_step_100")),
                "mean_speed_step_200": mean(values("avg_speed_step_200")),
                "mean_arrival_step": mean(arrival_steps) if arrival_steps else None,
                "total_arrivals": sum(int(item["arrive_count"]) for item in items),
                "total_wins": sum(int(item["win_count"]) for item in items),
                "total_losses": sum(int(item["lose_count"]) for item in items),
                "total_episodes": sum(int(item["num_episodes"]) for item in items),
            }
        )

    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\n=== Benchmark Summary ===")
    for item in summary:
        print(
            f"{item['mode']:>6} | {item['map']:<7} | {item['opponent_label']:<10} | "
            f"arrival_rate={item['mean_arrival_rate']:.2f} | "
            f"win_rate={item['mean_win_rate']:.2f} | "
            f"route@100={item['mean_route_step_100']:.2%} | "
            f"speed@100={item['mean_speed_step_100']:.1f} | "
            f"route={item['mean_avg_route_completion']:.2%} | "
            f"speed={item['mean_avg_speed']:.1f}"
        )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.agent_dir):
        raise FileNotFoundError(f"Agent directory not found: {args.agent_dir}")

    output_dir = os.path.abspath(args.output_dir)
    ensure_output_dir(output_dir)

    configs = build_benchmark_configs(args)
    raw_runs: list[dict[str, Any]] = []
    normalized_runs: list[dict[str, Any]] = []

    print(f"Running {len(configs)} benchmark configs for {args.agent_dir}")
    for index, config in enumerate(configs, start=1):
        print(
            f"\n[{index}/{len(configs)}] {config['name']} "
            f"(episodes={config['num_episodes']}, seed={config['seed']})"
        )
        normalized, raw_result = run_config(config)
        normalized_runs.append(normalized)
        raw_runs.append(
            {"config": config, "result": raw_result, "normalized": normalized}
        )

    summary = summarize_runs(normalized_runs)
    print_summary(summary)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    agent_label = os.path.basename(os.path.abspath(args.agent_dir))
    raw_path = os.path.join(output_dir, f"{agent_label}_{timestamp}_raw.json")
    summary_path = os.path.join(output_dir, f"{agent_label}_{timestamp}_summary.json")

    write_json(
        raw_path,
        {
            "benchmark_version": 1,
            "created_at": timestamp,
            "agent_dir": args.agent_dir,
            "runs": raw_runs,
        },
    )
    write_json(
        summary_path,
        {
            "benchmark_version": 1,
            "created_at": timestamp,
            "agent_dir": args.agent_dir,
            "summary": summary,
        },
    )

    print(f"\nSaved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
