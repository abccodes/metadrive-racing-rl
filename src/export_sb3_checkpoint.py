"""Export an SB3 PPO checkpoint to submission format."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from .train import convert_to_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an SB3 PPO checkpoint to an agent dir.")
    parser.add_argument("--checkpoint", required=True, help="Path to the SB3 .zip checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output submission agent directory")
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Frame stack used during training/export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.checkpoint, device="cpu")
    convert_to_submission(model, args.output_dir, frame_stack=args.frame_stack)
    print(f"Exported {args.checkpoint} -> {args.output_dir}")


if __name__ == "__main__":
    main()
