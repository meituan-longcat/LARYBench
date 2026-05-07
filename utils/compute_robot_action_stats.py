#!/usr/bin/env python3
"""Compute per-robot action normalization stats for robot regression datasets."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASETS = {
    "agibotbeta": {
        "action_dim": 16,
        "metadata_csv": "data/agibotbeta_metadata_seen_train.csv",
        "subdir": "agibot_45",
        "stats_name": "agibotbeta_stats.json",
    },
    "robocoin": {
        "action_dim": 12,
        "metadata_csv": "data/robocoin_metadata_seen_train.csv",
        "subdir": "robocoin_10",
        "stats_name": "robocoin_stats.json",
    },
}


class RunningStats:
    def __init__(self, dim):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def to_dict(self):
        if self.count == 0:
            raise ValueError("Cannot serialize empty statistics.")
        variance = self.m2 / self.count
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < 1e-6, 1.0, std)
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }


def resolve_action_path(action_root, value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    if "actions" in path.parts:
        return action_root / path
    return action_root / "actions" / path


def load_action(path, action_dim):
    action = np.load(path)
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    if flat.size % action_dim != 0:
        raise ValueError(f"{path}: action size {flat.size} is not divisible by {action_dim}")
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-robot action normalization stats from seen-train metadata."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="agibotbeta",
        help="Dataset to compute stats for.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Seen-train metadata CSV. Defaults to the canonical CSV for --dataset.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_DIR"),
        help="LARYBench data root containing regression/. Defaults to DATA_DIR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON. Defaults to <data-root>/regression/<dataset-subdir>/<stats-name>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick checks.",
    )
    args = parser.parse_args()

    if not args.data_root:
        raise ValueError("Please pass --data-root or set DATA_DIR.")

    dataset_cfg = DATASETS[args.dataset]
    action_dim = dataset_cfg["action_dim"]
    csv_path = Path(args.csv or dataset_cfg["metadata_csv"])
    action_root = Path(args.data_root) / "regression" / dataset_cfg["subdir"]
    output_path = Path(args.output) if args.output else action_root / dataset_cfg["stats_name"]

    df = pd.read_csv(csv_path)
    if args.limit is not None:
        df = df.head(args.limit)
    for col in ("robot_type", "action"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} must contain column {col!r}")

    global_stats = RunningStats(action_dim)
    robot_stats = {}
    missing = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Computing {args.dataset} stats"):
        robot_type = str(getattr(row, "robot_type"))
        action_value = getattr(row, "action")
        action_path = resolve_action_path(action_root, action_value)
        try:
            action = load_action(action_path, action_dim)
        except FileNotFoundError:
            missing.append(str(action_path))
            continue

        seq = action.reshape(-1, action_dim)
        if robot_type not in robot_stats:
            robot_stats[robot_type] = RunningStats(action_dim)
        for step in seq:
            global_stats.update(step)
            robot_stats[robot_type].update(step)

    payload = {
        "dataset": args.dataset,
        "action_mode": "absolute",
        "action_dim": action_dim,
        "source_csv": str(csv_path),
        "action_root": str(action_root),
        "missing_actions": len(missing),
        "missing_action_examples": missing[:20],
        "global": global_stats.to_dict(),
        "robot_stats": {robot: stats.to_dict() for robot, stats in sorted(robot_stats.items())},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved stats to {output_path}")
    print(f"Rows scanned: {len(df)}")
    print(f"Missing action files: {len(missing)}")
    for robot, stats in payload["robot_stats"].items():
        print(f"{robot}: count={stats['count']}")


if __name__ == "__main__":
    main()
