import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ACTION_DIMS = {
    "calvin": 7,
    "vlabench": 7,
    "vlabench_15": 7,
    "vlabench_30": 7,
    "agibotbeta": 16,
    "robocoin": 12,
}

DATASET_SUBDIR = {
    "calvin": "calvin",
    "vlabench": "vlabench",
    "vlabench_15": "vlabench",
    "vlabench_30": "vlabench",
    "agibotbeta": "agibot_45",
    "robocoin": "robocoin_10",
}

SPLIT_DIR = {
    ("calvin", "train"): "train_stride5",
    ("calvin", "val"): "val_stride5",
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


def split_from_csv_name(path):
    name = Path(path).name.lower()
    if "seen_train" in name:
        return "seen_train"
    if "seen_val" in name:
        return "seen_val"
    if "unseen" in name:
        return "unseen"
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    return "train"


def is_train_split(split):
    return split in {"train", "seen_train"}


def data_root(base_root, dataset, split):
    dataset_key = dataset.lower()
    root = Path(base_root) / "regression" / DATASET_SUBDIR.get(dataset_key, dataset_key)
    split_dir = SPLIT_DIR.get((dataset_key, split.lower()))
    if split_dir:
        root = root / split_dir
    return root


def relative_data_root(base_root, dataset, split):
    dataset_key = dataset.lower()
    root = Path(base_root) / "regression_relative" / DATASET_SUBDIR.get(dataset_key, dataset_key)
    split_dir = SPLIT_DIR.get((dataset_key, split.lower()))
    if split_dir:
        root = root / split_dir
    return root


def resolve_action_path(root, value):
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = Path(root)
    if "actions" in path.parts:
        return root / path
    return root / "actions" / path


def relative_action(action, action_dim):
    flat = np.asarray(action).reshape(-1)
    if flat.size % action_dim != 0:
        raise ValueError(f"Action size {flat.size} is not divisible by action_dim={action_dim}.")
    seq = flat.reshape(-1, action_dim)
    return (seq[-1] - seq[0]).astype(np.float32)


def output_path(src_path, src_root, dst_root):
    try:
        rel = src_path.relative_to(src_root)
    except ValueError:
        rel = Path("actions") / src_path.name
    return dst_root / rel


def process_action(value, src_root, dst_root, action_dim, overwrite=False):
    src = resolve_action_path(src_root, value)
    dst = output_path(src, src_root, dst_root)
    if dst.exists() and not overwrite:
        return np.load(dst), False

    action = np.load(src)
    rel = relative_action(action, action_dim)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, rel)
    return rel, True


def process_csv(csv_path, dataset, input_root, output_root, overwrite=False, workers=1):
    split = split_from_csv_name(csv_path)
    src_root = data_root(input_root, dataset, split)
    dst_root = relative_data_root(output_root, dataset, split)
    action_dim = ACTION_DIMS[dataset.lower()]
    df = pd.read_csv(csv_path)
    if "action" not in df.columns:
        raise ValueError(f"{csv_path} must contain an action column.")

    stats = RunningStats(action_dim) if is_train_split(split) else None
    robot_stats = {} if (stats is not None and "robot_type" in df.columns) else None
    written = 0
    skipped = 0
    values = df["action"].tolist()
    robot_types = df["robot_type"].astype(str).tolist() if robot_stats is not None else None

    def handle_result(index, result):
        nonlocal written, skipped
        rel, did_write = result
        if did_write:
            written += 1
        else:
            skipped += 1
        if stats is None:
            return
        rel64 = np.asarray(rel, dtype=np.float64).reshape(action_dim)
        stats.update(rel64)
        if robot_stats is not None:
            robot_type = robot_types[index]
            if robot_type not in robot_stats:
                robot_stats[robot_type] = RunningStats(action_dim)
            robot_stats[robot_type].update(rel64)

    if workers <= 1:
        iterator = (
            (idx, process_action(value, src_root, dst_root, action_dim, overwrite))
            for idx, value in enumerate(values)
        )
        for idx, result in tqdm(iterator, total=len(values), desc=f"{Path(csv_path).name}"):
            handle_result(idx, result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = (
                executor.map(
                    lambda item: (item[0], process_action(item[1], src_root, dst_root, action_dim, overwrite)),
                    enumerate(values),
                )
            )
            for idx, result in tqdm(futures, total=len(values), desc=f"{Path(csv_path).name}"):
                handle_result(idx, result)

    return {
        "csv": str(csv_path),
        "split": split,
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "written": written,
        "skipped": skipped,
        "stats": stats.to_dict() if stats is not None else None,
        "robot_stats": {robot: item.to_dict() for robot, item in robot_stats.items()} if robot_stats is not None else None,
    }


def merge_robot_stats(results, dataset):
    merged = {}
    for result in results:
        if result.get("robot_stats"):
            merged.update(result["robot_stats"])
    return merged


def write_stats(results, dataset, output_root):
    dataset_key = dataset.lower()
    stats_dir = Path(output_root) / "regression_relative" / DATASET_SUBDIR.get(dataset_key, dataset_key)
    stats_dir.mkdir(parents=True, exist_ok=True)
    train_stats = [item["stats"] for item in results if item["stats"]]
    if not train_stats:
        return None

    payload = {
        "action_mode": "relative",
        "definition": "last action step minus first action step",
        "dataset": dataset,
    }
    if dataset_key in {"agibotbeta", "robocoin"}:
        payload["robot_stats"] = merge_robot_stats(results, dataset)
    else:
        payload.update(train_stats[0])

    stats_path = stats_dir / f"relative_action_stats_{dataset_key}.json"
    with stats_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return stats_path


def main():
    parser = argparse.ArgumentParser(description="Generate non-overwriting relative-action files for regression.")
    parser.add_argument("--dataset", required=True, choices=sorted(ACTION_DIMS))
    parser.add_argument("--input-root", default=os.environ.get("DATA_DIR"),
                        help="LARYBench data root containing regression/. Defaults to DATA_DIR.")
    parser.add_argument("--output-root", default=None,
                        help="LARYBench data root to write into. Defaults to --input-root and creates regression_relative/<dataset>/.")
    parser.add_argument("--csv", action="append", required=True,
                        help="Metadata or latent-action CSV. Pass multiple --csv values for train/val/unseen.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker threads for npy read/write conversion.")
    args = parser.parse_args()

    if not args.input_root:
        raise ValueError("--input-root is required when DATA_DIR is not set.")
    if args.output_root is None:
        args.output_root = args.input_root

    results = []
    for csv_path in args.csv:
        results.append(process_csv(
            csv_path,
            args.dataset,
            args.input_root,
            args.output_root,
            args.overwrite,
            workers=args.workers,
        ))

    stats_path = write_stats(results, args.dataset, args.output_root)
    print(json.dumps({"results": results, "stats_path": str(stats_path) if stats_path else None}, indent=2))


if __name__ == "__main__":
    main()
