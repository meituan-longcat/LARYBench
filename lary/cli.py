"""
Unified CLI entry point for LARY.

Usage:
    lary extract-latent --model villa-x --dataset calvin
    lary classify --model lapa --dataset human_1st
    lary config show
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="lary",
        description="LARY - Latent Action Representation Learning Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract latent actions from video
    lary extract --model villa-x --dataset calvin --input data.csv

    # Run classification evaluation
    lary classify --model lapa --dataset human_1st --dim 128

    # Show current configuration
    lary config show
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ===== Extract latent action command =====
    extract_parser = subparsers.add_parser("extract", help="Extract latent actions from video/image pairs")
    extract_parser.add_argument("--model", type=str, required=True,
                                help="Model name: villa-x, vjepa2, wan2-2, flux2, dinov3, lapa, etc.")
    extract_parser.add_argument("--dataset", type=str, required=True,
                                help="Dataset name: calvin, agibotbeta, robocoin, libero, etc.")
    extract_parser.add_argument("--input", type=str,
                                help="Input CSV file path (optional, uses default if not provided)")
    extract_parser.add_argument("--output", type=str,
                                help="Output directory (optional, uses default if not provided)")
    extract_parser.add_argument("--split", type=str, default="all",
                                help="Data split: train, val, all")
    extract_parser.add_argument("--batch-size", type=int, default=16,
                                help="Batch size for processing")
    extract_parser.add_argument("--num-workers", type=int, default=8,
                                help="Number of data loading workers")
    extract_parser.add_argument("--gpus", type=str, default="0",
                                help="Comma-separated GPU IDs. Multiple IDs run extraction partitions in parallel.")
    extract_parser.add_argument("--mode", type=str, choices=["video", "image"], default="video",
                                help="Processing mode: video or image pairs")
    extract_parser.add_argument("--stride", type=int, default=5,
                                help="Stride for frame sampling (image mode)")
    extract_parser.add_argument("--perspective", type=str, default="1st",
                                help="Camera perspective (for some datasets)")
    extract_parser.add_argument("--partition", type=int, default=0,
                                help="Partition index for a single extraction worker")
    extract_parser.add_argument("--num-partitions", type=int, default=1,
                                help="Total number of extraction partitions")

    # ===== Classification command =====
    classify_parser = subparsers.add_parser("classify", help="Run classification evaluation")
    classify_parser.add_argument("--model", type=str, required=True,
                                 help="Latent action model name")
    classify_parser.add_argument("--dataset", type=str, required=True,
                                 help="Dataset name")
    classify_parser.add_argument("--dim", type=int, required=True,
                                 help="Classifier dimension")
    classify_parser.add_argument("--classes", type=int, required=True,
                                 help="Number of classes")
    classify_parser.add_argument("--config", type=str,
                                 help="Path to config file (YAML)")
    classify_parser.add_argument("--batch-size", type=int, default=256,
                                 help="Batch size")
    classify_parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                                 help="Comma-separated GPU IDs")

    # ===== Regression command =====
    regress_parser = subparsers.add_parser("regress", help="Run regression evaluation")
    regress_parser.add_argument("--model", type=str, required=True,
                                help="Model name")
    regress_parser.add_argument("--dataset", type=str, required=True,
                                help="Dataset name")
    regress_parser.add_argument("--stride", type=int, required=True,
                                help="Stride for frame sampling")
    regress_parser.add_argument("--model-type", type=str, required=True,
                                help="Model type for regression")
    regress_parser.add_argument("--action-mode", type=str, default="absolute", choices=["absolute", "relative"],
                                help="Regression target: full absolute action chunk or relative last-minus-first action.")
    regress_parser.add_argument("--action-data-root", type=str, default=None,
                                help="Optional LARYBench data root. Relative mode reads regression_relative/<dataset>/ under this root.")
    regress_parser.add_argument("--batch-size", type=int, default=256)
    regress_parser.add_argument("--epochs", type=int, default=20)
    regress_parser.add_argument("--lr", type=float, default=0.0001)
    regress_parser.add_argument("--global-stats-json", type=str, default=None,
                                help="Path to per-robot stats JSON (agibotbeta / robocoin). "
                                     "Auto-detected from DATA_DIR when not specified.")
    regress_parser.add_argument("--val-unseen-csv", type=str, default=None,
                                help="Unseen-split CSV for out-of-distribution evaluation "
                                     "(agibotbeta / robocoin). Auto-detected when not specified.")

    # ===== Config command =====
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("action", choices=["show", "init", "set"],
                               help="Config action")
    config_parser.add_argument("--key", type=str, help="Config key (for set)")
    config_parser.add_argument("--value", type=str, help="Config value (for set)")
    config_parser.add_argument("--output", type=str, help="Output file for init")

    # ===== Get indices command =====
    indices_parser = subparsers.add_parser("indices", help="Sample frame indices from videos")
    indices_parser.add_argument("--input", type=str, required=True,
                                help="Input CSV file")
    indices_parser.add_argument("--output", type=str, required=True,
                                help="Output directory")
    indices_parser.add_argument("--num-frames", type=int, default=9,
                                help="Number of frames to sample")
    indices_parser.add_argument("--num-workers", type=int, default=128,
                                help="Number of parallel workers")
    indices_parser.add_argument("--visualize", action="store_true",
                                help="Generate visualizations")

    return parser


def setup_environment(model: str) -> None:
    """Setup environment for the given model."""
    import os
    os.environ["USE_MODEL"] = model


def _parse_gpu_ids(gpus: str) -> List[int]:
    """Parse comma-separated GPU IDs."""
    ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return [int(g) for g in ids]


def _extract_csv_base_name(split: str, dataset: str, model: str, mode: str, stride: int) -> str:
    if mode == "image":
        return f"{split}_la_{dataset}_{stride}_{model}"
    return f"{split}_la_{dataset}_{model}"


def _merge_extract_partition_csvs(args, num_partitions: int) -> Path:
    """Merge partition CSVs into the canonical latent-action CSV."""
    import pandas as pd

    from lary.config import get_config

    config = get_config()
    lary_root = os.environ.get("LARY_ROOT", str(config.paths.project_root))
    data_out_dir = Path(lary_root) / "data"
    base_name = _extract_csv_base_name(args.split, args.dataset, args.model, args.mode, args.stride)
    merged_csv = data_out_dir / f"{base_name}.csv"
    partition_csvs = [data_out_dir / f"{base_name}_{i}.csv" for i in range(num_partitions)]
    missing = [p for p in partition_csvs if not p.exists()]
    if missing:
        missing_text = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Cannot merge extraction partitions. Missing CSVs:\n{missing_text}")

    merged = pd.concat((pd.read_csv(p) for p in partition_csvs), ignore_index=True)
    merged.to_csv(merged_csv, index=False)
    return merged_csv


def _spawn_extract_partitions(args, gpus: List[int]) -> None:
    """Run one extraction worker per GPU and merge partition CSVs."""
    import subprocess

    num_partitions = len(gpus)
    env_base = os.environ.copy()
    processes = []

    for partition, gpu in enumerate(gpus):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable, "-m", "lary.cli", "extract",
            "--model", args.model,
            "--dataset", args.dataset,
            "--split", args.split,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--gpus", str(gpu),
            "--mode", args.mode,
            "--stride", str(args.stride),
            "--perspective", args.perspective,
            "--partition", str(partition),
            "--num-partitions", str(num_partitions),
        ]
        if args.input:
            cmd += ["--input", args.input]
        if args.output:
            cmd += ["--output", args.output]

        print(f"Starting extraction partition {partition}/{num_partitions - 1} on GPU {gpu}")
        processes.append((partition, gpu, subprocess.Popen(cmd, env=env)))

    failures = []
    for partition, gpu, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((partition, gpu, returncode))

    if failures:
        for partition, gpu, returncode in failures:
            print(f"Extraction partition {partition} on GPU {gpu} failed with exit code {returncode}")
        sys.exit(1)

    merged_csv = _merge_extract_partition_csvs(args, num_partitions)
    print(f"All extraction partitions finished. Merged CSV saved to {merged_csv}")


def run_extract(args) -> None:
    """Run latent action extraction."""
    from lary.extract import extract_latent_actions

    gpus = _parse_gpu_ids(args.gpus)
    setup_environment(args.model)

    if len(gpus) > 1 and args.num_partitions == 1 and args.partition == 0:
        _spawn_extract_partitions(args, gpus)
        return

    extract_latent_actions(
        model=args.model,
        dataset=args.dataset,
        input_file=args.input,
        output_dir=args.output,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpus=gpus,
        mode=args.mode,
        stride=args.stride,
        perspective=args.perspective,
        partition=args.partition,
        num_partitions=args.num_partitions,
    )


def run_classify(args) -> None:
    """Run classification evaluation."""
    import os
    import subprocess
    import sys

    # Set MASTER_PORT for distributed training
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "11325")

    # Get config file
    config_file = args.config
    if config_file is None:
        from lary.config import get_config
        config = get_config()
        config_file = str(config.paths.project_root / "classification/configs/eval/vitl/manipulation.yaml")

    # Build devices list
    devices = [f"cuda:{g}" for g in args.gpus.split(",")]

    cmd = [
        sys.executable, "-m", "classification.evals.main",
        "--fname", config_file,
        "--lam", args.model,
        "--dataset", args.dataset,
        "--dim", str(args.dim),
        "--classes", str(args.classes),
        "--batch_size", str(args.batch_size),
        "--devices", *devices,
    ]

    subprocess.run(cmd)


# Datasets that use seen_train / seen_val / unseen splits instead of train / val,
# and require a per-robot stats JSON for action normalisation.
_SPLIT_DATASETS = {'agibotbeta', 'robocoin'}

# Default stats JSON filename inside the dataset data root.
_STATS_JSON_NAME = {
    'agibotbeta': 'agibotbeta_stats.json',
    'robocoin':   'robocoin_stats.json',
}

_REGRESSION_DATA_SUBDIR = {
    'calvin': 'calvin',
    'vlabench': 'vlabench',
    'vlabench_15': 'vlabench',
    'vlabench_30': 'vlabench',
    'agibotbeta': 'agibot_45',
    'robocoin': 'robocoin_10',
}


def _relative_stats_path(action_data_root: str, dataset: str) -> str:
    root = os.path.normpath(action_data_root)
    if os.path.basename(root) != "regression_relative":
        root = os.path.join(root, "regression_relative")
    dataset_key = dataset.lower()
    stats_dir = os.path.join(root, _REGRESSION_DATA_SUBDIR.get(dataset_key, dataset_key))
    return os.path.join(stats_dir, f"relative_action_stats_{dataset_key}.json")


def run_regress(args) -> None:
    """Run regression evaluation."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from lary.config import get_config
    from lary.path_resolver import get_data_root
    config = get_config()

    # Set PYTHONPATH to include project root
    project_root = str(config.paths.project_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    # All la CSVs live in $LARY_ROOT/data/
    data_dir = config.paths.project_root / "data"
    dataset = args.dataset

    # ── CSV paths ──────────────────────────────────────────────────────────
    # agibotbeta / robocoin use seen_train / seen_val / unseen splits.
    # All other datasets use the standard train / val naming.
    if dataset in _SPLIT_DATASETS:
        train_csv      = data_dir / f"seen_train_la_{dataset}_{args.stride}_{args.model}.csv"
        val_csv        = data_dir / f"seen_val_la_{dataset}_{args.stride}_{args.model}.csv"
        val_unseen_csv = data_dir / f"unseen_la_{dataset}_{args.stride}_{args.model}.csv"
    else:
        train_csv      = data_dir / f"train_la_{dataset}_{args.stride}_{args.model}.csv"
        val_csv        = data_dir / f"val_la_{dataset}_{args.stride}_{args.model}.csv"
        val_unseen_csv = None

    # Allow explicit CLI override
    if getattr(args, 'val_unseen_csv', None):
        val_unseen_csv = Path(args.val_unseen_csv)

    # ── global_stats_json ─────────────────────────────────────────────────
    # Explicit CLI override takes priority; otherwise auto-detect from DATA_DIR.
    global_stats_json = getattr(args, 'global_stats_json', None)
    if not global_stats_json and args.action_mode == "relative":
        action_root = args.action_data_root or os.environ.get("DATA_DIR")
        if action_root:
            global_stats_json = _relative_stats_path(action_root, dataset)
    if not global_stats_json and dataset in _STATS_JSON_NAME:
        data_root = get_data_root(dataset, 'seen_train')
        if data_root:
            candidate = os.path.join(data_root, _STATS_JSON_NAME[dataset])
            global_stats_json = candidate  # pass even if not (yet) accessible

    run_name = f"{dataset}_{args.stride}_{args.model}_{args.model_type}_{args.action_mode}"
    save_dir = config.paths.log_dir / "regression" / "logs" / run_name

    # Determine num_gpus from CUDA_VISIBLE_DEVICES
    # Default to 8 GPUs if not set or empty
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_devices:
        cuda_devices = "0,1,2,3,4,5,6,7"
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    num_gpus = len(cuda_devices.split(","))

    # Find a free port to avoid conflicts with other processes
    import socket
    def find_free_port(start=29500, end=29600):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start
    main_process_port = find_free_port()

    cmd = [
        "accelerate", "launch",
        "--num_machines=1",
        "--mixed_precision=no",
        "--dynamo_backend=no",
        f"--num_processes={num_gpus}",
        f"--main_process_port={main_process_port}",
        "regression/main.py",
        "--model_type", args.model_type,
        "--train_csv", str(train_csv),
        "--val_csv", str(val_csv),
        "--save_dir", str(save_dir),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--dataset", dataset,
        "--stride", str(args.stride),
        "--action_mode", args.action_mode,
        "--wandb_project", "lary",
        "--wandb_name", f"{args.model}-{dataset}-{args.stride}-{args.model_type}-{args.action_mode}",
    ]

    if args.action_data_root:
        cmd += ["--action_data_root", args.action_data_root]
    if val_unseen_csv:
        cmd += ["--val_unseen_csv", str(val_unseen_csv)]
    if global_stats_json:
        cmd += ["--global_stats_json", global_stats_json]

    subprocess.run(cmd, env=env)


def run_config(args) -> None:
    """Handle config commands."""
    from lary.config import get_config, Config

    if args.action == "show":
        config = get_config()
        print("Current Configuration:")
        print(f"  Project Root: {config.paths.project_root}")
        print(f"  Log Dir: {config.paths.log_dir}")
        print(f"  Data Dir: {config.paths.data_dir}")
        print(f"  Model Dir: {config.paths.model_dir}")
        print(f"  Model: {config.model.name}")
        print(f"  Batch Size: {config.data.batch_size}")

    elif args.action == "init":
        output = args.output or "lary_config.yaml"
        config = Config()
        config.to_yaml(output)
        print(f"Configuration file created: {output}")

    elif args.action == "set":
        if not args.key or not args.value:
            print("Error: --key and --value required for set action")
            sys.exit(1)
        # TODO: Implement config set
        print(f"Setting {args.key} = {args.value}")


def run_indices(args) -> None:
    """Run frame indices sampling."""
    # Import and run the indices extraction
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "get_latent_action.get_indices",
        "--input_file", args.input,
        "--output_dir", args.output,
        "--num_frames", str(args.num_frames),
        "--num_workers", str(args.num_workers),
    ]
    if args.visualize:
        cmd.append("--visualize")

    subprocess.run(cmd)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "extract": run_extract,
        "classify": run_classify,
        "regress": run_regress,
        "config": run_config,
        "indices": run_indices,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
