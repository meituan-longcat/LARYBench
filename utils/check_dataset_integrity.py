import argparse
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import time

import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_GROUPS = {
    "Human_1st": [
        "human_1st_metadata_train.csv",
        "human_1st_metadata_val.csv",
    ],
    "Robot_1st": [
        "robot_1st_metadata_train.csv",
        "robot_1st_metadata_val.csv",
    ],
    "Agibot": [
        "agibotbeta_metadata_seen_train.csv",
        "agibotbeta_metadata_seen_val.csv",
        "agibotbeta_metadata_unseen.csv",
    ],
    "CALVIN": [
        "calvin_metadata_train.csv",
        "calvin_metadata_val.csv",
    ],
    "RoboCOIN": [
        "robocoin_metadata_seen_train.csv",
        "robocoin_metadata_seen_val.csv",
        "robocoin_metadata_unseen.csv",
    ],
    "VLABench": [
        # "vlabench_15_metadata_train.csv",
        # "vlabench_15_metadata_val.csv",
        # "vlabench_30_metadata_train.csv",
        # "vlabench_30_metadata_val.csv",
        "vlabench_metadata_train.csv",
        "vlabench_metadata_val.csv",
    ],
}

CLASSIFICATION_GROUPS = {"Human_1st", "Robot_1st"}

REGRESSION_ROOTS = {
    "Agibot": "regression/agibot_45",
    "CALVIN": {
        "calvin_metadata_train.csv": "regression/calvin/train_stride5",
        "calvin_metadata_val.csv": "regression/calvin/val_stride5",
    },
    "RoboCOIN": "regression/robocoin_10",
    "VLABench": "regression/vlabench",
}

IMAGE_COLUMNS = ("src_img", "tgt_img")
VIDEO_COLUMNS = ("video_path",)
NPY_COLUMNS = ("action",)


@dataclass
class FileRecord:
    group: str
    csv_name: str
    media_type: str
    column: str
    path: Path
    relative_path: str
    refs: int


def resolve_root(data_root, group, csv_name):
    if group in CLASSIFICATION_GROUPS:
        return data_root / "classification"
    root = REGRESSION_ROOTS[group]
    if isinstance(root, dict):
        return data_root / root[csv_name]
    return data_root / root


def add_record(records, group, csv_name, media_type, column, root, rel, refs):
    if not rel:
        return
    rel_path = Path(rel)
    abs_path = rel_path if rel_path.is_absolute() else root / rel_path
    key = (group, media_type, str(abs_path))
    if key in records:
        records[key].refs += int(refs)
    else:
        records[key] = FileRecord(
            group=group,
            csv_name=csv_name,
            media_type=media_type,
            column=column,
            path=abs_path,
            relative_path=rel,
            refs=int(refs),
        )


def collect_records(repo_data_dir, data_root, groups, limit_per_csv=None):
    records = {}
    csv_stats = {}

    for group, csv_names in groups.items():
        for csv_name in csv_names:
            csv_path = repo_data_dir / csv_name
            csv_stats[(group, csv_name)] = {"exists": csv_path.exists(), "rows": 0}
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if limit_per_csv is not None:
                df = df.head(limit_per_csv)
            csv_stats[(group, csv_name)]["rows"] = len(df)

            root = resolve_root(data_root, group, csv_name)

            for col in VIDEO_COLUMNS:
                if col not in df.columns:
                    continue
                for rel, refs in df[col].dropna().astype(str).value_counts().items():
                    add_record(records, group, csv_name, "video", col, root, rel, refs)

            for col in IMAGE_COLUMNS:
                if col not in df.columns:
                    continue
                for rel, refs in df[col].dropna().astype(str).value_counts().items():
                    add_record(records, group, csv_name, "image", col, root, rel, refs)

            if group in REGRESSION_ROOTS:
                for col in NPY_COLUMNS:
                    if col not in df.columns:
                        continue
                    for rel, refs in df[col].dropna().astype(str).value_counts().items():
                        add_record(records, group, csv_name, "npy", col, root, rel, refs)

    return list(records.values()), csv_stats


def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return "ok", ""
    except FileNotFoundError:
        return "missing", "file does not exist"
    except Exception as exc:
        return "corrupt", f"{type(exc).__name__}: {exc}"


def check_video(path, timeout):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            if not path.exists():
                return "missing", "file does not exist"
            reason = result.stderr.strip() or "ffprobe could not open video"
            return "corrupt", reason.replace("\n", " ")
        if "video" not in result.stdout:
            return "corrupt", "ffprobe found no video stream"
        return "ok", ""
    except FileNotFoundError as exc:
        if exc.filename == "ffprobe":
            return "corrupt", "ffprobe command not found"
        return "missing", "file does not exist"
    except subprocess.TimeoutExpired:
        return "corrupt", f"ffprobe timeout after {timeout}s"
    except Exception as exc:
        return "corrupt", f"{type(exc).__name__}: {exc}"


def check_npy(path):
    try:
        if not path.exists():
            return "missing", "file does not exist"
        if not path.is_file():
            return "corrupt", "path is not a file"
        return "ok", ""
    except Exception as exc:
        return "corrupt", f"{type(exc).__name__}: {exc}"


def check_record(record, timeout):
    if record.media_type == "video":
        status, reason = check_video(record.path, timeout)
    elif record.media_type == "npy":
        status, reason = check_npy(record.path)
    else:
        status, reason = check_image(record.path)
    return record, status, reason


def worker_loop(worker_id, task_queue, result_queue, timeout):
    while True:
        item = task_queue.get()
        if item is None:
            return
        idx, record = item
        result_queue.put(("start", worker_id, idx))
        result_queue.put(("done", worker_id, idx, check_record(record, timeout)))


def start_worker(worker_id, task_queue, result_queue, timeout):
    proc = mp.Process(target=worker_loop, args=(worker_id, task_queue, result_queue, timeout))
    proc.daemon = True
    proc.start()
    return proc


def check_records(records, workers, timeout):
    results = []
    resolved = set()
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    procs = {}
    active = {}
    worker_task = {}
    next_idx = 0
    completed = 0
    worker_count = min(workers, len(records))

    for worker_id in range(worker_count):
        procs[worker_id] = start_worker(worker_id, task_queue, result_queue, timeout)

    def submit_next():
        nonlocal next_idx
        if next_idx >= len(records):
            return False
        task_queue.put((next_idx, records[next_idx]))
        next_idx += 1
        return True

    for _ in range(worker_count):
        submit_next()

    with tqdm(total=len(records), desc="Checking media/npy files") as pbar:
        while completed < len(records):
            try:
                message = result_queue.get(timeout=0.1)
            except queue.Empty:
                message = None

            if message is not None:
                kind = message[0]
                if kind == "start":
                    _, worker_id, idx = message
                    if idx in resolved:
                        continue
                    active[worker_id] = (idx, time.monotonic())
                    worker_task[worker_id] = idx
                elif kind == "done":
                    _, worker_id, idx, result = message
                    if idx in resolved:
                        continue
                    active.pop(worker_id, None)
                    worker_task.pop(worker_id, None)
                    resolved.add(idx)
                    results.append(result)
                    completed += 1
                    pbar.update(1)
                    submit_next()

            now = time.monotonic()
            for worker_id, (idx, start_time) in list(active.items()):
                if idx in resolved:
                    active.pop(worker_id, None)
                    worker_task.pop(worker_id, None)
                    continue
                if now - start_time <= timeout:
                    continue
                proc = procs.get(worker_id)
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1)
                active.pop(worker_id, None)
                worker_task.pop(worker_id, None)
                record = records[idx]
                resolved.add(idx)
                results.append((record, "corrupt", f"timeout after {timeout}s"))
                completed += 1
                pbar.update(1)
                procs[worker_id] = start_worker(worker_id, task_queue, result_queue, timeout)
                submit_next()

            for worker_id, proc in list(procs.items()):
                if worker_id in active:
                    continue
                if proc.is_alive():
                    continue
                idx = worker_task.pop(worker_id, None)
                if idx is not None and idx not in resolved:
                    resolved.add(idx)
                    results.append((records[idx], "corrupt", "worker exited before returning result"))
                    completed += 1
                    pbar.update(1)
                    submit_next()
                if completed < len(records):
                    procs[worker_id] = start_worker(worker_id, task_queue, result_queue, timeout)

    for _ in procs:
        task_queue.put(None)
    for proc in procs.values():
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)

    return results


def summarize(results):
    summary = {}
    issues = {}
    for record, status, reason in results:
        key = (record.group, record.media_type)
        if key not in summary:
            summary[key] = {"total": 0, "ok": 0, "missing": 0, "corrupt": 0}
        summary[key]["total"] += 1
        summary[key][status] += 1
        if status != "ok":
            issues.setdefault(record.group, []).append((record, status, reason))
    return summary, issues


def write_report(output, data_root, selected_groups, csv_stats, summary, issues):
    groups = list(selected_groups.keys())
    with output.open("w") as f:
        f.write("LARYBench Dataset Integrity Report\n")
        f.write(f"Data root: {data_root}\n")
        f.write("Check method: open image with PIL verify; open video with ffprobe; check npy existence\n\n")

        f.write("Metadata CSVs\n")
        for group in groups:
            f.write(f"\n[{group}]\n")
            for csv_name in DEFAULT_GROUPS[group]:
                stat = csv_stats.get((group, csv_name), {"exists": False, "rows": 0})
                state = "exists" if stat["exists"] else "missing"
                f.write(f"  {csv_name}: {state}, rows={stat['rows']}\n")

        f.write("\nSummary\n")
        for group in groups:
            f.write(f"\n[{group}]\n")
            for media_type in ("video", "image", "npy"):
                stat = summary.get((group, media_type), {"total": 0, "ok": 0, "missing": 0, "corrupt": 0})
                f.write(
                    f"  {media_type}: total={stat['total']} "
                    f"ok={stat['ok']} missing={stat['missing']} corrupt={stat['corrupt']}\n"
                )

        f.write("\nIssues\n")
        for group in groups:
            group_issues = issues.get(group, [])
            f.write(f"\n[{group}] issue_count={len(group_issues)}\n")
            for record, status, reason in group_issues:
                f.write(
                    f"  {status}\t{record.media_type}\trefs={record.refs}\t"
                    f"csv={record.csv_name}\tcolumn={record.column}\t"
                    f"path={record.path}\treason={reason}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Check LARYBench images, videos, and regression npy files.")
    parser.add_argument("--repo-data-dir", type=Path, default=Path("data"),
                        help="Directory containing metadata CSVs in this repository.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="LARYBench dataset root containing classification/ and regression/.")
    parser.add_argument("--output", type=Path, default=Path("dataset_integrity_report.txt"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout in seconds for opening each video with ffprobe.")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_GROUPS.keys()),
                        help=f"Comma-separated dataset groups to check. Choices: {', '.join(DEFAULT_GROUPS.keys())}.")
    parser.add_argument("--limit-per-csv", type=int, default=None,
                        help="Debug option: only check the first N rows of each CSV.")
    args = parser.parse_args()

    selected_names = [name.strip() for name in args.groups.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in DEFAULT_GROUPS]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}. Choices: {list(DEFAULT_GROUPS)}")
    selected_groups = {name: DEFAULT_GROUPS[name] for name in selected_names}

    records, csv_stats = collect_records(args.repo_data_dir, args.data_root, selected_groups, args.limit_per_csv)
    results = check_records(records, args.workers, args.timeout)
    summary, issues = summarize(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, args.data_root, selected_groups, csv_stats, summary, issues)
    print(f"Checked {len(records)} unique media files. Report saved to {args.output}")


if __name__ == "__main__":
    main()
