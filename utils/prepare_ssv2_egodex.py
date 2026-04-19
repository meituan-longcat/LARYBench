"""
Prepare SSv2 and EgoDex clips for LARYBench.

This script reads seed_ssv2.csv and seed_egodex.csv (which contain LLM-generated
time annotations in `seed_output`), clips the required segments using ffmpeg,
and produces reversed clips (those with a `_rev` suffix) using OpenCV.

Only clips actually referenced in human_1st_metadata_train.csv and
human_1st_metadata_val.csv are produced — everything else is skipped.

Clip naming conventions (mirrors the original clip.py exactly)
--------------------------------------------------------------
SSv2:
    SSv2/<seed_row_index>_<line_idx>[_rev].mp4

EgoDex:
    EgoDex/<video_basename>/<seed_row_index>_<line_idx>[_rev].mp4

    seed_row_index  = pandas integer index of the row in the seed CSV
    video_basename  = stem of the original video file  (e.g. "3279")
    line_idx        = 0-based index of the LINE within seed_output
                      (ALL lines are counted, including empty / non-matching ones;
                       only lines that contain a valid "start - end" time range
                       are actually clipped — this matches clip.py's enumerate loop)

Usage
-----
python data/prepare_ssv2_egodex.py \\
    --ssv2-root   /path/to/SSV2          \\   # dir containing 100020.webm etc.
    --egodex-root /path/to/EgoDex        \\   # dir containing part1/…/1068.mp4 etc.
    --output-dir  /path/to/classification \\   # clips written under output-dir/SSv2/ and output-dir/EgoDex/
    --workers     16
"""

import argparse
import os
import re
import subprocess
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to this script — lives in LARYBench/data/)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SEED_SSV2   = os.path.join(SCRIPT_DIR, "seed_ssv2.csv")
SEED_EGODEX = os.path.join(SCRIPT_DIR, "seed_egodex.csv")
META_TRAIN  = os.path.join(SCRIPT_DIR, "human_1st_metadata_train.csv")
META_VAL    = os.path.join(SCRIPT_DIR, "human_1st_metadata_val.csv")


# ---------------------------------------------------------------------------
# Helpers — identical logic to the original clip.py
# ---------------------------------------------------------------------------

def parse_clips(seed_output: str) -> list[tuple[int, float, float]]:
    """
    Parse seed_output and return a list of (line_idx, start_t, duration).

    line_idx counts ALL lines (including empty / non-matching ones), exactly
    as the original clip.py does with `for line_idx, line in enumerate(lines)`.
    Only lines that contain a valid "start - end" time range are included.
    """
    lines = str(seed_output).replace('\\n', '\n').strip().split('\n')
    clips = []
    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue
        matches = re.findall(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", line)
        if not matches:
            continue
        start_t, end_t = map(float, matches[0])
        duration = round(end_t - start_t, 2)
        if duration > 0:
            clips.append((line_idx, start_t, duration))
    return clips


def ffmpeg_clip(video_in: str, start_t: float, duration: float, out_path: str) -> bool:
    """Clip a segment with ffmpeg (same flags as original clip.py)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_t),
        '-t', str(duration),
        '-i', video_in,
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-an',
        '-loglevel', 'error',
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def reverse_video_opencv(src: str, dst: str) -> bool:
    """Reverse a video frame-by-frame using OpenCV (same logic as inverse_video.py)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return False
    try:
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return False
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dst, fourcc, fps, (width, height))
        for frame in reversed(frames):
            out.write(frame)
        out.release()
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception as e:
        print(f"  [reverse error] {src}: {e}")
        if os.path.exists(dst):
            os.remove(dst)
        return False


# ---------------------------------------------------------------------------
# Build the set of required clip paths from the metadata CSVs
# ---------------------------------------------------------------------------

def load_required_paths() -> set[str]:
    """
    Return the set of relative video_path values that appear in the two
    metadata CSVs and belong to SSv2 or EgoDex.
    """
    required = set()
    for csv_path in (META_TRAIN, META_VAL):
        df = pd.read_csv(csv_path)
        mask = df['dataset'].isin(['SSv2', 'EgoDex'])
        required.update(df.loc[mask, 'video_path'].tolist())
    return required


# ---------------------------------------------------------------------------
# Per-source clip builders
# ---------------------------------------------------------------------------

def build_ssv2_tasks(ssv2_root: str, output_dir: str, required: set[str]) -> list[dict]:
    """
    SSv2 naming: SSv2/<seed_row_index>_<line_idx>[_rev].mp4
    (flat directory, no video_basename subfolder)
    """
    df = pd.read_csv(SEED_SSV2)
    tasks = []
    for row_idx, row in df.iterrows():
        clips = parse_clips(row['seed_output'])
        src   = os.path.join(ssv2_root, os.path.basename(str(row['video_path'])))

        for line_idx, start_t, duration in clips:
            for rev in (False, True):
                suffix = '_rev' if rev else ''
                key = f"SSv2/{row_idx}_{line_idx}{suffix}.mp4"
                if key not in required:
                    continue
                tasks.append(dict(
                    src=src,
                    start_t=start_t,
                    duration=duration,
                    out_path=os.path.join(output_dir, key),
                    rev=rev,
                ))
    return tasks


def build_egodex_tasks(egodex_root: str, output_dir: str, required: set[str]) -> list[dict]:
    """
    EgoDex naming: EgoDex/<video_basename>/<seed_row_index>_<line_idx>[_rev].mp4

    Original paths look like: .../unzipped/part1/add_remove_lid/1068.mp4
    We keep everything after "unzipped/" when reconstructing under egodex_root.
    """
    df = pd.read_csv(SEED_EGODEX)
    tasks = []
    for row_idx, row in df.iterrows():
        orig_path      = str(row['video_path'])
        video_basename = os.path.splitext(os.path.basename(orig_path))[0]
        clips          = parse_clips(row['seed_output'])

        rel = re.split(r'unzipped/', orig_path, maxsplit=1)
        src = os.path.join(egodex_root, rel[1] if len(rel) == 2 else os.path.basename(orig_path))

        for line_idx, start_t, duration in clips:
            for rev in (False, True):
                suffix = '_rev' if rev else ''
                key = f"EgoDex/{video_basename}/{row_idx}_{line_idx}{suffix}.mp4"
                if key not in required:
                    continue
                tasks.append(dict(
                    src=src,
                    start_t=start_t,
                    duration=duration,
                    out_path=os.path.join(output_dir, key),
                    rev=rev,
                ))
    return tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_task(task: dict) -> tuple[bool, str]:
    src      = task['src']
    out_path = task['out_path']
    rev      = task['rev']

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, out_path

    if not os.path.exists(src):
        return False, f"source not found: {src}"

    if rev:
        fwd_path = out_path.replace('_rev.mp4', '.mp4')
        if not (os.path.exists(fwd_path) and os.path.getsize(fwd_path) > 0):
            return False, f"forward clip missing: {fwd_path}"
        ok = reverse_video_opencv(fwd_path, out_path)
        return ok, out_path if ok else f"reverse failed: {fwd_path}"
    else:
        ok = ffmpeg_clip(src, task['start_t'], task['duration'], out_path)
        return ok, out_path if ok else f"ffmpeg failed: {src}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clip SSv2 and EgoDex segments required by LARYBench."
    )
    parser.add_argument('--ssv2-root',   required=True,
                        help='Root dir of SSv2 raw videos (contains *.webm files)')
    parser.add_argument('--egodex-root', required=True,
                        help='Root dir of EgoDex (the "unzipped/" folder or its parent)')
    parser.add_argument('--output-dir',  required=True,
                        help='Output root; clips go to <output-dir>/SSv2/ and <output-dir>/EgoDex/')
    parser.add_argument('--workers',     type=int, default=16,
                        help='Number of parallel worker processes (default: 16)')
    args = parser.parse_args()

    print("Loading required clip paths from metadata CSVs …")
    required = load_required_paths()
    print(f"  {len(required)} clips needed (SSv2 + EgoDex combined)")

    print("Building task list …")
    tasks  = build_ssv2_tasks(args.ssv2_root,   args.output_dir, required)
    tasks += build_egodex_tasks(args.egodex_root, args.output_dir, required)
    print(f"  {len(tasks)} tasks queued")

    if not tasks:
        print("Nothing to do.")
        return

    # Two-pass execution: forward clips first, then reversed clips.
    # This avoids race conditions where a _rev worker starts before its
    # corresponding forward clip has been written.
    fwd_tasks = [t for t in tasks if not t['rev']]
    rev_tasks = [t for t in tasks if t['rev']]

    ok_count   = 0
    fail_count = 0
    failures   = []

    def run_batch(batch, desc):
        nonlocal ok_count, fail_count
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_task, t): t for t in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                ok, msg = future.result()
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    failures.append(msg)

    print(f"Pass 1/2 — forward clips ({len(fwd_tasks)} tasks) …")
    run_batch(fwd_tasks, "Forward clips")

    print(f"Pass 2/2 — reversed clips ({len(rev_tasks)} tasks) …")
    run_batch(rev_tasks, "Reversed clips")

    print(f"\nDone.  success={ok_count}  failed={fail_count}")
    if failures:
        print("Failed tasks (first 20):")
        for f in failures[:20]:
            print(f"  {f}")


if __name__ == '__main__':
    main()
