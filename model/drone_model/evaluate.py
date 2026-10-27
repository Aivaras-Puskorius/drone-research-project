#!/usr/bin/env python3
"""Run detectors on labelled video windows, track and score: python -m drone_model.evaluate.

For every window of the plan, each frame is decoded in order and cut into tiles; each model proposes boxes
above a 0.005 confidence floor, and the proposals are saved. Then, for each confidence threshold, the most
confident proposal per frame goes to the tracker (tracking.py) and the tracked boxes are scored against the
labels (scoring.py). Saved proposals are reused, so rescoring at other thresholds needs no GPU.

Window plan (JSON): {"windows": [{"key", "video", "video_sha256", "dimensions": [w, h], "fps", "start", "count",
"references": [{"source_frame_zero_based", "boxes_xywh", "annotation", "annotation_sha256"}, ...]}]}, one
reference per frame; "boxes_xywh" is [] for a frame labelled empty and null for an unlabelled one, and
"annotation" (optional) names the label file it came from. The video and label files are checked against
their SHA-256 before use. The plan's windows need start >= 4, so that f-4 exists.

Outputs in --out: run.json (models, plan, settings, library versions), proposals/<model>/<window>.jsonl,
metrics/<model>/conf<threshold>.json (per-window counts and per-frame distances) and summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import statistics
import threading
import time

import cv2
import numpy as np

from drone_model.alignment import background_shift
from drone_model.gpu_share import GpuShare
from drone_model.inputs import ALIGNMENTS, gpu_alignment_possible, predict_tiles, tile_inputs
from drone_model.scoring import COUNTS, score_window, summarize
from drone_model.tracking import replay_window
from drone_model.versions import runtime_versions

FLOOR = 0.005
PAST = 4


def sha(path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(text)
    os.replace(temporary, path)


def model_name(weights: Path) -> str:
    """<run> for <run>/weights/last.pt, else the file name without its suffix."""
    return weights.parent.parent.name if weights.parent.name == 'weights' else weights.stem


def check_plan(plan: dict, check_videos: bool) -> None:
    """Every window has a label per frame, and its video and label files are the ones the plan was built from."""
    checked = {}

    def same(path: str, expected: str) -> None:
        if path not in checked:
            checked[path] = sha(path)
        if checked[path] != expected:
            raise ValueError(f'{path} differs from the window plan')

    for window in plan['windows']:
        if window['start'] < PAST or len(window['references']) != window['count']:
            raise ValueError(f"{window['key']}: needs start >= {PAST} and one reference per frame")
        if check_videos:
            same(window['video'], window['video_sha256'])
        for reference in window['references']:
            for key in ('annotation', 'absent_file'):
                if reference.get(key):
                    same(reference[key], reference[f'{key}_sha256'])


def prefetch(items, depth: int = 1):
    """Iterate `items` in a background thread, at most `depth` items ahead, so decoding and CPU alignment of the
    next frame overlap the predictions of this one. An error in the thread is raised here."""
    queue, end = Queue(maxsize=depth), object()

    def fill():
        try:
            for item in items:
                queue.put((item, None))
        except BaseException as error:  # handed to the consumer, which raises it
            queue.put((None, error))
        else:
            queue.put((end, None))

    worker = threading.Thread(target=fill, daemon=True)
    worker.start()
    while True:
        item, error = queue.get()
        if error is not None:
            raise error
        if item is end:
            break
        yield item
    worker.join()


def infer_window(window: dict, models: dict, args, share: GpuShare) -> tuple[dict, dict]:
    """Per model, one row per frame with the background shift and the proposals; and seconds per frame."""
    video, start, count = window['video'], int(window['start']), int(window['count'])
    width, height = window['dimensions']
    gpu = not args.cpu_alignment and gpu_alignment_possible(width, height, args.alignment, args.device)

    def frames():
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start - PAST)
        grays = {}
        try:
            for frame in range(start - PAST, start + count):
                okay, bgr = capture.read()
                if not okay or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > .5:
                    raise OSError(f'exact decode failed: {video} frame {frame}')
                if bgr.shape[:2] != (height, width):
                    raise ValueError(f'{video}: frame size differs from the plan')
                grays[frame] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                grays.pop(frame - PAST - 1, None)
                if frame < start:
                    continue
                began = time.perf_counter()
                stack = np.stack((grays[frame - 4], grays[frame - 2], grays[frame]), axis=2)
                built = tile_inputs(bgr, stack, args.alignment, device=args.device, gpu=gpu)
                yield frame, list(background_shift(grays[frame - 1], grays[frame])), built, time.perf_counter() - began
        finally:
            capture.release()

    rows = {name: [] for name in models}
    seconds = {name: [] for name in models}
    for frame, shift, (windows, inputs), build_s in prefetch(frames()):
        busy = build_s if gpu else 0.0
        for name, model in models.items():
            began = time.perf_counter()
            found = predict_tiles(model, windows, inputs, FLOOR, device=args.device)
            if args.device != 'cpu':
                import torch
                torch.cuda.synchronize()
            busy += time.perf_counter() - began
            seconds[name].append(build_s + time.perf_counter() - began)
            rows[name].append({'frame': frame, 'background_shift': shift, 'proposals': found})
        share.wait(busy)
    return rows, seconds


def score(out: Path, plan: dict, names: list[str], thresholds: list[float], hits: int) -> None:
    table = []
    for name in names:
        (out / 'metrics' / name).mkdir(parents=True, exist_ok=True)
        cached = {w['key']: [json.loads(line) for line in (out / 'proposals' / name / f"{w['key']}.jsonl").read_text().splitlines()]
                  for w in plan['windows']}
        for threshold in thresholds:
            results = []
            for window in plan['windows']:
                width, height = window['dimensions']
                frames, events = replay_window(cached[window['key']], window['fps'], width, height,
                                               threshold=threshold, hits_required=hits)
                results.append(score_window(window, frames, events))
            total = summarize(results)
            write(out / 'metrics' / name / f'conf{threshold:g}.json',
                  json.dumps({'model': name, 'threshold': threshold, 'hits_required': hits, 'summary': total,
                              'windows': results}, indent=1) + '\n')
            table.append({'model': name, 'threshold': threshold, **total})
    with open(out / 'summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'threshold', 'windows', *COUNTS])
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, nargs='+', required=True)
    parser.add_argument('--windows', type=Path, required=True, help='window plan (see above)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alignment', choices=ALIGNMENTS, default='shift',
                        help="per-tile alignment after the whole-frame shift: 'shift' (as in training) or 'homography' (CPU, slower)")
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.25, 0.1])
    parser.add_argument('--hits', type=int, default=3, help='consecutive detections that start a track')
    parser.add_argument('--device', default='0', help="CUDA device number or 'cpu'")
    parser.add_argument('--cpu-alignment', action='store_true',
                        help='align on the CPU even where the GPU could (the GPU result differs by 1 gray level in a few pixels)')
    parser.add_argument('--skip-video-hash', action='store_true', help='do not check the videos against the plan (slow for large files)')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted --out folder; finished windows are kept')
    parser.add_argument('--gpu-share-file', type=Path, help='file holding 1-100, the percent of normal pace (gpu_share.py)')
    args = parser.parse_args()
    out, plan_path = args.out.resolve(), args.windows.resolve()
    plan = json.loads(plan_path.read_text())
    check_plan(plan, not args.skip_video_hash)
    weights = {model_name(w): w.resolve() for w in args.weights}
    if len(weights) != len(args.weights):
        raise ValueError('two weight files have the same model name')
    identity = {'models': {n: sha(w) for n, w in weights.items()}, 'windows_sha256': sha(plan_path),
                'alignment': args.alignment, 'cpu_alignment': args.cpu_alignment, 'proposal_floor': FLOOR}
    if out.exists():
        if not args.resume:
            raise FileExistsError(f'{out} exists; use --resume to continue it')
        run = json.loads((out / 'run.json').read_text())
        if run['identity'] != identity:
            raise ValueError('--resume needs the same models, window plan and alignment')
    else:
        out.mkdir(parents=True)
        run = {'identity': identity, 'weights': {n: str(w) for n, w in weights.items()}, 'windows': str(plan_path),
               'versions': runtime_versions(args.device), 'started_unix_s': time.time(), 'seconds_per_frame': {}}
    write(out / 'run.json', json.dumps(run, indent=2) + '\n')
    from ultralytics import YOLO
    models = {name: YOLO(str(path)) for name, path in weights.items()}
    share = GpuShare(args.gpu_share_file.resolve() if args.gpu_share_file else None)
    for window in plan['windows']:
        key = window['key']
        if all((out / 'proposals' / name / f'{key}.jsonl').exists() for name in models):
            continue
        rows, seconds = infer_window(window, models, args, share)
        for name in models:
            (out / 'proposals' / name).mkdir(parents=True, exist_ok=True)
            write(out / 'proposals' / name / f'{key}.jsonl', ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows[name]))
            run['seconds_per_frame'].setdefault(name, {})[key] = statistics.median(seconds[name])
        write(out / 'run.json', json.dumps(run, indent=2) + '\n')
        print(f"{key}: {window['count']} frames x {len(models)} models", flush=True)
    score(out, plan, list(models), args.thresholds, args.hits)
    run['finished_unix_s'] = time.time()
    write(out / 'run.json', json.dumps(run, indent=2) + '\n')
    print((out / 'summary.csv').read_text())


if __name__ == '__main__':
    main()
