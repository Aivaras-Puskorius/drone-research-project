#!/usr/bin/env python3
"""Train the detector from random weights on a tile set: python -m drone_model.train.

Settings come from configs/train.yaml, with --set overrides. Training is deterministic for a given seed,
tile set, GPU and library versions. Ultralytics saves last.pt after every epoch, so a stopped run continues
with --resume and loses at most the epoch in progress. A resumed run is valid but not bit-identical to an
uninterrupted one, because ultralytics resumes from its saved half-precision copy of the weights.

--gpu-share-file names a file holding 1-100, the percent of normal pace; edit it while training to slow the
run down (gpu_share.py). The final weights are <out>/<name>/weights/last.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import yaml
from ultralytics import YOLO

from drone_model.gpu_share import GpuShare
from drone_model.versions import runtime_versions

CONFIG = Path(__file__).resolve().parents[1] / 'configs/train.yaml'


def sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def finished_epoch(last: Path) -> int:
    """The last finished epoch (0-based) stored in an ultralytics last.pt, or -1 once training has ended."""
    return int(torch.load(last, map_location='cpu', weights_only=False).get('epoch', -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help="the tile set's dataset.yaml")
    parser.add_argument('--out', type=Path, required=True, help='folder for training runs')
    parser.add_argument('--name', required=True, help='run name, a subfolder of --out')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--model', default='yolo26n.yaml', help='ultralytics model definition, e.g. yolo26s.yaml for the larger network')
    parser.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE', help='override a setting, e.g. epochs=2')
    parser.add_argument('--resume', action='store_true', help='continue a stopped run with the same arguments')
    parser.add_argument('--gpu-share-file', type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    for item in args.set:
        key, value = item.split('=', 1)
        settings[key] = yaml.safe_load(value)
    settings['seed'] = args.seed
    out, data = args.out.resolve(), args.data.resolve()
    run, record_path = out / args.name, out / f'{args.name}.json'
    last = run / 'weights' / 'last.pt'
    share = GpuShare(args.gpu_share_file.resolve() if args.gpu_share_file else None)
    started = time.time()
    if args.resume:
        record = json.loads(record_path.read_text())
        if record['status'] != 'running' or record['settings'] != settings or record['data'] != str(data):
            raise ValueError('--resume needs a stopped run with the same data and settings')
        if not last.exists():
            raise FileNotFoundError(f'{last} does not exist yet; start the run again without --resume')
        epoch = finished_epoch(last)
        if epoch < 0:
            # WARN: train(resume=True) on a finished (stripped) last.pt silently starts a new training from it.
            raise ValueError(f'{args.name} has already finished')
        record.setdefault('resumed_after_epochs', []).append(epoch + 1)
        write(record_path, record)
        model = YOLO(str(last))
    else:
        if run.exists() or record_path.exists():
            raise FileExistsError(f'{run} exists; use --resume to continue it')
        out.mkdir(parents=True, exist_ok=True)
        record = {'status': 'running', 'data': str(data), 'data_sha256': sha(data), 'model': args.model, 'settings': settings,
                  'versions': runtime_versions(args.device), 'started_unix_s': started}
        write(record_path, record)
        model = YOLO(args.model)  # random weights; the input channel count comes from dataset.yaml
    for event in ('on_train_batch_end', 'on_val_batch_end'):
        model.add_callback(event, share.callback)
    model.add_callback('on_train_epoch_start', share.restart)
    if args.resume:
        model.train(resume=True, data=str(data), device=args.device)
    else:
        model.train(data=str(data), device=args.device, project=str(out), name=args.name, **settings)
    record.update(status='complete', train_seconds=record.get('train_seconds', 0) + time.time() - started,
                  gpu_share_sleep_s=share.slept_s, last_sha256=sha(last))
    write(record_path, record)
    print(json.dumps({'weights': str(last), 'sha256': record['last_sha256']}))


if __name__ == '__main__':
    main()
