#!/usr/bin/env python3
"""Rebuild a test window plan, with labels, from your copy of a dataset: python -m drone_model.make_plans.

windows/<set>.json says where each of our test windows lies and how it was chosen, but holds no labels,
because the labels belong to each dataset's publisher. This script reads the labels of those frames from the
dataset's own files and writes the window plan that evaluate.py reads. It stops when a window's label files
differ from the ones we used, compared by SHA-256. evaluate.py checks the videos the same way.

Sets: ard100, nps, uav_anti_uav and local4k (our own flights, not public). Each windows file names the
folders --videos and --labels must point to, and how the labels are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

WINDOWS = Path(__file__).resolve().parents[1] / 'windows'
SETS = ('ard100', 'nps', 'uav_anti_uav', 'local4k')
_HASHES: dict[Path, str] = {}


def sha(path: Path) -> str:
    if path not in _HASHES:
        with path.open('rb') as handle:
            _HASHES[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return _HASHES[path]


def labels_digest(entries: list[tuple[str, str]]) -> str:
    """SHA-256 over 'relative name sha256' lines, sorted, one per label file a window reads."""
    return hashlib.sha256(''.join(f'{name} {digest}\n' for name, digest in sorted(set(entries))).encode()).hexdigest()


def reference(frame: int, boxes, path: Path | None = None, **extra) -> dict:
    item = {'source_frame_zero_based': frame, 'frame_one_based': frame + 1, 'boxes_xywh': boxes}
    if path is not None:
        item.update(annotation=str(path), annotation_sha256=sha(path))
    return {**item, **extra}


def voc_boxes(path: Path) -> tuple[list[int], list[list[float]]]:
    """Image size and drone boxes [x, y, w, h] of one ARD100 VOC XML file."""
    root = ET.parse(path).getroot()
    found = []
    for item in root.findall('object'):
        if (item.findtext('name') or '').lower() != 'drone':
            continue
        box = item.find('bndbox')
        x0, y0, x1, y1 = (float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f'invalid box in {path}')
        # The dataset authors' VOC converter subtracts one from the corner.
        found.append([x0 - 1, y0 - 1, x1 - x0, y1 - y0])
    return [int(root.findtext('size/width')), int(root.findtext('size/height'))], found


def ard100(window: dict, labels: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    clip = window['clip']
    files = {int(p.stem.rsplit('_', 1)[-1]) - 1: p for p in (labels / clip).glob(f'{clip}_*.xml')}
    references, entries = [], []
    for frame in range(window['start'], window['start'] + window['count']):
        path = files.get(frame)
        if path is None:
            references.append(reference(frame, None))
            continue
        size, boxes = voc_boxes(path)
        if size != window['dimensions']:
            raise ValueError(f'{path}: image size differs from the video')
        references.append(reference(frame, boxes, path))
        entries.append((f'{clip}/{path.name}', sha(path)))
    return references, entries


def nps(window: dict, labels: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    path = labels / f"{window['clip']}_refined.txt"
    width, height = window['dimensions']
    boxes = {}
    with path.open(newline='') as handle:
        for row in csv.reader(handle):
            frame_one_based, left, top, w, h = int(float(row[0])), *(float(v) for v in row[2:6])
            if 0 <= left + w / 2 < width and 0 <= top + h / 2 < height:
                boxes.setdefault(frame_one_based, []).append([left, top, w, h])
    references = [reference(f, boxes.get(f + 1), path) for f in range(window['start'], window['start'] + window['count'])]
    return references, [(path.name, sha(path))]


def uav_anti_uav(window: dict, labels: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    folder = labels / window['clip']
    truth, absent_path = folder / 'groundtruth_rect.txt', folder / 'absent.txt'
    boxes = [[float(v) for v in line.replace('\t', ',').split(',')] for line in truth.read_text().splitlines() if line.strip()]
    absent = [int(v) for v in absent_path.read_text().split()]
    if len(boxes) != len(absent):
        raise ValueError(f'{folder}: groundtruth_rect.txt and absent.txt have different lengths')
    references = []
    for frame in range(window['start'], window['start'] + window['count']):
        x, y, w, h = boxes[frame][:4]
        found = [] if absent[frame] else [[x, y, w, h]] if w > 0 and h > 0 else None
        references.append(reference(frame, found, truth, absent_file=str(absent_path), absent_file_sha256=sha(absent_path)))
    return references, [(f"{window['clip']}/{truth.name}", sha(truth)), (f"{window['clip']}/{absent_path.name}", sha(absent_path))]


def local4k(window: dict, labels: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    path = labels / f"{window['clip']}.csv"
    with path.open(newline='') as handle:
        rows = {int(r['frame']): r for r in csv.DictReader(handle)}
    references = []
    for frame in range(window['start'], window['start'] + window['count']):
        r = rows[frame]
        boxes = []
        if window['kind'] == 'present':
            length, width = float(r['mavic3e_l_px']), float(r['mavic3e_w_px'])
            boxes = [[float(r['pix_xc']) - length / 2, float(r['pix_yc']) - width / 2, length, width]]
        references.append(reference(frame, boxes, path))
    return references, [(path.name, sha(path))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', choices=SETS, required=True)
    parser.add_argument('--videos', type=Path, required=True, help="the dataset's video folder (see windows/<set>.json)")
    parser.add_argument('--labels', type=Path, help="the dataset's label folder; default: --videos")
    parser.add_argument('--out', type=Path, required=True, help='window plan to write, e.g. plans/ard100.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    listing = json.loads((WINDOWS / f'{args.set}.json').read_text())
    videos, labels = args.videos.resolve(), (args.labels or args.videos).resolve()
    reader = {'ard100': ard100, 'nps': nps, 'uav_anti_uav': uav_anti_uav, 'local4k': local4k}[args.set]
    windows = []
    for window in listing['windows']:
        video = videos / window['video']
        if not video.is_file():
            raise FileNotFoundError(video)
        references, entries = reader(window, labels)
        if labels_digest(entries) != window['labels_sha256']:
            raise ValueError(f"{window['key']}: your label files differ from the ones our results used")
        windows.append({**{k: v for k, v in window.items() if k != 'labels_sha256'}, 'video': str(video), 'references': references})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'set': args.set, 'rule': listing['rule'], 'labels': listing['labels'], 'windows': windows}) + '\n')
    print(f'{args.out}: {len(windows)} windows, {sum(w["count"] for w in windows)} frames')


if __name__ == '__main__':
    main()
