#!/usr/bin/env python3
"""Build the detector's training tiles from rendered scenes: python -m drone_model.build_tiles.

Each scene folder holds the rendered 3840 x 2160 video and one `drone*_flight.csv` per drone with per-frame
boxes. The rule:
- A drone counts in a frame when `bbox_visible` is 1 and its box, clipped to the frame, has positive size.
  CSV frame n is video frame n-1.
- A scene video can hold several clips with a camera jump between them; the past frames f-4 and f-2 always
  come from the same clip as f.
- Sampled frames: every second frame from the fourth frame of each clip. Every tenth sampled frame goes to
  the validation split, the rest to training.
- Per sampled frame: up to 3 drone tiles, one per visible drone in a seeded random order, each placed so that
  drone's box centre lies at least 48 px inside the 640 px tile, and one empty tile at a seeded random place
  that no drone box overlaps. A tile's labels are every visible drone whose box centre falls inside it,
  clipped to the tile.
- Each tile has 5 channels (inputs.compose_tile with the per-tile shift): B, G, R of f, aligned f-2, aligned f-4.
- Scenes in --standout-scenes keep a training drone tile only when every drone in it stands out in the motion
  image: its brightest |f - aligned f-2| pixel is brighter than 99.9% of the tile's motion outside every
  drone box grown by its own size (at least 8 px). This drops drones the alignment cannot separate from
  moving background, such as parallax from trees and buildings.
- Scenes in --no-empty-scenes keep no empty tile.
The defaults give the training set of the current model (E1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

from drone_model.inputs import CHANNELS, TILE, compose_tile
from drone_model.versions import runtime_versions

MARGIN, DRONE_TILES, SEED, FIRST_FRAME = 48, 3, 20261012, 4
# Video file and zero-based [start, end) clips per scene.
SCENES = {'centras1': ('0001-0400.mp4', [(0, 400)]), 'ktuxi2': ('0001-0400.mp4', [(0, 400)]),
          'eiguliai1': ('0001-1120.mp4', [(0, 400), (400, 800), (800, 1120)]),
          'amaliai1': ('0001-1120.mp4', [(0, 400), (400, 800), (800, 1120)]),
          'centras2': ('0001-1000.mp4', [(0, 1000)]), 'ktuxi3': ('0001-1000.mp4', [(0, 1000)])}


def sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if lines and lines[0].startswith('sep='):
        lines = lines[1:]
    return list(csv.DictReader(lines))


def visible_boxes(scene_dir: Path, width: int, height: int) -> dict[int, list[list[float]]]:
    """Zero-based video frame -> visible drone boxes [x0, y0, x1, y1], clipped to the frame."""
    boxes: dict[int, list[list[float]]] = {}
    for drone in sorted(scene_dir.glob('drone*_flight.csv')):
        for row in read_csv(drone):
            if row['bbox_visible'] != '1':
                continue
            x0, y0, x1, y1 = (float(row[k]) for k in ('bbox_xmin_px', 'bbox_ymin_px', 'bbox_xmax_px', 'bbox_ymax_px'))
            x0, y0, x1, y1 = max(0., x0), max(0., y0), min(float(width), x1), min(float(height), y1)
            if x1 > x0 and y1 > y0:
                boxes.setdefault(int(row['frame']) - 1, []).append([x0, y0, x1, y1])
    return boxes


def placement(rng: random.Random, centre: float, size: int) -> int:
    """Tile start that puts the centre at least MARGIN inside the tile, or as close as the frame edge allows."""
    low, high = int(max(0, centre - TILE + MARGIN)), int(min(size - TILE, centre - MARGIN))
    if high < low:
        return int(min(max(0, centre - TILE / 2), size - TILE))
    return rng.randint(low, high)


def empty_place(rng: random.Random, boxes: list[list[float]], width: int, height: int) -> tuple[int, int] | None:
    """A random tile start that no box overlaps, or None after 50 tries."""
    for _ in range(50):
        ox, oy = rng.randint(0, width - TILE), rng.randint(0, height - TILE)
        if all(x1 <= ox or x0 >= ox + TILE or y1 <= oy or y0 >= oy + TILE for x0, y0, x1, y1 in boxes):
            return ox, oy
    return None


def tile_labels(boxes: list[list[float]], ox: int, oy: int) -> list[str]:
    """YOLO label lines for the boxes whose centre falls in the tile at (ox, oy), clipped to it."""
    lines = []
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2 - ox, (y0 + y1) / 2 - oy
        if not (0 <= cx < TILE and 0 <= cy < TILE):
            continue
        a, b = max(0., x0 - ox), max(0., y0 - oy)
        c, d = min(float(TILE), x1 - ox), min(float(TILE), y1 - oy)
        lines.append(f'0 {(a + c) / 2 / TILE:.6f} {(b + d) / 2 / TILE:.6f} {(c - a) / TILE:.6f} {(d - b) / TILE:.6f}')
    return lines


def stands_out(pixels: np.ndarray, labels: list[str]) -> bool:
    """True when every labelled drone's brightest motion pixel beats the 99.9th percentile of the background.

    Motion is |gray f - aligned f-2|. The background is the tile outside every box grown by max(8, w, h) px.
    """
    gray = cv2.cvtColor(np.ascontiguousarray(pixels[:, :, :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = np.abs(gray - pixels[:, :, 3].astype(np.float32))
    height, width = gray.shape
    background = np.ones(gray.shape, bool)
    peaks = []
    for line in labels:
        _, cx, cy, w, h = map(float, line.split())
        x0, x1 = int(round((cx - w / 2) * width)), int(round((cx + w / 2) * width))
        y0, y1 = int(round((cy - h / 2) * height)), int(round((cy + h / 2) * height))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        grow = max(8, x1 - x0, y1 - y0)
        background[max(y0 - grow, 0):min(y1 + grow, height), max(x0 - grow, 0):min(x1 + grow, width)] = False
        peaks.append(float(motion[y0:y1, x0:x1].max()))
    threshold = float(np.percentile(motion[background], 99.9))
    return bool(peaks) and all(peak > threshold for peak in peaks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--renders', type=Path, required=True, help='folder with one subfolder per scene')
    parser.add_argument('--out', type=Path, required=True, help='new folder for the tiles')
    parser.add_argument('--scenes', nargs='+', default=['centras1', 'ktuxi2', 'eiguliai1', 'amaliai1'], choices=sorted(SCENES),
                        help='processed in this order, which fixes the random tile places')
    parser.add_argument('--standout-scenes', nargs='*', default=['eiguliai1', 'amaliai1'], choices=sorted(SCENES))
    parser.add_argument('--no-empty-scenes', nargs='*', default=['amaliai1'], choices=sorted(SCENES))
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    for kind in ('images', 'labels'):
        for split in ('train', 'val'):
            (out / kind / split).mkdir(parents=True)
    rng = random.Random(SEED)
    tiles, dropped = [], []
    counts = {split: {'drone_tiles': 0, 'empty_tiles': 0, 'boxes': 0} for split in ('train', 'val')}
    for scene in args.scenes:
        video_name, clips = SCENES[scene]
        video = args.renders / scene / video_name
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise OSError(f'cannot open {video}')
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if clips[-1][1] != frame_count:
            raise ValueError(f'{scene}: clips end at {clips[-1][1]}, video has {frame_count} frames')
        boxes = visible_boxes(args.renders / scene, width, height)
        sampled = {f: i for i, f in enumerate(f for start, end in clips for f in range(start + FIRST_FRAME, end, 2))}
        clip_start = {f: start for start, end in clips for f in range(start, end)}
        grays = {}
        try:
            for frame in range(frame_count):
                okay, bgr = capture.read()
                if not okay:
                    raise OSError(f'decode failed: {video} frame {frame}')
                grays[frame] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                grays.pop(frame - 9, None)
                if frame not in sampled:
                    continue
                if frame - FIRST_FRAME < clip_start[frame]:
                    raise ValueError(f'{scene} frame {frame}: past frames cross a clip boundary')
                split = 'val' if sampled[frame] % 10 == 0 else 'train'
                present = boxes.get(frame, [])
                places = []
                for x0, y0, x1, y1 in rng.sample(present, min(DRONE_TILES, len(present))):
                    places.append(('drone', placement(rng, (x0 + x1) / 2, width), placement(rng, (y0 + y1) / 2, height)))
                spot = empty_place(rng, present, width, height)
                if spot:
                    places.append(('empty', *spot))
                stack = np.stack((grays[frame - 4], grays[frame - 2], grays[frame]), axis=2)
                for number, (kind, ox, oy) in enumerate(places):
                    name = f'{scene}_f{frame:04d}_{number}'
                    labels = tile_labels(present, ox, oy)
                    if (kind == 'drone') != bool(labels):
                        raise ValueError(f'{name}: {kind} tile has {len(labels)} labels')
                    if kind == 'empty' and scene in args.no_empty_scenes:
                        continue
                    pixels = compose_tile(bgr[oy:oy + TILE, ox:ox + TILE], stack[oy:oy + TILE, ox:ox + TILE])
                    if kind == 'drone' and split == 'train' and scene in args.standout_scenes and not stands_out(pixels, labels):
                        dropped.append(name)
                        continue
                    if not cv2.imwritemulti(str(out / 'images' / split / f'{name}.tiff'), [pixels[:, :, c] for c in range(CHANNELS)]):
                        raise OSError(f'write failed: {name}')
                    (out / 'labels' / split / f'{name}.txt').write_text(''.join(line + '\n' for line in labels))
                    counts[split][f'{kind}_tiles'] += 1
                    counts[split]['boxes'] += len(labels)
                    tiles.append({'tile': name, 'scene': scene, 'frame': frame, 'split': split, 'kind': kind,
                                  'crop_xywh': [ox, oy, TILE, TILE], 'labels': len(labels)})
        finally:
            capture.release()
    (out / 'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nchannels: {CHANNELS}\nnames:\n  0: drone\n')
    record = {'rule': __doc__.strip(), 'seed': SEED, 'scenes': args.scenes, 'standout_scenes': args.standout_scenes,
              'no_empty_scenes': args.no_empty_scenes, 'videos_sha256': {s: sha(args.renders / s / SCENES[s][0]) for s in args.scenes},
              'versions': runtime_versions('cpu'), 'counts': counts, 'dropped_not_standing_out': dropped, 'tiles': tiles}
    (out / 'tiles.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'counts': counts, 'dropped_not_standing_out': len(dropped)}))


if __name__ == '__main__':
    main()
