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

Two options add tiles; each draws from its own random generator, so the tiles above keep their names, places
and pixels-before-pasting:
- --cars FOLDER: in half of the sampled frames, 1-3 car patches cut from the renders (BGRA PNG files) are pasted
  into f-4, f-2 and f before alignment, each rotated, scaled 0.6-1.2, mirrored at random and moving 2-8 px per
  frame in a straight line, inside the frame, their centres at least 92 px from every drone box. They are not
  labelled.
  One extra empty tile centred on a pasted car is added per such frame (name `_d0`), in every scene. The
  stand-out rule is decided on the tile without the cars.
- --cars-control (with --cars): the same extra tiles at the same places, but no car is pasted anywhere, so the
  set differs from the --cars set only in the car pixels. It separates the effect of the cars from that of the
  extra empty tiles.
- --scale-copies: per sampled frame, one drone tile and one empty tile cut from the frame downscaled by 2 or 3
  (names `_s2_` or `_s3_`), so drones of about 7-35 px appear. They are cut from the frames without pasted cars,
  and the stand-out and no-empty rules apply to them like to every other tile. --scale-factors changes the
  factors drawn from (a fraction becomes `p` in names, `_s3p375_`), and --copies-per-frame adds more copies per
  frame, named `_s<factor>c<k>_` from the second one on.

--car-tile-scenes names rendered scenes with moving cars, whose folders also hold one `car*_flight.csv` per car
(same box columns as the drone files). Such a scene gives only car tiles: per sampled frame, one empty tile placed
like a drone tile around a visible car that no drone box overlaps (name `_rc`), drawn from its own generator; no
drone tiles, no other empty tiles, no pasted cars and no downscaled copies. Every other scene's tiles stay as they
were, and two renders of one scene with and without the cars, each holding the same CSVs, give the same tile
places, so they differ only in the car pixels.
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
          'centras2': ('0001-1000.mp4', [(0, 1000)]), 'ktuxi3': ('0001-1000.mp4', [(0, 1000)]),
          # Our own renders, made with the dataset_creation add-on in the centras2 scene (centras terrain, centras2
          # camera path): hovering, distant and moving drones. Not published yet.
          'r1_hover': ('0001-1000.mp4', [(0, 1000)]), 'r1_far': ('0001-1000.mp4', [(0, 1000)]),
          'r1_move': ('0001-1000.mp4', [(0, 1000)]),
          # The centras2 scene with moving cars added in Blender, and the same render with the cars hidden.
          'r2_cars': ('0001-1000.mp4', [(0, 1000)]), 'r2_nocars': ('0001-1000.mp4', [(0, 1000)])}
# Seeds of the option generators, as in the research builder (SEED + 2 was its degraded-video option).
SCALE_SEED, CARS_SEED, CAR_TILE_SEED = SEED + 1, SEED + 3, SEED + 4


def sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if lines and lines[0].startswith('sep='):
        lines = lines[1:]
    return list(csv.DictReader(lines))


def visible_boxes(scene_dir: Path, width: int, height: int, pattern: str = 'drone*_flight.csv') -> dict[int, list[list[float]]]:
    """Zero-based video frame -> visible boxes [x0, y0, x1, y1] of the files matching `pattern`, clipped to the frame."""
    boxes: dict[int, list[list[float]]] = {}
    for drone in sorted(scene_dir.glob(pattern)):
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


def car_tile_place(rng: random.Random, cars: list[list[float]], drones: list[list[float]], width: int,
                   height: int) -> tuple[int, int] | None:
    """A tile start placed like a drone tile around one of the cars (in a random order, up to 10 places each) that no
    drone box overlaps, or None."""
    for x0, y0, x1, y1 in rng.sample(cars, len(cars)):
        for _ in range(10):
            ox, oy = placement(rng, (x0 + x1) / 2, width), placement(rng, (y0 + y1) / 2, height)
            if all(b[2] <= ox or b[0] >= ox + TILE or b[3] <= oy or b[1] >= oy + TILE for b in drones):
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


def load_cars(folder: Path) -> list[np.ndarray]:
    patches = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in sorted(folder.glob('*.png'))]
    if not patches or any(p is None or p.ndim != 3 or p.shape[2] != 4 for p in patches):
        raise ValueError(f'{folder}: expected BGRA car patches')
    return patches


def paste(image: np.ndarray, patch: np.ndarray, cx: float, cy: float) -> None:
    """Alpha-blend a BGRA patch into a BGR or gray image in place, centred at (cx, cy)."""
    side_y, side_x = patch.shape[:2]
    x0, y0 = int(round(cx - side_x / 2)), int(round(cy - side_y / 2))
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(image.shape[1], x0 + side_x), min(image.shape[0], y0 + side_y)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    part = patch[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    alpha = part[:, :, 3:4].astype(np.float32) / 255
    source = part[:, :, :3].astype(np.float32)
    region = image[ay0:ay1, ax0:ax1]
    if image.ndim == 2:
        source = cv2.cvtColor(part[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
        region = region[:, :, None]
    blended = source * alpha + region.astype(np.float32) * (1 - alpha)
    image[ay0:ay1, ax0:ax1] = np.round(blended).astype(np.uint8).reshape(image[ay0:ay1, ax0:ax1].shape)


def car_patch(rng: random.Random, cars: list[np.ndarray]) -> np.ndarray:
    """One car patch, mirrored at random, rotated 0-360 degrees and scaled 0.6-1.2 on a square BGRA canvas."""
    patch = rng.choice(cars)
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    height, width = patch.shape[:2]
    angle, scale = rng.uniform(0, 360), rng.uniform(0.6, 1.2)
    side = int(np.ceil(np.hypot(height, width) * scale)) + 2
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += (side / 2 - width / 2, side / 2 - height / 2)
    return cv2.warpAffine(np.ascontiguousarray(patch), matrix, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def add_cars(color: np.ndarray, stack: np.ndarray, boxes: list[list[float]], rng: random.Random,
             cars: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Paste 1-3 moving cars into copies of f and the gray stack f-4, f-2, f; returns them and the cars' centres at f."""
    color, stack = color.copy(), stack.copy()
    height, width = color.shape[:2]
    centres = []
    for _ in range(rng.randint(1, 3)):
        speed, heading = rng.uniform(2, 8), rng.uniform(0, 2 * np.pi)
        vx, vy = speed * np.cos(heading), speed * np.sin(heading)
        for _ in range(50):
            cx, cy = rng.uniform(80, width - 80), rng.uniform(80, height - 80)
            path = [(cx + vx * o, cy + vy * o) for o in (-4, -2, 0)]
            inside = all(40 <= x < width - 40 and 40 <= y < height - 40 for x, y in path)
            # 32 px from the drone box plus up to 60 px for half a car; written as in the research builder, so the
            # floating-point sums and therefore the accepted places stay the same.
            clear = all(x1 + 32 + 60 <= px or x0 - 32 - 60 >= px or y1 + 32 + 60 <= py or y0 - 32 - 60 >= py
                        for x0, y0, x1, y1 in boxes for px, py in path)
            if inside and clear:
                break
        else:
            continue
        patch = car_patch(rng, cars)
        paste(color, patch, *path[-1])
        for channel, (px, py) in enumerate(path):
            plane = np.ascontiguousarray(stack[:, :, channel])
            paste(plane, patch, px, py)
            stack[:, :, channel] = plane
        centres.append(path[-1])
    return color, stack, centres


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--renders', type=Path, required=True, help='folder with one subfolder per scene')
    parser.add_argument('--out', type=Path, required=True, help='new folder for the tiles')
    parser.add_argument('--scenes', nargs='+', default=['centras1', 'ktuxi2', 'eiguliai1', 'amaliai1'], choices=sorted(SCENES),
                        help='processed in this order, which fixes the random tile places')
    parser.add_argument('--standout-scenes', nargs='*', default=['eiguliai1', 'amaliai1'], choices=sorted(SCENES))
    parser.add_argument('--no-empty-scenes', nargs='*', default=['amaliai1'], choices=sorted(SCENES))
    parser.add_argument('--cars', type=Path, help='folder of BGRA car patches to paste as unlabelled moving objects')
    parser.add_argument('--cars-control', action='store_true', help='with --cars: add the car tiles at the same places, without cars')
    parser.add_argument('--scale-copies', action='store_true', help='add tiles cut from the frame downscaled by 2 or 3')
    parser.add_argument('--scale-factors', type=float, nargs='+', default=[2, 3],
                        help='with --scale-copies: downscale factors to draw from (default 2 3); the frame must stay at least 640 px high')
    parser.add_argument('--copies-per-frame', type=int, default=1, help='with --scale-copies: downscaled copies per sampled frame')
    parser.add_argument('--car-tile-scenes', nargs='*', default=[], choices=sorted(SCENES),
                        help='rendered scenes with moving cars (car*_flight.csv) that give only car-centred empty tiles')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.cars_control and not args.cars:
        raise ValueError('--cars-control needs --cars, whose random car places it repeats')
    cars = load_cars(args.cars) if args.cars else []
    # Whole-number factors stay ints, so the default draws, sizes and names equal the research builder's.
    factors = tuple(int(f) if float(f).is_integer() else f for f in args.scale_factors)
    for kind in ('images', 'labels'):
        for split in ('train', 'val'):
            (out / kind / split).mkdir(parents=True)
    rng, rng_scale, rng_cars = random.Random(SEED), random.Random(SCALE_SEED), random.Random(CARS_SEED)
    rng_car_tiles = random.Random(CAR_TILE_SEED)
    tiles, dropped = [], []
    counts = {split: {'drone_tiles': 0, 'empty_tiles': 0, 'boxes': 0} for split in ('train', 'val')}

    def write(name, split, scene, frame, kind, crop, color, stack, labels, decide=None, scale=1, car_tile=False):
        """Write one tile, unless a rule drops it. `decide` holds the crops to judge stand-out on, if not these."""
        if (kind == 'drone') != bool(labels):
            raise ValueError(f'{name}: {kind} tile has {len(labels)} labels')
        if kind == 'empty' and scene in args.no_empty_scenes and not car_tile:
            return
        pixels = compose_tile(color, stack)
        if kind == 'drone' and split == 'train' and scene in args.standout_scenes:
            judged = pixels if decide is None else compose_tile(*decide)
            if not stands_out(judged, labels):
                dropped.append(name)
                return
        if not cv2.imwritemulti(str(out / 'images' / split / f'{name}.tiff'), [pixels[:, :, c] for c in range(CHANNELS)]):
            raise OSError(f'write failed: {name}')
        (out / 'labels' / split / f'{name}.txt').write_text(''.join(line + '\n' for line in labels))
        counts[split][f'{kind}_tiles'] += 1
        counts[split]['boxes'] += len(labels)
        tiles.append({'tile': name, 'scene': scene, 'frame': frame, 'split': split, 'kind': kind,
                      'crop_xywh': [*crop, TILE, TILE], 'labels': len(labels), 'scale': scale})

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
        rendered_cars = visible_boxes(args.renders / scene, width, height, 'car*_flight.csv') if scene in args.car_tile_scenes else {}
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
                if scene in args.car_tile_scenes:
                    spot = car_tile_place(rng_car_tiles, rendered_cars.get(frame, []), present, width, height)
                    if spot:
                        ox, oy = spot
                        window = (slice(oy, oy + TILE), slice(ox, ox + TILE))
                        stack = np.stack((grays[frame - 4], grays[frame - 2], grays[frame]), axis=2)
                        write(f'{scene}_f{frame:04d}_rc', split, scene, frame, 'empty', (ox, oy), bgr[window], stack[window], [],
                              car_tile=True)
                    continue
                places = []
                for x0, y0, x1, y1 in rng.sample(present, min(DRONE_TILES, len(present))):
                    places.append(('drone', placement(rng, (x0 + x1) / 2, width), placement(rng, (y0 + y1) / 2, height)))
                spot = empty_place(rng, present, width, height)
                if spot:
                    places.append(('empty', *spot))
                clean_stack = np.stack((grays[frame - 4], grays[frame - 2], grays[frame]), axis=2)
                color, stack, centres = bgr, clean_stack, []
                if cars and rng_cars.random() < 0.5:
                    color, stack, centres = add_cars(bgr, clean_stack, present, rng_cars, cars)
                    if args.cars_control:
                        color, stack = bgr, clean_stack  # same draws and tile places, no car pixels
                if centres:
                    ox, oy = placement(rng_cars, centres[0][0], width), placement(rng_cars, centres[0][1], height)
                    if all(x1 <= ox or x0 >= ox + TILE or y1 <= oy or y0 >= oy + TILE for x0, y0, x1, y1 in present):
                        write(f'{scene}_f{frame:04d}_d0', split, scene, frame, 'empty', (ox, oy),
                              color[oy:oy + TILE, ox:ox + TILE], stack[oy:oy + TILE, ox:ox + TILE], [], car_tile=True)
                for number, (kind, ox, oy) in enumerate(places):
                    window = (slice(oy, oy + TILE), slice(ox, ox + TILE))
                    write(f'{scene}_f{frame:04d}_{number}', split, scene, frame, kind, (ox, oy), color[window], stack[window],
                          tile_labels(present, ox, oy), decide=(bgr[window], clean_stack[window]) if centres else None)
                for copy in range(args.copies_per_frame if args.scale_copies else 0):
                    factor = rng_scale.choice(factors)
                    size = (int(width // factor), int(height // factor))
                    if min(size) < TILE:
                        raise ValueError(f'{scene}: factor {factor} leaves {size}, smaller than a {TILE} px tile')
                    small_color = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                    small_stack = np.dstack([cv2.resize(clean_stack[:, :, c], size, interpolation=cv2.INTER_AREA) for c in range(3)])
                    small_boxes = [b for b in ([v / factor for v in box] for box in present) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
                    extra = []
                    if small_boxes:
                        x0, y0, x1, y1 = rng_scale.choice(small_boxes)
                        extra.append(('drone', placement(rng_scale, (x0 + x1) / 2, size[0]), placement(rng_scale, (y0 + y1) / 2, size[1])))
                    spot = empty_place(rng_scale, small_boxes, *size)
                    if spot:
                        extra.append(('empty', *spot))
                    for number, (kind, ox, oy) in enumerate(extra):
                        window = (slice(oy, oy + TILE), slice(ox, ox + TILE))
                        tag = f's{factor}'.replace('.', 'p') + (f'c{copy}' if copy else '')
                        write(f'{scene}_f{frame:04d}_{tag}_{number}', split, scene, frame, kind, (ox, oy), small_color[window],
                              small_stack[window], tile_labels(small_boxes, ox, oy), scale=factor)
        finally:
            capture.release()
    (out / 'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nchannels: {CHANNELS}\nnames:\n  0: drone\n')
    record = {'rule': __doc__.strip(), 'seed': SEED, 'scenes': args.scenes, 'standout_scenes': args.standout_scenes,
              'no_empty_scenes': args.no_empty_scenes, 'scale_copies': args.scale_copies, 'cars_control': args.cars_control,
              'scale_factors': list(factors), 'copies_per_frame': args.copies_per_frame, 'car_tile_scenes': args.car_tile_scenes,
              'cars': {p.name: sha(p) for p in sorted(args.cars.glob('*.png'))} if args.cars else None,
              'videos_sha256': {s: sha(args.renders / s / SCENES[s][0]) for s in args.scenes},
              'versions': runtime_versions('cpu'), 'counts': counts, 'dropped_not_standing_out': dropped, 'tiles': tiles}
    (out / 'tiles.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'counts': counts, 'dropped_not_standing_out': len(dropped)}))


if __name__ == '__main__':
    main()
