"""Model input tiles and tiled detection.

A frame is cut into 640 px tiles at a stride of 512 px (32 tiles for 3840 x 2160). Each tile becomes a
5-channel input: B, G, R of the current frame f, then the gray frames f-2 and f-4 aligned to f
(alignment.py). The detector proposes boxes per tile, and boxes from overlapping tiles whose centres lie
within 24 px of a more confident box are dropped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np

from drone_model.alignment import coarse_shift, homography_tile, shift_tile, shift_tiles_gpu

TILE, STRIDE = 640, 512
ALIGNMENTS = ('shift', 'homography')
CHANNELS = 5
CPU_TILE_WORKERS = 8  # measured: 8 threads 0.71-0.75 s per 4K frame for the slowest CPU alignment, 16 threads slower


def positions(size: int) -> list[int]:
    """Tile starts along one axis: every 512 px, plus one tile flush with the far edge."""
    return sorted(set(range(0, max(1, size - TILE + 1), STRIDE)) | {max(0, size - TILE)})


def tile_windows(width: int, height: int) -> list[tuple[int, int]]:
    """(x, y) of every tile, row by row."""
    return [(x, y) for y in positions(height) for x in positions(width)]


def compose_tile(color: np.ndarray, stack: np.ndarray, alignment: str = 'shift') -> np.ndarray:
    """5-channel tile from a BGR crop of f and the gray crops f-4, f-2, f: B, G, R, aligned f-2, aligned f-4."""
    if alignment not in ALIGNMENTS:
        raise ValueError(f'unknown alignment {alignment!r}')
    if color.shape[:2] != stack.shape[:2] or color.shape[2:] != (3,) or stack.shape[2:] != (3,):
        raise ValueError('expected matching BGR and f-4/f-2/f crops')
    stack = np.ascontiguousarray(stack)
    aligned = shift_tile(stack) if alignment == 'shift' else homography_tile(stack)
    return np.dstack([color, aligned[:, :, 1], aligned[:, :, 0]])


def _tiles_gpu(image: np.ndarray, windows, device):
    """Upload an H x W x C uint8 frame once and return its tiles as N x C x 640 x 640."""
    import torch
    target = torch.device('cpu' if str(device).lower() == 'cpu' else f'cuda:{device}')
    frame = torch.from_numpy(np.ascontiguousarray(image)).to(target)
    return torch.stack([frame[y:y + TILE, x:x + TILE] for x, y in windows]).permute(0, 3, 1, 2).contiguous()


def gpu_alignment_possible(width: int, height: int, alignment: str, device) -> bool:
    """The GPU path needs full 640 px tiles and the per-tile shift."""
    return alignment == 'shift' and str(device).lower() != 'cpu' and width >= TILE and height >= TILE


def tile_inputs(bgr: np.ndarray, stack: np.ndarray, alignment: str = 'shift', *, device='0', gpu: bool = False):
    """Every tile of one frame as model input: (windows, inputs).

    `stack` is the raw gray stack f-4, f-2, f. The whole-frame shift runs first, then the per-tile alignment.
    With gpu=True (see gpu_alignment_possible) inputs is a float N x 5 x 640 x 640 tensor on the device, scaled
    to 0-1 as ultralytics scales NumPy input; otherwise a list of H x W x 5 uint8 arrays.
    """
    height, width = stack.shape[:2]
    if bgr.shape[:2] != (height, width):
        raise ValueError('color and gray frames must have the same size')
    stack = coarse_shift(stack)
    windows = tile_windows(width, height)
    if gpu:
        import torch
        grays = shift_tiles_gpu(_tiles_gpu(stack, windows, device))
        colors = _tiles_gpu(bgr, windows, device)
        return windows, torch.cat([colors, grays[:, 1:2], grays[:, 0:1]], dim=1).float().div_(255)

    def one(window):
        x, y = window
        return compose_tile(bgr[y:y + TILE, x:x + TILE], stack[y:y + TILE, x:x + TILE], alignment)

    if alignment == 'shift':
        return windows, [one(w) for w in windows]
    # PERF: the homography costs about 11 ms per tile on one core, so tiles run in threads (OpenCV releases the
    # GIL); about 3 frames per second at 4K on a desktop CPU. A GPU version is needed before onboard use.
    with ThreadPoolExecutor(CPU_TILE_WORKERS) as pool:
        return windows, list(pool.map(one, windows))


def predict_tiles(model, windows, inputs, threshold: float, *, device='0', batch_size: int = 8) -> list[dict]:
    """Detections of one frame in frame coordinates, most confident first, centres within 24 px merged."""
    results = []
    for start in range(0, len(inputs), batch_size):
        results += model.predict(inputs[start:start + batch_size], imgsz=TILE, conf=threshold, iou=0.5, max_det=40,
                                 device=device, verbose=False)
    records = []
    for (ox, oy), result in zip(windows, results, strict=True):
        for box in result.boxes:
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
            records.append({'x': ox + (x0 + x1) / 2, 'y': oy + (y0 + y1) / 2,
                            'xyxy': [ox + x0, oy + y0, ox + x1, oy + y1], 'confidence': float(box.conf[0])})
    merged = []
    for item in sorted(records, key=lambda r: -r['confidence']):
        if all(math.dist((item['x'], item['y']), (prior['x'], prior['y'])) > 24 for prior in merged):
            merged.append(item)
    return merged
