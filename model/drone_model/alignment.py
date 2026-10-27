"""Aligning the past gray frames to the current one, so that the background cancels in the motion channels.

A gray stack is H x W x 3 uint8 with the channels f-4, f-2, f (oldest first, current frame last).

- `coarse_shift` moves each whole past frame by one translation of up to 256 px, measured by phase
  correlation on a copy at most 960 px wide. It runs first at test time, so fast camera motion does not
  exceed the per-tile limit.
- `shift_tile` moves each past frame of one 640 px tile by its own translation of up to 32 px, measured by
  phase correlation. It is used in training and at test time. `shift_tiles_gpu` does the same for all tiles
  of a frame at once on the GPU.
- `homography_tile` warps each past frame of one tile with a homography fitted to tracked corners, and falls
  back to `shift_tile` where no homography fits. It is an optional test-time alignment and runs on the CPU.

A shift is only used when the phase-correlation response is at least 0.1 and the shift is within its limit;
otherwise that past frame stays as it was. The current frame is never changed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CURRENT, PAST = 2, (0, 1)
MIN_RESPONSE = 0.1
MAX_TILE_SHIFT_PX = 32.0
COARSE_MAX_SHIFT_PX, COARSE_WIDTH = 256.0, 960
MIN_INLIERS, MIN_INLIER_SHARE, MAX_CORNER_MOVE_PX = 20, 0.3, 64.0
_HANN = {}


def _check_stack(stack: np.ndarray) -> None:
    if not isinstance(stack, np.ndarray) or stack.dtype != np.uint8 or stack.ndim != 3 or stack.shape[2] != 3:
        raise ValueError('expected a uint8 H x W x 3 gray stack f-4, f-2, f')


def coarse_shift(stack: np.ndarray) -> np.ndarray:
    """Shift each whole past frame by one translation of up to 256 px."""
    _check_stack(stack)
    height, width, channels = stack.shape
    scale = min(1.0, COARSE_WIDTH / width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = [cv2.resize(stack[:, :, c], size, interpolation=cv2.INTER_AREA).astype(np.float32) for c in range(channels)]
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    output = stack.copy()
    for channel in PAST:
        (dx, dy), response = cv2.phaseCorrelate(small[channel], small[CURRENT], window)
        dx, dy = dx / scale, dy / scale
        if all(map(math.isfinite, (dx, dy, response))) and response >= MIN_RESPONSE and math.hypot(dx, dy) <= COARSE_MAX_SHIFT_PX:
            output[:, :, channel] = cv2.warpAffine(stack[:, :, channel], np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                                                   (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return output


def shift_tile(stack: np.ndarray) -> np.ndarray:
    """Shift each past frame of one tile by its own translation of up to 32 px."""
    _check_stack(stack)
    height, width = stack.shape[:2]
    output = np.array(stack, copy=True, order='C')
    current = stack[:, :, CURRENT]
    for channel in PAST:
        past = stack[:, :, channel]
        try:
            # phaseCorrelate may modify its float inputs, so each call gets fresh copies.
            (dx, dy), response = cv2.phaseCorrelate(np.ascontiguousarray(past, dtype=np.float32),
                                                    np.ascontiguousarray(current, dtype=np.float32),
                                                    cv2.createHanningWindow((width, height), cv2.CV_32F))
        except cv2.error:
            continue
        dx, dy, response = float(dx), float(dy), float(response)
        if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(response)):
            continue
        if response < MIN_RESPONSE or float(np.hypot(dx, dy)) > MAX_TILE_SHIFT_PX:
            continue
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        output[:, :, channel] = cv2.warpAffine(past, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REFLECT_101)
    return output


def _homography(past: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    """`past` warped onto `current` by a homography from tracked corners, or None when no reliable fit exists.

    Corners are tracked by pyramidal Lucas-Kanade with a 1 px forward-backward check. RANSAC at 1.5 px needs at
    least 20 inliers and a 30% inlier share, and the tile corners may move at most 64 px.
    """
    corners = cv2.goodFeaturesToTrack(past, 500, 0.01, 8, blockSize=7)
    if corners is None or len(corners) < MIN_INLIERS:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(past, current, corners, None, winSize=(21, 21), maxLevel=3)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(current, past, moved, None, winSize=(21, 21), maxLevel=3)
    good = (status.ravel() == 1) & (status_back.ravel() == 1) & (np.linalg.norm((back - corners)[:, 0], axis=1) < 1.0)
    if good.sum() < MIN_INLIERS:
        return None
    matrix, inliers = cv2.findHomography(corners[good], moved[good], cv2.RANSAC, 1.5)
    if matrix is None:
        return None
    count = int(inliers.sum())
    if count < MIN_INLIERS or count / int(good.sum()) < MIN_INLIER_SHARE:
        return None
    h, w = past.shape
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    if np.abs(cv2.perspectiveTransform(frame, matrix) - frame).max() > MAX_CORNER_MOVE_PX:
        return None
    return cv2.warpPerspective(past, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def homography_tile(stack: np.ndarray) -> np.ndarray:
    """Warp each past frame of one tile by a homography; where none fits, use the shift_tile result."""
    aligned = shift_tile(stack)
    current = np.ascontiguousarray(stack[:, :, CURRENT])
    for channel in PAST:
        warped = _homography(np.ascontiguousarray(stack[:, :, channel]), current)
        if warped is not None:
            aligned[:, :, channel] = warped
    return aligned


def _tile_shifts_gpu(tiles):
    """cv2.phaseCorrelate(past, current, Hann) for both past channels of N x 3 x 640 x 640 tiles, batched.

    Follows OpenCV's phasecorr.cpp: Hann-windowed DFTs, cross-power spectrum F_past * conj(F_current) /
    (|.| + FLT_EPSILON), inverse DFT, fftshift, first maximum, 5 x 5 weighted centroid clamped at the borders,
    response = centroid sum / (M * N), shift = centre - centroid. Returns dx, dy and response, each N x 2.
    """
    import torch
    size = tiles.shape[-1]
    if tiles.device not in _HANN:
        _HANN[tiles.device] = torch.from_numpy(cv2.createHanningWindow((size, size), cv2.CV_32F)).to(tiles.device)
    spectra = torch.fft.rfft2(tiles.float() * _HANN[tiles.device])
    product = spectra[:, list(PAST)] * spectra[:, CURRENT:CURRENT + 1].conj()
    magnitude = product.abs()
    surface = torch.fft.fftshift(torch.fft.irfft2(product * (magnitude / (magnitude * magnitude + np.finfo(np.float32).eps)),
                                                  s=(size, size)), dim=(-2, -1))
    peaks = surface.flatten(-2).argmax(-1)
    offsets = torch.arange(-2, 3, device=tiles.device)
    rows = (peaks // size)[..., None] + offsets
    cols = (peaks % size)[..., None] + offsets
    inside = ((rows >= 0) & (rows < size))[..., :, None] & ((cols >= 0) & (cols < size))[..., None, :]
    flat = rows.clamp(0, size - 1)[..., :, None] * size + cols.clamp(0, size - 1)[..., None, :]
    values = surface.flatten(-2).gather(-1, flat.flatten(-2)).view(flat.shape) * inside
    values, rows, cols = (t.cpu().numpy().astype(np.float64) for t in (values, rows, cols))
    total = values.sum((-2, -1))
    centroid_x = (values * cols[..., None, :]).sum((-2, -1)) / (total + np.finfo(np.float64).eps)
    centroid_y = (values * rows[..., :, None]).sum((-2, -1)) / (total + np.finfo(np.float64).eps)
    return size / 2 - centroid_x, size / 2 - centroid_y, total


def _warp_planes_gpu(planes, dx, dy):
    """cv2.warpAffine(plane, [[1, 0, dx], [0, 1, dy]], INTER_LINEAR, BORDER_REFLECT_101) for K uint8 planes.

    Uses the float32 bilinear arithmetic of OpenCV 5.0's warpAffine: source coordinate x - dx in float32, lerp
    along x, then along y, round half to even. On 144 frames, 158 pixel values differed from OpenCV, each by 1.
    """
    import torch
    size = planes.shape[-1]
    grid = torch.arange(size, device=planes.device, dtype=torch.float32)
    source_x, source_y = grid + (-dx)[:, None], grid + (-dy)[:, None]
    left, top = source_x.floor(), source_y.floor()
    fraction_x, fraction_y = source_x - left, source_y - top

    def reflect(index):
        index = index.long().abs()
        return torch.where(index > size - 1, 2 * (size - 1) - index, index)

    columns, rows = (reflect(left), reflect(left + 1)), (reflect(top), reflect(top + 1))
    values = planes.float()

    def pick(row, column):
        picked = values.gather(1, row[:, :, None].expand(-1, -1, size))
        return picked.gather(2, column[:, None, :].expand(-1, size, -1))

    upper_left, upper_right = pick(rows[0], columns[0]), pick(rows[0], columns[1])
    lower_left, lower_right = pick(rows[1], columns[0]), pick(rows[1], columns[1])
    upper = upper_left + fraction_x[:, None, :] * (upper_right - upper_left)
    lower = lower_left + fraction_x[:, None, :] * (lower_right - lower_left)
    blended = upper + fraction_y[:, :, None] * (lower - upper)
    return blended.round().clamp(0, 255).to(torch.uint8)


def shift_tiles_gpu(tiles):
    """shift_tile for N x 3 x 640 x 640 uint8 gray tiles on the GPU, in place; returns the tiles.

    The gates are those of shift_tile. Results can differ from the CPU path where the float32 FFTs move a shift
    across a rounding step: on 144 sampled frames, 1,754 of 2.75 billion input values differed, each by 1.
    """
    import torch
    shifts_x, shifts_y, responses = _tile_shifts_gpu(tiles)
    usable = (np.isfinite(shifts_x) & np.isfinite(shifts_y) & np.isfinite(responses)
              & (responses >= MIN_RESPONSE) & (np.hypot(shifts_x, shifts_y) <= MAX_TILE_SHIFT_PX))
    # shift_tile casts the shift to float32 for its warp matrix.
    shifts_x = np.where(usable, shifts_x, 0).astype(np.float32)
    shifts_y = np.where(usable, shifts_y, 0).astype(np.float32)
    count, size = tiles.shape[0], tiles.shape[-1]
    past = tiles[:, list(PAST)].reshape(-1, size, size)
    warped = _warp_planes_gpu(past, torch.from_numpy(shifts_x.reshape(-1)).to(tiles.device),
                              torch.from_numpy(shifts_y.reshape(-1)).to(tiles.device))
    mask = torch.from_numpy(usable.reshape(-1)).to(tiles.device)[:, None, None]
    tiles[:, list(PAST)] = torch.where(mask, warped, past).view(count, len(PAST), size, size)
    return tiles


def background_shift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Whole-frame camera translation from gray frame f-1 to f, for the tracker's prediction.

    Measured on 512 x 288 copies; (0, 0) when the response is under 0.1 or the shift exceeds 64 px.
    """
    h, w = current.shape
    (dx, dy), response = cv2.phaseCorrelate(cv2.resize(previous, (512, 288)).astype(np.float32),
                                            cv2.resize(current, (512, 288)).astype(np.float32))
    dx, dy = dx * w / 512, dy * h / 288
    if response < .1 or max(abs(dx), abs(dy)) > 64 or not all(map(math.isfinite, (dx, dy, response))):
        return 0., 0.
    return float(dx), float(dy)
