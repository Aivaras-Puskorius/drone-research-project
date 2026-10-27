"""Score tracked boxes against a window's labels.

A window plan lists, for every frame, the labelled drone boxes as [x, y, w, h] in frame pixels: an empty list
for a frame labelled as having no drone, and null for a frame without a label, which is skipped. A frame
counts as tracked within d px when the tracked point of a shown box lies within d px of a labelled box centre,
and as far when it lies more than d px from every centre. Our own 4K flights are read at 50 and 75 px, the
public sets at 20 and 50 px, because our labels are projected from flight logs.
"""
from __future__ import annotations

import math

COUNTS = ('drone_frames', 'tracked20', 'tracked50', 'far50', 'far75', 'empty_frames', 'boxes_on_empty',
          'unknown_frames', 'first_correct20', 'first_correct50')


def _centre(x: float, y: float, w: float, h: float) -> list[float]:
    box = [x, y, x + w, y + h]
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def score_window(window: dict, frames: list[dict], events: list[dict]) -> dict:
    """Counts for one window, plus each frame's distance from the tracked point to the nearest drone.

    first_correct20 is true when the first track of the window started within 20 px of a drone. first_correct50
    looks at the first track started on a frame showing a drone, and is true when it was within 50 px.
    """
    references = window['references']
    if [f['frame'] for f in frames] != [r['source_frame_zero_based'] for r in references] or \
            len(frames) != window['count'] or (frames and frames[0]['frame'] != window['start']):
        raise ValueError(f"{window['key']}: tracked frames do not match the labelled frames")
    result: dict = dict.fromkeys(COUNTS, 0)
    distances, visibility = {}, {}
    for frame, reference in zip(frames, references):
        boxes = reference['boxes_xywh']
        if boxes is None:
            result['unknown_frames'] += 1
            continue
        if not boxes:
            visibility[frame['frame']] = 'empty'
            result['empty_frames'] += 1
            result['boxes_on_empty'] += frame['box'] is not None
            continue
        visibility[frame['frame']] = 'visible'
        result['drone_frames'] += 1
        point = frame['point'] if frame['box'] is not None else None
        distance = min(math.dist(point, _centre(*b)) for b in boxes) if point else None
        distances[frame['frame']] = distance
        if distance is not None:
            result['tracked20'] += distance <= 20
            result['tracked50'] += distance <= 50
            result['far50'] += distance > 50
            result['far75'] += distance > 75
    if events:
        first = distances.get(events[0]['frame'])
        result['first_correct20'] = first is not None and first <= 20
    on_drone = next((e for e in events if visibility.get(e['frame']) == 'visible'), None)
    if on_drone is not None:
        first = distances[on_drone['frame']]
        result['first_correct50'] = first is not None and first <= 50
    result['key'] = window['key']
    result['distances'] = [distances.get(f['frame']) for f in frames]
    result['acquisitions'] = events
    return result


def summarize(results: list[dict]) -> dict:
    """Sums over windows; first_correct20 and first_correct50 become window counts."""
    return {'windows': len(results), **{name: int(sum(r[name] for r in results)) for name in COUNTS}}
