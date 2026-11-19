#!/usr/bin/env python3
"""Colour-only copy of a tile set, for training the standard single-frame YOLO26n: python -m drone_model.colour_tiles.

Each 5-channel tile (a 5-page TIFF: B, G, R of frame f, aligned f-2, aligned f-4, as build_tiles writes it) becomes a
3-channel PNG of its first three pages, written by OpenCV, so the file holds ordinary colour. Labels are copied
unchanged, tile names keep their stems, and dataset.yaml says channels: 3. Writes <out>_sha256.txt, a SHA-256 listing
of every file under images/ and labels/ in the format of `sha256sum`, sorted by path.
Score such a model with `python -m drone_model.evaluate --channels 3`.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import cv2
import numpy as np

from drone_model.inputs import CHANNELS, TILE


def convert(source: Path, out: Path) -> int:
    """Write the colour-only copy of `source` to `out` (which must not exist); returns the number of tiles."""
    if out.exists():
        raise FileExistsError(out)
    count = 0
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True)
        shutil.copytree(source / 'labels' / split, out / 'labels' / split)
        for tiff in sorted((source / 'images' / split).glob('*.tiff')):
            okay, pages = cv2.imreadmulti(str(tiff), flags=cv2.IMREAD_UNCHANGED)
            if not okay or len(pages) != CHANNELS or pages[0].shape != (TILE, TILE):
                raise ValueError(f'{tiff}: expected {CHANNELS} pages of {TILE} x {TILE}')
            if not cv2.imwrite(str(out / 'images' / split / f'{tiff.stem}.png'), np.dstack(pages[:3])):
                raise OSError(f'could not write {tiff.stem}.png')
            count += 1
        images = {p.stem for p in (out / 'images' / split).iterdir()}
        if images != {p.stem for p in (source / 'images' / split).iterdir()}:
            raise ValueError(f'{split}: tiles other than TIFF in {source}')
        if not {p.stem for p in (out / 'labels' / split).iterdir()} <= images:
            raise ValueError(f'{split}: labels without a tile')
    (out / 'dataset.yaml').write_text(f'path: {out.resolve()}\ntrain: images/train\nval: images/val\nchannels: 3\nnames:\n  0: drone\n')
    files = sorted(str(p.relative_to(out)) for folder in ('images', 'labels') for p in (out / folder).rglob('*') if p.is_file())
    Path(f'{out}_sha256.txt').write_text(''.join(f'{hashlib.sha256((out / f).read_bytes()).hexdigest()}  {f}\n' for f in files))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='tile set written by build_tiles')
    parser.add_argument('out', type=Path, help='new folder for the colour-only copy')
    args = parser.parse_args()
    print(convert(args.source, args.out), 'tiles written to', args.out)


if __name__ == '__main__':
    main()
