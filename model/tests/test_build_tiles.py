import unittest

import numpy as np

from drone_model.build_tiles import stands_out, tile_labels


def tile_with_drone(background_clutter: int):
    """A 5-channel tile with a 6 px drone that moved 20 px between f-2 and f, on a flat background."""
    rng = np.random.default_rng(0)
    gray = np.full((640, 640), 100, np.uint8) + rng.integers(0, 3, (640, 640), dtype=np.uint8)
    current, past = gray.copy(), gray.copy()
    current[317:323, 317:323] = 150
    past[317:323, 297:303] = 150
    if background_clutter:
        # Strong motion elsewhere: 1% of pixels change by much more than the drone does. The drone's
        # surroundings are kept clean, so only the background gets brighter.
        clean = current.copy()
        spots = rng.choice(640 * 640, 640 * 640 // 100, replace=False)
        flat = current.reshape(-1)
        flat[spots] = np.clip(flat[spots].astype(int) + background_clutter, 0, 255)
        current[290:350, 280:350] = clean[290:350, 280:350]
    return np.dstack([current, current, current, past, past])


class StandOut(unittest.TestCase):
    label = ['0 0.500000 0.500000 0.009375 0.009375']

    def test_drone_on_quiet_background_stands_out(self):
        self.assertTrue(stands_out(tile_with_drone(0), self.label))

    def test_drone_under_brighter_clutter_does_not(self):
        self.assertFalse(stands_out(tile_with_drone(150), self.label))


class TileLabels(unittest.TestCase):
    def test_box_clipped_to_tile_and_centre_inside(self):
        boxes = [[600.0, 100.0, 660.0, 120.0], [700.0, 100.0, 720.0, 120.0]]
        self.assertEqual(tile_labels(boxes, 0, 0), ['0 0.968750 0.171875 0.062500 0.031250'])


if __name__ == '__main__':
    unittest.main()
