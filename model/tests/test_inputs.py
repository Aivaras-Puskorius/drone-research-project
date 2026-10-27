import unittest

import cv2
import numpy as np

from drone_model.alignment import coarse_shift, shift_tile
from drone_model.inputs import TILE, compose_tile, positions, tile_windows


def texture(height, width, seed=0):
    """Smooth random gray texture, so phase correlation has something to lock on to."""
    noise = np.random.default_rng(seed).integers(0, 256, (height, width)).astype(np.float32)
    return cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def interior_difference(a, b, margin=48):
    return float(np.abs(a[margin:-margin, margin:-margin].astype(int) - b[margin:-margin, margin:-margin]).mean())


class TilePositions(unittest.TestCase):
    def test_4k_frame_has_32_tiles(self):
        self.assertEqual(positions(3840), [0, 512, 1024, 1536, 2048, 2560, 3072, 3200])
        self.assertEqual(positions(2160), [0, 512, 1024, 1520])
        self.assertEqual(len(tile_windows(3840, 2160)), 32)

    def test_small_frame_has_one_row(self):
        self.assertEqual(tile_windows(856, 480), [(0, 0), (216, 0)])


class ShiftAlignment(unittest.TestCase):
    def setUp(self):
        self.big = texture(TILE + 100, TILE + 100)
        self.current = self.big[50:50 + TILE, 50:50 + TILE]

    def stack(self, dx, dy):
        past = self.big[50 + dy:50 + dy + TILE, 50 + dx:50 + dx + TILE]
        return np.dstack([past, past, self.current])

    def test_small_shift_is_undone(self):
        stack = self.stack(5, -3)
        self.assertGreater(interior_difference(stack[:, :, 1], self.current), 10)
        aligned = shift_tile(stack)
        self.assertLess(interior_difference(aligned[:, :, 1], self.current), 1)
        self.assertTrue(np.array_equal(aligned[:, :, 2], self.current))

    def test_shift_over_32_px_is_left_alone(self):
        stack = self.stack(40, 0)
        self.assertTrue(np.array_equal(shift_tile(stack), stack))

    def test_compose_tile_channel_order(self):
        color = np.dstack([self.current, self.current // 2, self.current // 3])
        stack = np.dstack([self.big[50:50 + TILE, 54:54 + TILE], self.big[50:50 + TILE, 52:52 + TILE], self.current])
        tile = compose_tile(color, stack)
        self.assertEqual(tile.shape, (TILE, TILE, 5))
        self.assertTrue(np.array_equal(tile[:, :, :3], color))
        # Channel 3 is f-2, channel 4 is f-4, both aligned to f.
        self.assertLess(interior_difference(tile[:, :, 3], self.current), 1)
        self.assertLess(interior_difference(tile[:, :, 4], self.current), 1)


class CoarseShift(unittest.TestCase):
    def test_large_whole_frame_shift_is_undone(self):
        big = texture(1080 + 300, 1920 + 300, seed=1)
        current = big[150:150 + 1080, 150:150 + 1920]
        past = big[150 + 20:150 + 20 + 1080, 150 + 100:150 + 100 + 1920]
        shifted = coarse_shift(np.dstack([past, past, current]))
        self.assertGreater(interior_difference(past, current, 200), 10)
        self.assertLess(interior_difference(shifted[:, :, 0], current, 200), 3)


if __name__ == '__main__':
    unittest.main()
