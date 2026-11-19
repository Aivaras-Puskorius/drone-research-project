import unittest

import numpy as np

from drone_model.evaluate import colour_only, to_original


class ToOriginal(unittest.TestCase):
    def test_factor_two_halves_every_coordinate(self):
        shift = [4.0, -2.0]
        found = [{'x': 200.0, 'y': 100.0, 'xyxy': [190.0, 90.0, 210.0, 110.0], 'confidence': 0.7}]
        shift_back, found_back = to_original(shift, found, 2.0)
        self.assertEqual(shift_back, [2.0, -1.0])
        self.assertEqual(found_back, [{'x': 100.0, 'y': 50.0, 'xyxy': [95.0, 45.0, 105.0, 55.0], 'confidence': 0.7}])

    def test_factor_one_changes_nothing(self):
        shift = [1.5, 0.5]
        found = [{'x': 3.0, 'y': 4.0, 'xyxy': [1.0, 2.0, 5.0, 6.0], 'confidence': 0.2}]
        self.assertEqual(to_original(shift, found, 1.0), (shift, found))

    def test_inputs_are_not_modified(self):
        found = [{'x': 10.0, 'y': 10.0, 'xyxy': [8.0, 8.0, 12.0, 12.0], 'confidence': 0.5}]
        to_original([1.0, 1.0], found, 2.0)
        self.assertEqual(found[0]['x'], 10.0)


class ColourOnly(unittest.TestCase):
    def setUp(self):
        # Channel c of every tile holds the value 10 * (c + 1): B 10, G 20, R 30, f-2 40, f-4 50.
        self.tile = np.stack([np.full((4, 4), 10 * (c + 1), np.uint8) for c in range(5)], axis=2)

    def test_array_list_keeps_bgr(self):
        windows, inputs = colour_only(([(0, 0), (512, 0)], [self.tile, self.tile]))
        self.assertEqual(windows, [(0, 0), (512, 0)])
        self.assertEqual([a.shape for a in inputs], [(4, 4, 3), (4, 4, 3)])
        self.assertEqual(inputs[0][0, 0].tolist(), [10, 20, 30])

    def test_tensor_becomes_rgb(self):
        batch = np.stack([self.tile.transpose(2, 0, 1)] * 2)  # N x 5 x H x W, as the GPU path builds it
        _, inputs = colour_only(([(0, 0), (512, 0)], batch))
        self.assertEqual(inputs.shape, (2, 3, 4, 4))
        self.assertEqual(inputs[0, :, 0, 0].tolist(), [30, 20, 10])


if __name__ == '__main__':
    unittest.main()
