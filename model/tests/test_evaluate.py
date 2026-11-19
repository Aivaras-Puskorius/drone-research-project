import unittest

from drone_model.evaluate import to_original


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


if __name__ == '__main__':
    unittest.main()
