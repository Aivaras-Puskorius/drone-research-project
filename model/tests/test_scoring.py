import unittest

from drone_model.scoring import score_window, summarize


def reference(frame, boxes):
    return {'source_frame_zero_based': frame, 'boxes_xywh': boxes}


class ScoreWindow(unittest.TestCase):
    def setUp(self):
        drone = [[100, 100, 10, 10]]  # centre (105, 105)
        self.window = {'key': 'w', 'start': 0, 'count': 5,
                       'references': [reference(0, drone), reference(1, drone), reference(2, drone),
                                      reference(3, []), reference(4, None)]}
        self.frames = [{'frame': 0, 'point': None, 'box': None},
                       {'frame': 1, 'point': [110, 105], 'box': [100, 95, 120, 115]},
                       {'frame': 2, 'point': [170, 105], 'box': [160, 95, 180, 115]},
                       {'frame': 3, 'point': [170, 105], 'box': [160, 95, 180, 115]},
                       {'frame': 4, 'point': [170, 105], 'box': [160, 95, 180, 115]}]

    def test_counts(self):
        result = score_window(self.window, self.frames, [{'frame': 1, 'kind': 'initial_acquisition'}])
        self.assertEqual(result['drone_frames'], 3)
        self.assertEqual(result['tracked20'], 1)
        self.assertEqual(result['tracked50'], 1)
        self.assertEqual(result['far50'], 1)  # 65 px away
        self.assertEqual(result['far75'], 0)
        self.assertEqual(result['empty_frames'], 1)
        self.assertEqual(result['boxes_on_empty'], 1)
        self.assertEqual(result['unknown_frames'], 1)
        self.assertTrue(result['first_correct20'])
        self.assertTrue(result['first_correct50'])

    def test_first_start_far_from_the_drone(self):
        result = score_window(self.window, self.frames, [{'frame': 2, 'kind': 'initial_acquisition'},
                                                         {'frame': 1, 'kind': 'reacquisition'}])
        self.assertFalse(result['first_correct20'])
        self.assertFalse(result['first_correct50'])

    def test_first_start_on_an_empty_frame_is_not_correct(self):
        result = score_window(self.window, self.frames, [{'frame': 3, 'kind': 'initial_acquisition'}])
        self.assertFalse(result['first_correct20'])

    def test_frames_must_match_the_window(self):
        with self.assertRaises(ValueError):
            score_window(self.window, self.frames[1:], [])

    def test_summary_adds_windows(self):
        one = score_window(self.window, self.frames, [{'frame': 1, 'kind': 'initial_acquisition'}])
        total = summarize([one, one])
        self.assertEqual((total['windows'], total['tracked20'], total['first_correct20']), (2, 2, 2))


if __name__ == '__main__':
    unittest.main()
