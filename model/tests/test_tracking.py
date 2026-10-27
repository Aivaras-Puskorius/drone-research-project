import unittest

from drone_model.tracking import LOST, PREDICTED, TENTATIVE, TRACKING, Tracker, replay_window, select_candidate

FPS = 30.0


def proposal(x, y, confidence):
    return {'x': x, 'y': y, 'xyxy': [x - 5, y - 5, x + 5, y + 5], 'confidence': confidence}


class SelectCandidate(unittest.TestCase):
    def test_highest_confidence_above_threshold(self):
        rows = [proposal(10, 10, 0.3), proposal(50, 50, 0.9), proposal(90, 90, 0.6)]
        self.assertEqual(select_candidate(rows, 0.5)['x'], 50)

    def test_none_above_threshold(self):
        self.assertIsNone(select_candidate([proposal(10, 10, 0.2)], 0.25))


class TrackerStates(unittest.TestCase):
    def test_three_hits_start_a_track(self):
        tracker = Tracker(hits_required=3)
        states = [tracker.step(f / FPS, (100 + 2 * f, 100), (0, 0)).state for f in range(3)]
        self.assertEqual(states, [TENTATIVE, TENTATIVE, TRACKING])

    def test_track_coasts_then_is_lost(self):
        tracker = Tracker(hits_required=3)
        for f in range(5):
            tracker.step(f / FPS, (100 + 2 * f, 100), (0, 0))
        states = [tracker.step(f / FPS, None, (0, 0)).state for f in range(5, 17)]
        # The last detection was at frame 4; 0.35 s is 10.5 frames at 30 fps.
        self.assertEqual(states[:10], [PREDICTED] * 10)
        self.assertEqual(states[10:], [LOST, LOST])

    def test_far_detection_is_not_used_by_a_track(self):
        tracker = Tracker(hits_required=3)
        for f in range(5):
            tracker.step(f / FPS, (100 + 2 * f, 100), (0, 0))
        result = tracker.step(5 / FPS, (600, 600), (0, 0))
        self.assertEqual(result.state, PREDICTED)
        self.assertFalse(result.used)
        self.assertLess(abs(result.point[0] - 110), 3)


class ReplayWindow(unittest.TestCase):
    def test_boxes_and_first_acquisition(self):
        rows = [{'frame': 10 + f, 'proposals': [proposal(100 + 2 * f, 100, 0.8)], 'background_shift': [0, 0]}
                for f in range(6)]
        frames, events = replay_window(rows, FPS, 640, 480, threshold=0.5, hits_required=3)
        self.assertEqual([f['box'] is not None for f in frames], [False, False, True, True, True, True])
        self.assertEqual(events, [{'frame': 12, 'kind': 'initial_acquisition'}])

    def test_point_outside_frame_shows_no_box(self):
        rows = [{'frame': f, 'proposals': [proposal(630 + 4 * f, 100, 0.8)], 'background_shift': [0, 0]}
                for f in range(6)]
        frames, _ = replay_window(rows, FPS, 640, 480, threshold=0.5, hits_required=3)
        self.assertIsNotNone(frames[2]['box'])
        self.assertIsNone(frames[5]['box'])


if __name__ == '__main__':
    unittest.main()
