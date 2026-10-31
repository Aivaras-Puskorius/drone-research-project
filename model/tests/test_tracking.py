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


def drone_and_car(frames=21, start_car=6):
    """A drone tracked from frame 0; from `start_car` a stronger detection far away (a car) appears every frame."""
    rows = []
    for f in range(frames):
        found = [proposal(100 + 2 * f, 100, 0.8 if f < start_car else 0.6)]
        if f >= start_car:
            found.append(proposal(500, 400, 0.9))
        rows.append({'frame': f, 'proposals': found, 'background_shift': [0, 0]})
    return rows


class GatedAssociation(unittest.TestCase):
    def test_default_is_top1(self):
        rows = drone_and_car()
        self.assertEqual(replay_window(rows, FPS, 640, 480, threshold=0.5),
                         replay_window(rows, FPS, 640, 480, threshold=0.5, association='top1'))

    def test_top1_loses_the_drone_to_a_stronger_far_detection(self):
        frames, _ = replay_window(drone_and_car(), FPS, 640, 480, threshold=0.5, association='top1')
        self.assertLess(abs(frames[-1]['point'][0] - 500), 3)

    def test_gated_keeps_the_established_track_on_the_drone(self):
        frames, events = replay_window(drone_and_car(), FPS, 640, 480, threshold=0.5, association='gated')
        self.assertTrue(all(f['box'] is not None for f in frames[2:]))
        self.assertLess(abs(frames[-1]['point'][0] - 140), 3)
        self.assertEqual(len(events), 1)

    def test_gated_without_a_track_takes_the_most_confident(self):
        rows = [{'frame': 0, 'proposals': [proposal(100, 100, 0.6), proposal(300, 300, 0.9)], 'background_shift': [0, 0]}]
        frames, _ = replay_window(rows, FPS, 640, 480, threshold=0.5, association='gated')
        self.assertEqual(frames[0]['detection'], [300, 300])

    def test_gated_leaves_tentative_tracks_alone(self):
        # Two hits on a weak detection, then a stronger one far away: as with top1, the tentative track restarts there.
        rows = [{'frame': f, 'proposals': [proposal(100 + 2 * f, 100, 0.6)] + ([proposal(400, 300, 0.9)] if f == 2 else []),
                 'background_shift': [0, 0]} for f in range(3)]
        gated, _ = replay_window(rows, FPS, 640, 480, threshold=0.5, association='gated')
        top1, _ = replay_window(rows, FPS, 640, 480, threshold=0.5, association='top1')
        self.assertEqual(gated, top1)
        self.assertEqual(gated[2]['detection'], [400, 300])

    def test_unknown_association_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_window(drone_and_car(3), FPS, 640, 480, threshold=0.5, association='nearest')


if __name__ == '__main__':
    unittest.main()
