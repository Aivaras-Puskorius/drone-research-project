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


class PastedCars(unittest.TestCase):
    def test_paste_blends_by_alpha(self):
        from drone_model.build_tiles import paste
        image = np.full((20, 20, 3), 100, np.uint8)
        patch = np.zeros((4, 4, 4), np.uint8)
        patch[:, :, :3] = 200
        patch[1:3, 1:3, 3] = 255
        paste(image, patch, 10, 10)
        self.assertEqual(image[9, 9].tolist(), [200, 200, 200])
        self.assertEqual(image[8, 8].tolist(), [100, 100, 100])

    def test_cars_stay_clear_of_drones_and_inside_the_frame(self):
        import random
        from drone_model.build_tiles import add_cars
        rng = random.Random(1)
        car = np.zeros((30, 50, 4), np.uint8)
        car[:, :, :3], car[:, :, 3] = 60, 255
        color = np.full((1080, 1920, 3), 120, np.uint8)
        stack = np.full((1080, 1920, 3), 120, np.uint8)
        drone = [900.0, 500.0, 930.0, 520.0]
        for _ in range(20):
            new_color, new_stack, centres = add_cars(color, stack, [drone], rng, [car])
            self.assertTrue(np.array_equal(color, np.full_like(color, 120)))  # inputs are not changed
            for x, y in centres:
                self.assertTrue(40 <= x < 1920 - 40 and 40 <= y < 1080 - 40)
                self.assertTrue(x <= drone[0] - 92 or x >= drone[2] + 92 or y <= drone[1] - 92 or y >= drone[3] + 92)
            if centres:
                self.assertFalse(np.array_equal(new_stack[:, :, 0], new_stack[:, :, 2]))  # the car moved between f-4 and f



class RenderedCarTiles(unittest.TestCase):
    def test_tile_holds_the_car_and_no_drone(self):
        import random
        from drone_model.build_tiles import TILE, car_tile_place
        rng = random.Random(3)
        car, drone = [1000.0, 800.0, 1080.0, 840.0], [1500.0, 820.0, 1530.0, 850.0]
        for _ in range(50):
            ox, oy = car_tile_place(rng, [car], [drone], 3840, 2160)
            self.assertTrue(ox <= 1040 < ox + TILE and oy <= 820 < oy + TILE)
            self.assertTrue(1500 >= ox + TILE or 1530 <= ox or 820 >= oy + TILE or 850 <= oy)

    def test_no_car_or_no_clear_place_gives_none(self):
        import random
        from drone_model.build_tiles import car_tile_place
        rng = random.Random(3)
        self.assertIsNone(car_tile_place(rng, [], [], 3840, 2160))
        everywhere = [0.0, 0.0, 3840.0, 2160.0]
        self.assertIsNone(car_tile_place(rng, [[1000.0, 800.0, 1080.0, 840.0]], [everywhere], 3840, 2160))


class LongerPastGap(unittest.TestCase):
    def test_oldest_frame_is_clamped_to_the_clip_start(self):
        from drone_model.build_tiles import past_stack
        grays = {f: np.full((4, 4), f, np.uint8) for f in range(0, 40)}
        self.assertEqual(past_stack(grays, 30, 16, 0)[0, 0].tolist(), [14, 28, 30])
        self.assertEqual(past_stack(grays, 10, 16, 0)[0, 0].tolist(), [0, 8, 10])
        self.assertEqual(past_stack(grays, 10, 4, 0)[0, 0].tolist(), [6, 8, 10])

    def test_cars_keep_their_places_and_move_further_in_the_oldest_frame(self):
        import random
        from drone_model.build_tiles import add_cars
        car = np.zeros((30, 50, 4), np.uint8)
        car[:, :, :3], car[:, :, 3] = 60, 255
        color = np.full((1080, 1920, 3), 120, np.uint8)
        stack = np.full((1080, 1920, 3), 120, np.uint8)
        _, short, centres_short = add_cars(color, stack, [], random.Random(5), [car])
        _, long, centres_long = add_cars(color, stack, [], random.Random(5), [car], oldest=16)
        self.assertEqual(centres_short, centres_long)
        self.assertTrue(np.array_equal(short[:, :, 1:], long[:, :, 1:]))
        self.assertFalse(np.array_equal(short[:, :, 0], long[:, :, 0]))



class KeepEvery(unittest.TestCase):
    def test_parses_scene_steps_and_keeps_every_kth_sampled_frame(self):
        from drone_model.build_tiles import keep_steps, kept
        steps = keep_steps(['r3_move_grey=4'])
        self.assertEqual(steps, {'r3_move_grey': 4})
        self.assertEqual([i for i in range(10) if kept(steps, 'r3_move_grey', i)], [0, 4, 8])
        self.assertTrue(all(kept(steps, 'centras1', i) for i in range(10)))

    def test_rejects_unknown_scenes_and_bad_steps(self):
        from drone_model.build_tiles import keep_steps
        with self.assertRaises(ValueError):
            keep_steps(['nowhere=2'])
        with self.assertRaises(ValueError):
            keep_steps(['r3_move_grey=0'])


class ScreenFixedText(unittest.TestCase):
    """--text-overlays: on-screen text drawn at the same pixels in f and in every past frame, away from the drones."""

    def setUp(self):
        import random
        rng = np.random.default_rng(0)
        self.color = rng.integers(60, 120, (720, 1280, 3), dtype=np.uint8)
        self.stack = rng.integers(60, 120, (720, 1280, 3), dtype=np.uint8)
        self.boxes = [[600.0, 300.0, 640.0, 330.0]]
        self.rng = random.Random(5)

    def test_same_pixels_in_every_frame(self):
        from drone_model.build_tiles import add_text
        color, stack, places = add_text(self.color, self.stack, self.boxes, self.rng)
        self.assertTrue(places)
        changed = [stack[:, :, c] != self.stack[:, :, c] for c in range(3)]
        self.assertTrue(changed[0].any())
        self.assertTrue((changed[0] == changed[1]).all() and (changed[1] == changed[2]).all())
        self.assertTrue((stack[:, :, 0][changed[0]] == stack[:, :, 2][changed[0]]).all())
        self.assertTrue(((color != self.color).any(axis=2) == changed[2]).all())

    def test_clear_of_the_drones(self):
        from drone_model.build_tiles import add_text
        color, stack, _ = add_text(self.color, self.stack, self.boxes, self.rng)
        x0, y0, x1, y1 = (int(v) for v in self.boxes[0])
        near = (slice(y0 - 32, y1 + 32), slice(x0 - 32, x1 + 32))
        self.assertTrue((color[near] == self.color[near]).all())
        self.assertTrue((stack[near] == self.stack[near]).all())

    def test_inputs_unchanged(self):
        from drone_model.build_tiles import add_text
        before_color, before_stack = self.color.copy(), self.stack.copy()
        add_text(self.color, self.stack, self.boxes, self.rng)
        self.assertTrue((self.color == before_color).all() and (self.stack == before_stack).all())


if __name__ == '__main__':
    unittest.main()
