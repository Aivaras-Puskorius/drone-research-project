from pathlib import Path
import tempfile
import unittest

from drone_model.make_plans import ard100, labels_digest, nps, uav_anti_uav

VOC = """<annotation><size><width>1920</width><height>1080</height></size>
<object><name>Drone</name><bndbox><xmin>101</xmin><ymin>51</ymin><xmax>111</xmax><ymax>58</ymax></bndbox></object>
<object><name>bird</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


class LabelReaders(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_ard100_box_corner_minus_one_and_missing_frame(self):
        (self.root / 'clip').mkdir()
        (self.root / 'clip/clip_0005.xml').write_text(VOC)
        window = {'clip': 'clip', 'start': 4, 'count': 2, 'dimensions': [1920, 1080]}
        references, entries = ard100(window, self.root)
        self.assertEqual(references[0]['boxes_xywh'], [[100.0, 50.0, 10.0, 7.0]])
        self.assertIsNone(references[1]['boxes_xywh'])
        self.assertEqual([name for name, _ in entries], ['clip/clip_0005.xml'])

    def test_nps_drops_box_centred_outside_and_skips_unlisted_frames(self):
        (self.root / 'Clip_1_refined.txt').write_text('5,1,100,100,10,10\n6,1,1915,100,20,10\n')
        window = {'clip': 'Clip_1', 'start': 4, 'count': 3, 'dimensions': [1920, 1080]}
        references, _ = nps(window, self.root)
        self.assertEqual([r['boxes_xywh'] for r in references], [[[100.0, 100.0, 10.0, 10.0]], None, None])

    def test_uav_absent_frame_is_empty_and_bad_box_is_skipped(self):
        (self.root / 'v').mkdir()
        (self.root / 'v/groundtruth_rect.txt').write_text('1,2,3,4\n0,0,0,0\n5,6,0,8\n')
        (self.root / 'v/absent.txt').write_text('0\n1\n0\n')
        window = {'clip': 'v', 'start': 0, 'count': 3}
        references, entries = uav_anti_uav(window, self.root)
        self.assertEqual([r['boxes_xywh'] for r in references], [[[1.0, 2.0, 3.0, 4.0]], [], None])
        self.assertEqual(len(entries), 2)

    def test_digest_ignores_order_and_repeats(self):
        self.assertEqual(labels_digest([('b', '2'), ('a', '1'), ('a', '1')]), labels_digest([('a', '1'), ('b', '2')]))


if __name__ == '__main__':
    unittest.main()
