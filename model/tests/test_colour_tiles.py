import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from drone_model.colour_tiles import convert


class Convert(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.source = self.root / 'tiles'
        for split in ('train', 'val'):
            (self.source / 'images' / split).mkdir(parents=True)
            (self.source / 'labels' / split).mkdir(parents=True)
        pages = [np.full((640, 640), 10 * (c + 1), np.uint8) for c in range(5)]
        pages[2][0, 0] = 99
        cv2.imwritemulti(str(self.source / 'images/train/a.tiff'), pages)
        cv2.imwritemulti(str(self.source / 'images/val/b.tiff'), pages)
        (self.source / 'labels/train/a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
        (self.source / 'labels/val/b.txt').write_text('')

    def test_first_three_pages_become_a_colour_png(self):
        out = self.root / 'colour'
        self.assertEqual(convert(self.source, out), 2)
        image = cv2.imread(str(out / 'images/train/a.png'), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(image[1, 1].tolist(), [10, 20, 30])
        self.assertEqual(image[0, 0].tolist(), [10, 20, 99])

    def test_labels_yaml_and_checksums(self):
        out = self.root / 'colour'
        convert(self.source, out)
        self.assertEqual((out / 'labels/train/a.txt').read_text(), '0 0.5 0.5 0.1 0.1\n')
        self.assertIn('channels: 3', (out / 'dataset.yaml').read_text())
        listing = Path(f'{out}_sha256.txt').read_text().splitlines()
        self.assertEqual([line.split('  ')[1] for line in listing],
                         ['images/train/a.png', 'images/val/b.png', 'labels/train/a.txt', 'labels/val/b.txt'])

    def test_existing_output_is_refused(self):
        out = self.root / 'colour'
        out.mkdir()
        with self.assertRaises(FileExistsError):
            convert(self.source, out)


if __name__ == '__main__':
    unittest.main()
