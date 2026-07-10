import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

import hashing  # noqa: E402


def make_gradient_image(size: int = 256) -> np.ndarray:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    row = np.tile(x, (size, 1))
    img = np.stack([row, row.T, (row + row.T) // 2], axis=-1).astype(np.uint8)
    return img


def make_checkerboard_image(size: int = 256, block: int = 16) -> np.ndarray:
    tile = np.indices((size, size)).sum(axis=0) // block % 2
    img = (tile * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


class TestHashingInvariance(unittest.TestCase):
    def test_resize_invariance_gradient(self):
        img = make_gradient_image(256)
        resized = cv2.resize(img, (128, 128))

        h1 = hashing.compute_hashes(img)
        h2 = hashing.compute_hashes(resized)

        for variant in ("ahash", "dhash", "phash"):
            distance = hashing.hamming_distance(h1[variant], h2[variant])
            self.assertLessEqual(
                distance, 4, f"{variant} distance {distance} too high after resize"
            )

    def test_format_recompression_invariance(self):
        img = make_gradient_image(256)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        self.assertTrue(ok)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        h1 = hashing.compute_hashes(img)
        h2 = hashing.compute_hashes(recompressed)

        matches = [
            hashing.hamming_distance(h1[v], h2[v]) <= 6 for v in ("ahash", "dhash", "phash")
        ]
        self.assertTrue(any(matches), "no hash variant survived JPEG recompression")

    def test_distinct_images_have_high_distance(self):
        img_a = make_gradient_image(256)
        img_b = make_checkerboard_image(256)

        h1 = hashing.compute_hashes(img_a)
        h2 = hashing.compute_hashes(img_b)

        for variant in ("ahash", "dhash", "phash"):
            distance = hashing.hamming_distance(h1[variant], h2[variant])
            self.assertGreater(
                distance, 8, f"{variant} distance {distance} too low for distinct images"
            )

    def test_hamming_distance_symmetry_and_identity(self):
        img = make_gradient_image(64)
        h = hashing.compute_hashes(img)
        self.assertEqual(hashing.hamming_distance(h["phash"], h["phash"]), 0)
        self.assertEqual(
            hashing.hamming_distance(h["ahash"], h["dhash"]),
            hashing.hamming_distance(h["dhash"], h["ahash"]),
        )


if __name__ == "__main__":
    unittest.main()
