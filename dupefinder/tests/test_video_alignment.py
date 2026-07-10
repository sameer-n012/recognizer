import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from match_video_pairs import best_diagonal_overlap  # noqa: E402


def make_offset_similarity_matrix(
    n: int, m: int, offset: int, overlap_len: int, high: float = 0.95, low: float = 0.1
) -> np.ndarray:
    """Simulates video A (n frames) and video B (m frames) that share `overlap_len`
    matching frames starting at (i=0, j=offset) — i.e. B is A trimmed/offset by
    `offset` frames, like a cut source video vs. a shifted subclip."""
    rng = np.random.default_rng(0)
    sim = rng.uniform(0.0, low, size=(n, m))
    i_start = max(0, -offset)
    for k in range(overlap_len):
        i = i_start + k
        j = i + offset
        if 0 <= i < n and 0 <= j < m:
            sim[i, j] = high
    return sim


class TestVideoAlignment(unittest.TestCase):
    def test_recovers_zero_offset_full_overlap(self):
        sim = make_offset_similarity_matrix(n=10, m=10, offset=0, overlap_len=10)
        best = best_diagonal_overlap(sim, align_threshold=0.8)
        self.assertEqual(best["offset"], 0)
        self.assertEqual(best["match_count"], 10)

    def test_recovers_positive_offset_trim(self):
        # B is a trim that starts 4 frames into A: A[i] matches B[i-4] for the
        # overlapping region -> best offset (j - i) should be -4.
        sim = make_offset_similarity_matrix(n=20, m=10, offset=-4, overlap_len=10)
        best = best_diagonal_overlap(sim, align_threshold=0.8)
        self.assertEqual(best["offset"], -4)
        self.assertEqual(best["match_count"], 10)

    def test_recovers_negative_offset_trim(self):
        # B starts 3 frames before A's content (superset case).
        sim = make_offset_similarity_matrix(n=10, m=20, offset=3, overlap_len=10)
        best = best_diagonal_overlap(sim, align_threshold=0.8)
        self.assertEqual(best["offset"], 3)
        self.assertEqual(best["match_count"], 10)

    def test_no_overlap_returns_low_match_count(self):
        rng = np.random.default_rng(1)
        sim = rng.uniform(0.0, 0.2, size=(10, 10))
        best = best_diagonal_overlap(sim, align_threshold=0.8)
        self.assertEqual(best["match_count"], 0)

    def test_partial_overlap_shorter_than_either_video(self):
        # A 30-frame and a 25-frame video that only share a 6-frame overlap somewhere
        # in the middle — the realistic "two distinct videos of the same short event"
        # case, not a pure trim of one into the other.
        sim = make_offset_similarity_matrix(n=30, m=25, offset=10, overlap_len=6)
        best = best_diagonal_overlap(sim, align_threshold=0.8)
        self.assertEqual(best["offset"], 10)
        self.assertEqual(best["match_count"], 6)
        overlap_ratio = best["match_count"] / min(30, 25)
        self.assertAlmostEqual(overlap_ratio, 6 / 25)


if __name__ == "__main__":
    unittest.main()
