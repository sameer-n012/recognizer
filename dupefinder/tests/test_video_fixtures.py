"""Integration tests against the real sample videos in tests/videos/. These shell out
to ffmpeg (via utils.extract_frames_dense) and require it to be on PATH, same as the
rest of the pipeline — no CLIP/torch model is loaded here, so no network access or
model download is required to run this file.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VIDEOS_DIR = ROOT / "tests" / "videos"
sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

import hashing  # noqa: E402
from utils import extract_frames_dense  # noqa: E402

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg not found on PATH")
class TestSameSourceDifferentResolution(unittest.TestCase):
    """sample1-5s-360p.mp4 and sample1-5s-720p.mp4 are the same source clip encoded at
    two resolutions — this is exactly the case tier 1 (perceptual hashing) needs to
    catch without ever running CLIP.
    """

    @classmethod
    def setUpClass(cls):
        cls.low_res = VIDEOS_DIR / "sample1-5s-360p.mp4"
        cls.high_res = VIDEOS_DIR / "sample1-5s-720p.mp4"
        if not cls.low_res.exists() or not cls.high_res.exists():
            raise unittest.SkipTest("sample1 fixture videos not found")
        cls.tmpdir = tempfile.mkdtemp(prefix="dupefinder_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_frame_hashes_agree_across_resolutions(self):
        interval_sec = 1.0
        low_frames = extract_frames_dense(
            self.low_res, Path(self.tmpdir) / "low", interval_sec
        )
        high_frames = extract_frames_dense(
            self.high_res, Path(self.tmpdir) / "high", interval_sec
        )

        self.assertGreater(len(low_frames), 0)
        self.assertGreater(len(high_frames), 0)

        n = min(len(low_frames), len(high_frames))
        matched = 0
        for i in range(n):
            low_img = cv2.imread(str(low_frames[i]))
            high_img = cv2.imread(str(high_frames[i]))
            if low_img is None or high_img is None:
                continue
            h_low = hashing.compute_hashes(low_img)
            h_high = hashing.compute_hashes(high_img)
            # OR-across-variants, same design as near_duplicates.hashes_match, with a
            # slightly looser threshold to allow for resize/recompression artifacts.
            if any(
                hashing.hamming_distance(h_low[v], h_high[v]) <= 8
                for v in ("ahash", "dhash", "phash")
            ):
                matched += 1

        match_ratio = matched / n
        self.assertGreaterEqual(
            match_ratio,
            0.8,
            f"only {matched}/{n} corresponding frames hashed as near-duplicates "
            "across resolutions",
        )


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg not found on PATH")
class TestDistinctVideosHashDifferently(unittest.TestCase):
    """Sanity check in the other direction: sample3 and sample4 are different camera
    angles of the same park (not duplicates of each other, confirmed by inspection —
    an earlier, looser threshold set incorrectly merged them at the CLIP/tier-3 stage,
    see README "Threshold tuning"), so at minimum the cheap hash tier should NOT flag
    them as near-duplicates (guards against a threshold set so loose that tier 1's
    false-positive rate stops being near zero). This does not test tier 2/3 — whole-
    frame CLIP similarity between them is legitimately high (0.905) since they show the
    same kind of scene; that's tier 3's alignment search to disambiguate, not tier 1's.
    """

    @classmethod
    def setUpClass(cls):
        cls.video_a = VIDEOS_DIR / "sample3-20s-360p.mp4"
        cls.video_b = VIDEOS_DIR / "sample4-30s-360p.mp4"
        if not cls.video_a.exists() or not cls.video_b.exists():
            raise unittest.SkipTest("sample3/sample4 fixture videos not found")
        cls.tmpdir = tempfile.mkdtemp(prefix="dupefinder_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_first_frames_do_not_hash_as_duplicates(self):
        frames_a = extract_frames_dense(self.video_a, Path(self.tmpdir) / "a", 2.0)
        frames_b = extract_frames_dense(self.video_b, Path(self.tmpdir) / "b", 2.0)
        self.assertGreater(len(frames_a), 0)
        self.assertGreater(len(frames_b), 0)

        img_a = cv2.imread(str(frames_a[0]))
        img_b = cv2.imread(str(frames_b[0]))
        h_a = hashing.compute_hashes(img_a)
        h_b = hashing.compute_hashes(img_b)

        for variant in ("ahash", "dhash", "phash"):
            distance = hashing.hamming_distance(h_a[variant], h_b[variant])
            self.assertGreater(
                distance, 4, f"{variant} distance {distance} unexpectedly low for distinct videos"
            )


if __name__ == "__main__":
    unittest.main()
