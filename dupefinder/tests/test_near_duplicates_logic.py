import itertools
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from near_duplicates import (  # noqa: E402
    UnionFind,
    asset_pair_is_duplicate,
    band_keys,
    build_candidate_pairs,
    hashes_match,
)

THRESHOLDS = {"ahash": 4, "dhash": 4, "phash": 4}


def h(ahash="0" * 16, dhash="0" * 16, phash="0" * 16):
    return {"ahash": ahash, "dhash": dhash, "phash": phash}


class TestHashesMatch(unittest.TestCase):
    def test_identical_hashes_match(self):
        self.assertTrue(hashes_match(h(), h(), THRESHOLDS))

    def test_any_single_variant_within_threshold_is_enough(self):
        # ahash and dhash wildly different, but phash identical -> still a match
        # (the OR-across-variants design: one confident signal is enough).
        a = h(ahash="f" * 16, dhash="f" * 16, phash="0" * 16)
        b = h(ahash="0" * 16, dhash="0" * 16, phash="0" * 16)
        self.assertTrue(hashes_match(a, b, THRESHOLDS))

    def test_all_variants_far_apart_is_not_a_match(self):
        a = h(ahash="f" * 16, dhash="f" * 16, phash="f" * 16)
        b = h(ahash="0" * 16, dhash="0" * 16, phash="0" * 16)
        self.assertFalse(hashes_match(a, b, THRESHOLDS))


class TestAssetPairIsDuplicate(unittest.TestCase):
    def test_image_vs_image(self):
        self.assertTrue(asset_pair_is_duplicate([h()], [h()], THRESHOLDS, 0.6))

    def test_image_vs_video_any_frame_match_is_enough(self):
        photo = [h()]
        video_frames = [h(ahash="f" * 16, dhash="f" * 16, phash="f" * 16), h()]
        self.assertTrue(asset_pair_is_duplicate(photo, video_frames, THRESHOLDS, 0.6))

    def test_image_vs_video_no_frame_matches(self):
        photo = [h()]
        video_frames = [h(ahash="f" * 16, dhash="f" * 16, phash="f" * 16)] * 3
        self.assertFalse(asset_pair_is_duplicate(photo, video_frames, THRESHOLDS, 0.6))

    def test_video_vs_video_above_ratio_threshold(self):
        # all 3 shorter-side frames have a match -> ratio 1.0 >= 0.6
        a_frames = [h(), h(), h(), h(ahash="f" * 16, dhash="f" * 16, phash="f" * 16)]
        b_frames = [h(), h(), h()]
        self.assertTrue(asset_pair_is_duplicate(a_frames, b_frames, THRESHOLDS, 0.6))

    def test_video_vs_video_below_ratio_threshold(self):
        a_frames = [h(ahash="f" * 16, dhash="f" * 16, phash="f" * 16)] * 3 + [h()]
        b_frames = [h(), h(), h()]
        # only 1 of 3 shorter-side frames matches -> ratio 0.33 < 0.6
        self.assertFalse(asset_pair_is_duplicate(a_frames, b_frames, THRESHOLDS, 0.6))


class TestUnionFind(unittest.TestCase):
    def test_union_and_find_groups_transitively(self):
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))
        self.assertNotEqual(uf.find("a"), uf.find("d"))


class TestBandKeys(unittest.TestCase):
    def test_partitions_full_hash_without_overlap(self):
        h_ = "0123456789abcdef"
        keys = band_keys(h_, 3)
        self.assertEqual("".join(keys), h_)

    def test_identical_hashes_share_every_band(self):
        h_ = "abcdef0123456789"
        self.assertEqual(band_keys(h_, 5), band_keys(h_, 5))


class TestBuildCandidatePairsNoRecallLoss(unittest.TestCase):
    """The whole point of band_keys/build_candidate_pairs is to prune the O(n^2) loop
    without ever missing a pair the brute-force scan would have found (pigeonhole
    guarantee: threshold+1 disjoint bands means at least one band must match exactly
    for any pair within the threshold). Verify that guarantee empirically against a
    brute-force ground truth over random data, rather than just trusting the proof."""

    def test_candidates_are_a_superset_of_brute_force_matches(self):
        random.seed(0)
        thresholds = {"ahash": 2, "dhash": 2, "phash": 2}

        def random_hex():
            return format(random.getrandbits(64), "016x")

        def flip_bits(hex_hash, num_bits):
            value = int(hex_hash, 16)
            for bit in random.sample(range(64), num_bits):
                value ^= 1 << bit
            return format(value, "016x")

        frames_by_asset = {}
        for i in range(40):
            frames_by_asset[f"asset{i}"] = [
                {"ahash": random_hex(), "dhash": random_hex(), "phash": random_hex()}
            ]
        # Deliberately plant near-duplicates at varying Hamming distances (0, 1, 2, 3)
        # on a single hash variant, so some are true matches (<=2) and some aren't (3).
        for i, num_bits in enumerate([0, 1, 2, 3] * 3):
            src = frames_by_asset[f"asset{i}"][0]
            near = dict(src)
            near["ahash"] = flip_bits(src["ahash"], num_bits) if num_bits else src["ahash"]
            frames_by_asset[f"near{i}"] = [near]

        candidates = build_candidate_pairs(frames_by_asset, thresholds)
        candidate_set = {frozenset(pair) for pair in candidates}

        asset_ids = list(frames_by_asset.keys())
        true_pairs = {
            frozenset((a, b))
            for a, b in itertools.combinations(asset_ids, 2)
            if asset_pair_is_duplicate(frames_by_asset[a], frames_by_asset[b], thresholds, 0.8)
        }

        missing = true_pairs - candidate_set
        self.assertEqual(missing, set(), f"candidate generation missed true matches: {missing}")
        # Sanity check the test data actually exercises some true matches.
        self.assertGreater(len(true_pairs), 0)


if __name__ == "__main__":
    unittest.main()
