import argparse
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path
from hashing import hamming_distance

logger = logging.getLogger(__name__)


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def band_keys(hex_hash: str, num_bands: int) -> list[str]:
    """Split a hex-encoded hash into num_bands disjoint, contiguous substrings — a
    partition of the full hash into non-overlapping bit-ranges, used as LSH bucket
    keys."""
    n = len(hex_hash)
    base, extra = divmod(n, num_bands)
    keys = []
    start = 0
    for i in range(num_bands):
        length = base + (1 if i < extra else 0)
        keys.append(hex_hash[start : start + length])
        start += length
    return keys


def build_candidate_pairs(
    frames_by_asset: dict[str, list[dict]], thresholds: dict[str, int]
) -> set[tuple[str, str]]:
    """LSH-style candidate generation, exact (no recall loss vs. brute force): for
    each hash type, split every hash into (threshold + 1) disjoint bands. By
    pigeonhole, two hashes within that hash type's Hamming-distance threshold must
    share at least one band exactly — so bucketing every (asset, frame) row by every
    band, for every hash type, and taking any two assets that ever land in the same
    bucket is guaranteed to include every pair the O(n^2) brute-force loop would have
    flagged, while pruning away the vast majority of unrelated pairs up front."""
    candidates: set[tuple[str, str]] = set()
    for hash_type, threshold in thresholds.items():
        num_bands = threshold + 1
        buckets: dict[tuple[int, str], list[str]] = defaultdict(list)
        for asset_id, frames in frames_by_asset.items():
            for frame in frames:
                for band_idx, key in enumerate(band_keys(frame[hash_type], num_bands)):
                    buckets[(band_idx, key)].append(asset_id)

        for members in buckets.values():
            unique_assets = sorted(set(members))
            if len(unique_assets) < 2:
                continue
            candidates.update(itertools.combinations(unique_assets, 2))

    return candidates


def hashes_match(a: dict, b: dict, thresholds: dict[str, int]) -> bool:
    """Duplicate if ANY hash variant is within its own strict threshold."""
    return (
        hamming_distance(a["ahash"], b["ahash"]) <= thresholds["ahash"]
        or hamming_distance(a["dhash"], b["dhash"]) <= thresholds["dhash"]
        or hamming_distance(a["phash"], b["phash"]) <= thresholds["phash"]
    )


def asset_pair_is_duplicate(
    frames_a: list[dict],
    frames_b: list[dict],
    thresholds: dict[str, int],
    min_frame_match_ratio: float,
) -> bool:
    if len(frames_a) == 1 and len(frames_b) == 1:
        return hashes_match(frames_a[0], frames_b[0], thresholds)

    # At least one side has multiple (sparse video) frames: require that a large
    # enough fraction of the shorter side's frames each have a matching frame on
    # the other side. A single shared frame (e.g. a photo matching one video frame)
    # is enough when one side is a lone image.
    if len(frames_a) == 1 or len(frames_b) == 1:
        short, long_ = (frames_a, frames_b) if len(frames_a) == 1 else (frames_b, frames_a)
        return any(hashes_match(short[0], f, thresholds) for f in long_)

    matched = sum(
        1 for fa in frames_a if any(hashes_match(fa, fb, thresholds) for fb in frames_b)
    )
    ratio = matched / min(len(frames_a), len(frames_b))
    return ratio >= min_frame_match_ratio


def main(
    exact_groups_path: Path,
    hashes_path: Path,
    out_path: Path,
    ahash_threshold: int,
    dhash_threshold: int,
    phash_threshold: int,
    min_frame_match_ratio: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = {"ahash": ahash_threshold, "dhash": dhash_threshold, "phash": phash_threshold}

    resolved_ids: set[str] = set()
    if exact_groups_path.exists():
        exact_df = pd.read_parquet(exact_groups_path)
        resolved_ids = set(exact_df["asset_id"])

    hashes_df = pd.read_parquet(hashes_path)
    frames_by_asset: dict[str, list[dict]] = {}
    for asset_id, group in hashes_df.groupby("asset_id"):
        if asset_id in resolved_ids:
            continue
        rows = group.sort_values("frame_index").to_dict("records")
        frames_by_asset[asset_id] = rows

    asset_ids = list(frames_by_asset.keys())
    uf = UnionFind(asset_ids)

    candidate_pairs = build_candidate_pairs(frames_by_asset, thresholds)

    pairs_checked = 0
    matches_found = 0
    for a_id, b_id in tqdm(candidate_pairs, desc="Comparing hash candidates", unit="pair"):
        pairs_checked += 1
        if asset_pair_is_duplicate(
            frames_by_asset[a_id], frames_by_asset[b_id], thresholds, min_frame_match_ratio
        ):
            uf.union(a_id, b_id)
            matches_found += 1

    roots: dict[str, list[str]] = {}
    for asset_id in asset_ids:
        root = uf.find(asset_id)
        roots.setdefault(root, []).append(asset_id)

    rows = []
    for group_id, (_, members) in enumerate(roots.items()):
        if len(members) < 2:
            continue
        for asset_id in members:
            rows.append({"asset_id": asset_id, "group_id": group_id})

    result = pd.DataFrame(rows, columns=["asset_id", "group_id"])
    result.to_parquet(out_path, index=False)

    n_groups = result["group_id"].nunique() if not result.empty else 0
    print(
        f"Near-duplicate detection complete: {n_groups} groups, {len(result)} assets "
        f"({pairs_checked} pairs checked, {matches_found} matched) -> {out_path}"
    )
    logger.info(
        "Near-duplicate detection complete: %d groups, %d assets (%d pairs checked, %d matched)",
        n_groups,
        len(result),
        pairs_checked,
        matches_found,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--exact-groups", type=Path, default=None)
    ap.add_argument("--hashes", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--ahash-threshold", type=int, default=None)
    ap.add_argument("--dhash-threshold", type=int, default=None)
    ap.add_argument("--phash-threshold", type=int, default=None)
    ap.add_argument("--min-frame-match-ratio", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "near_duplicates")

    exact_groups_path = resolve_path(resolve(args.exact_groups, section.get("exact_groups")))
    hashes_path = resolve_path(resolve(args.hashes, section.get("hashes")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    ahash_threshold = resolve(args.ahash_threshold, section.get("ahash_threshold"))
    dhash_threshold = resolve(args.dhash_threshold, section.get("dhash_threshold"))
    phash_threshold = resolve(args.phash_threshold, section.get("phash_threshold"))
    min_frame_match_ratio = resolve(
        args.min_frame_match_ratio, section.get("min_frame_match_ratio")
    )

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting near-duplicate detection")
    main(
        exact_groups_path,
        hashes_path,
        out_path,
        ahash_threshold,
        dhash_threshold,
        phash_threshold,
        min_frame_match_ratio,
    )
    logger.info("Near-duplicate detection complete")
