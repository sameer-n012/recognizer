import argparse
import logging
from pathlib import Path

import pandas as pd

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main(
    index_path: Path,
    exact_groups_path: Path,
    near_dup_groups_path: Path,
    scored_pairs_path: Path,
    video_alignment_path: Path,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)
    uf = UnionFind()
    for asset_id in df["asset_id"]:
        uf.add(asset_id)

    n_edges = 0

    for path in (exact_groups_path, near_dup_groups_path):
        if not path.exists():
            continue
        groups = pd.read_parquet(path)
        for _, members in groups.groupby("group_id"):
            ids = members["asset_id"].tolist()
            for other in ids[1:]:
                uf.union(ids[0], other)
                n_edges += 1

    if scored_pairs_path.exists():
        scored = pd.read_parquet(scored_pairs_path)
        resolved = scored[scored["status"] == "resolved"]
        for _, row in resolved.iterrows():
            uf.union(row["asset_id_a"], row["asset_id_b"])
            n_edges += 1

    if video_alignment_path.exists():
        alignment = pd.read_parquet(video_alignment_path)
        confirmed = alignment[alignment["is_duplicate"]] if not alignment.empty else alignment
        for _, row in confirmed.iterrows():
            uf.union(row["asset_id_a"], row["asset_id_b"])
            n_edges += 1

    roots: dict[str, list[str]] = {}
    for asset_id in df["asset_id"]:
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
        f"Clustering complete: {n_groups} duplicate groups, {len(result)} assets "
        f"({n_edges} merge edges) -> {out_path}"
    )
    logger.info(
        "Clustering complete: %d groups, %d assets, %d edges", n_groups, len(result), n_edges
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--exact-groups", type=Path, default=None)
    ap.add_argument("--near-dup-groups", type=Path, default=None)
    ap.add_argument("--scored-pairs", type=Path, default=None)
    ap.add_argument("--video-alignment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "cluster_duplicates")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    exact_groups_path = resolve_path(resolve(args.exact_groups, section.get("exact_groups")))
    near_dup_groups_path = resolve_path(
        resolve(args.near_dup_groups, section.get("near_dup_groups"))
    )
    scored_pairs_path = resolve_path(resolve(args.scored_pairs, section.get("scored_pairs")))
    video_alignment_path = resolve_path(
        resolve(args.video_alignment, section.get("video_alignment"))
    )
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting duplicate clustering")
    main(
        index_path,
        exact_groups_path,
        near_dup_groups_path,
        scored_pairs_path,
        video_alignment_path,
        out_path,
    )
    logger.info("Duplicate clustering complete")
