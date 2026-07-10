import argparse
import logging
from pathlib import Path

import pandas as pd

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def compute_duration_bucket(row: pd.Series, epsilon_seconds: float) -> float:
    """Videos within epsilon_seconds of each other are treated as "the same length"
    (same bucket) so minor re-encode/frame-rate rounding doesn't matter; anything
    longer wins outright, ahead of resolution — a higher-resolution video that's only
    a subsection of another should never outrank the full-length one."""
    duration = row.get("duration")
    if row["type"] != "video" or not duration or duration <= 0:
        return 0.0
    return round(float(duration) / epsilon_seconds) * epsilon_seconds


def compute_quality_key(row: pd.Series) -> float:
    """Bitrate for video (size/duration), raw file size for images — both are rough
    proxies for "least compressed / most information retained", used only to break
    ties after duration and resolution."""
    if row["type"] == "video" and row.get("duration") and row["duration"] > 0:
        return float(row["size"]) / float(row["duration"])
    return float(row["size"])


def main(
    index_path: Path, groups_path: Path, out_path: Path, duration_epsilon_seconds: float
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path).set_index("asset_id")
    groups = pd.read_parquet(groups_path)

    if groups.empty:
        pd.DataFrame(
            columns=[
                "asset_id",
                "group_id",
                "path",
                "rank",
                "is_keeper",
                "duration_bucket",
                "resolution",
                "quality_key",
            ]
        ).to_parquet(out_path, index=False)
        print("No duplicate groups to rank")
        return

    rows = []
    for group_id, members in groups.groupby("group_id"):
        member_rows = []
        for asset_id in members["asset_id"]:
            if asset_id not in df.index:
                continue
            info = df.loc[asset_id]
            resolution = float(info["width"] or 0) * float(info["height"] or 0)
            member_rows.append(
                {
                    "asset_id": asset_id,
                    "group_id": int(group_id),
                    "path": info["path"],
                    "duration_bucket": compute_duration_bucket(info, duration_epsilon_seconds),
                    "resolution": resolution,
                    "quality_key": compute_quality_key(info),
                    "mtime": info["mtime"],
                }
            )

        ranked = sorted(
            member_rows,
            key=lambda r: (r["duration_bucket"], r["resolution"], r["quality_key"], r["mtime"]),
            reverse=True,
        )
        for rank, entry in enumerate(ranked):
            entry["rank"] = rank
            entry["is_keeper"] = rank == 0
            rows.append(entry)

    result = pd.DataFrame(
        rows,
        columns=[
            "asset_id",
            "group_id",
            "path",
            "rank",
            "is_keeper",
            "duration_bucket",
            "resolution",
            "quality_key",
            "mtime",
        ],
    )
    result.to_parquet(out_path, index=False)

    n_groups = result["group_id"].nunique() if not result.empty else 0
    print(f"Keeper ranking complete: {n_groups} groups, {len(result)} assets -> {out_path}")
    logger.info("Keeper ranking complete: %d groups, %d assets", n_groups, len(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--groups", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--duration-epsilon-seconds", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "rank_group_members")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    groups_path = resolve_path(resolve(args.groups, section.get("groups")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    duration_epsilon_seconds = resolve(
        args.duration_epsilon_seconds, section.get("duration_epsilon_seconds"), 1.5
    )

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting keeper ranking")
    main(index_path, groups_path, out_path, duration_epsilon_seconds)
    logger.info("Keeper ranking complete")
