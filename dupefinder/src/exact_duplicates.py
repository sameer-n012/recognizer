import argparse
import logging
from pathlib import Path

import pandas as pd

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def main(index_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)
    df = df[df["sha256"].notna()]

    groups = []
    for group_id, (_, members) in enumerate(df.groupby("sha256")):
        if len(members) < 2:
            continue
        for asset_id in members["asset_id"]:
            groups.append({"asset_id": asset_id, "group_id": group_id, "sha256": members["sha256"].iloc[0]})

    result = pd.DataFrame(groups, columns=["asset_id", "group_id", "sha256"])
    result.to_parquet(out_path, index=False)

    n_groups = result["group_id"].nunique() if not result.empty else 0
    print(f"Exact-duplicate detection complete: {n_groups} groups, {len(result)} assets -> {out_path}")
    logger.info("Exact-duplicate detection complete: %d groups, %d assets", n_groups, len(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "exact_duplicates")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting exact-duplicate detection")
    main(index_path, out_path)
    logger.info("Exact-duplicate detection complete")
