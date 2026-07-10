import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def main(
    index_path: Path,
    exact_groups_path: Path,
    near_dup_groups_path: Path,
    clip_dir: Path,
    out_path: Path,
    top_k: int,
    similarity_floor: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    resolved_ids: set[str] = set()
    for path in (exact_groups_path, near_dup_groups_path):
        if path.exists():
            resolved_ids |= set(pd.read_parquet(path)["asset_id"])

    asset_ids = []
    embeddings = []
    for asset_id in df["asset_id"]:
        if asset_id in resolved_ids:
            continue
        emb_path = clip_dir / f"{asset_id}.npy"
        if not emb_path.exists():
            continue
        asset_ids.append(asset_id)
        embeddings.append(np.load(emb_path))

    if len(asset_ids) < 2:
        pd.DataFrame(columns=["asset_id_a", "asset_id_b", "coarse_similarity"]).to_parquet(
            out_path, index=False
        )
        print("Not enough unresolved assets with coarse embeddings to build candidates")
        return

    matrix = np.stack(embeddings)
    sim = matrix @ matrix.T

    pairs = {}
    n = len(asset_ids)
    for i in tqdm(range(n), desc="Building candidates", unit="asset"):
        row = sim[i]
        # top_k neighbors excluding self, above the similarity floor
        order = np.argsort(-row)
        taken = 0
        for j in order:
            if j == i:
                continue
            if row[j] < similarity_floor:
                break
            a, b = (asset_ids[i], asset_ids[j]) if i < j else (asset_ids[j], asset_ids[i])
            key = (a, b)
            if key not in pairs or row[j] > pairs[key]:
                pairs[key] = float(row[j])
            taken += 1
            if taken >= top_k:
                break

    result = pd.DataFrame(
        [{"asset_id_a": a, "asset_id_b": b, "coarse_similarity": s} for (a, b), s in pairs.items()]
    )
    result.to_parquet(out_path, index=False)

    print(f"Candidate generation complete: {len(result)} pairs -> {out_path}")
    logger.info("Candidate generation complete: %d pairs", len(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--exact-groups", type=Path, default=None)
    ap.add_argument("--near-dup-groups", type=Path, default=None)
    ap.add_argument("--clip-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--similarity-floor", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "build_candidates")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    exact_groups_path = resolve_path(resolve(args.exact_groups, section.get("exact_groups")))
    near_dup_groups_path = resolve_path(
        resolve(args.near_dup_groups, section.get("near_dup_groups"))
    )
    clip_dir = resolve_path(resolve(args.clip_dir, section.get("clip_dir")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    top_k = resolve(args.top_k, section.get("top_k"))
    similarity_floor = resolve(args.similarity_floor, section.get("similarity_floor"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting candidate generation")
    main(
        index_path,
        exact_groups_path,
        near_dup_groups_path,
        clip_dir,
        out_path,
        top_k,
        similarity_floor,
    )
    logger.info("Candidate generation complete")
