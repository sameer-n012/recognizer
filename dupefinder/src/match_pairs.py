import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def classify_pair(
    type_a: str, type_b: str, similarity: float, resolve_threshold: float, escalate_threshold: float
) -> str:
    if similarity >= resolve_threshold:
        return "resolved"
    if type_a == "video" and type_b == "video" and similarity >= escalate_threshold:
        return "escalate"
    return "dropped"


def main(
    index_path: Path,
    candidates_path: Path,
    out_path: Path,
    resolve_threshold: float,
    escalate_threshold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path).set_index("asset_id")
    candidates = pd.read_parquet(candidates_path)

    rows = []
    for _, row in tqdm(
        candidates.iterrows(), total=len(candidates), desc="Scoring candidate pairs", unit="pair"
    ):
        a_id, b_id = row["asset_id_a"], row["asset_id_b"]
        if a_id not in df.index or b_id not in df.index:
            continue
        type_a, type_b = df.loc[a_id, "type"], df.loc[b_id, "type"]
        status = classify_pair(
            type_a, type_b, row["coarse_similarity"], resolve_threshold, escalate_threshold
        )
        if status == "dropped":
            continue
        rows.append(
            {
                "asset_id_a": a_id,
                "asset_id_b": b_id,
                "type_a": type_a,
                "type_b": type_b,
                "similarity": row["coarse_similarity"],
                "status": status,
            }
        )

    result = pd.DataFrame(
        rows, columns=["asset_id_a", "asset_id_b", "type_a", "type_b", "similarity", "status"]
    )
    result.to_parquet(out_path, index=False)

    n_resolved = (result["status"] == "resolved").sum() if not result.empty else 0
    n_escalate = (result["status"] == "escalate").sum() if not result.empty else 0
    print(
        f"Pair matching complete: {n_resolved} resolved, {n_escalate} escalated to tier 3 "
        f"-> {out_path}"
    )
    logger.info(
        "Pair matching complete: %d resolved, %d escalated", n_resolved, n_escalate
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--resolve-threshold", type=float, default=None)
    ap.add_argument("--escalate-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "match_pairs")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    candidates_path = resolve_path(resolve(args.candidates, section.get("candidates")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    resolve_threshold = resolve(args.resolve_threshold, section.get("resolve_threshold"))
    escalate_threshold = resolve(args.escalate_threshold, section.get("escalate_threshold"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting pair matching")
    main(index_path, candidates_path, out_path, resolve_threshold, escalate_threshold)
    logger.info("Pair matching complete")
