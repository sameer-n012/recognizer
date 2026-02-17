import argparse
import itertools as it
import json
import logging
from pathlib import Path

# import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

logger = logging.getLogger(__name__)


def load_similarity_config(config_path: Path) -> dict[str, dict | float]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config missing: {config_path}")
    payload = json.loads(config_path.read_text())
    similarity = {
        "metric": payload.get("metric", "cosine"),
        "transform": payload.get("transform", "reciprocal"),
        "scale": float(payload.get("scale", 1.0)),
        "epsilon": float(payload.get("epsilon", 1e-6)),
    }
    similarity["hdbscan"] = {
        "min_cluster_size": int(payload.get("hdbscan", {}).get("min_cluster_size", 3)),
        "min_samples": int(payload.get("hdbscan", {}).get("min_samples", 3)),
    }
    similarity["cluster_split"] = {
        "max_cluster_size": int(
            payload.get("cluster_split", {}).get("max_cluster_size", 40)
        ),
        "distance_threshold": float(
            payload.get("cluster_split", {}).get("distance_threshold", 0.25)
        ),
    }
    return similarity


def split_cluster(
    embeddings: np.ndarray, metric: str, distance_threshold: float
) -> np.ndarray:
    if embeddings.shape[0] <= 1:
        return np.zeros(embeddings.shape[0], dtype=int)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage="average",
    )
    return clusterer.fit_predict(embeddings)


def remap_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    cluster_split_cfg: dict[str, float | int],
    metric: str,
) -> np.ndarray:
    final_labels = np.full_like(labels, fill_value=-1)
    next_cluster_id = 0
    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if len(indices) <= cluster_split_cfg["max_cluster_size"]:
            final_labels[indices] = next_cluster_id
            next_cluster_id += 1
            continue

        sub_labels = split_cluster(
            embeddings[indices],
            metric,
            cluster_split_cfg["distance_threshold"],
        )
        unique_sub = sorted(set(sub_labels))
        mapping = {
            sub_id: next_cluster_id + idx for idx, sub_id in enumerate(unique_sub)
        }
        for idx, sub_id in zip(indices, sub_labels):
            final_labels[idx] = mapping[sub_id]
        next_cluster_id += len(unique_sub)
        logger.info(
            "Split cluster %d (size %d) into %d sub-clusters",
            cluster_id,
            len(indices),
            len(unique_sub),
        )
    return final_labels


def main(fused_dir: Path, out_dir: Path, config_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_similarity_config(config_path)
    metric = config["metric"]
    hdbscan_cfg = config["hdbscan"]

    entities = []
    embeddings = []
    for f in fused_dir.glob("*.npy"):
        entities.append(f.stem)
        embeddings.append(np.load(f))

    if not embeddings:
        raise ValueError("No fused embeddings found")

    embeddings = np.stack(embeddings)

    clusterer = HDBSCAN(
        min_cluster_size=hdbscan_cfg["min_cluster_size"],
        min_samples=hdbscan_cfg["min_samples"],
        metric=metric,
        cluster_selection_method="eom",
        # algorithm="brute",
        copy=True,
    )
    labels = clusterer.fit_predict(embeddings)
    final_labels = remap_clusters(
        labels,
        embeddings,
        config["cluster_split"],
        metric,
    )

    assignments = pd.DataFrame({"entity_id": entities, "cluster_id": final_labels})
    assignments_path = out_dir / "assignments.parquet"
    assignments.to_parquet(assignments_path, index=False)

    centroids = {}
    for cluster_id in sorted(set(final_labels) - {-1}):
        mask = final_labels == cluster_id
        centroids[cluster_id] = embeddings[mask].mean(axis=0).tolist()
    centroids_path = out_dir / "centroids.npy"
    np.save(centroids_path, centroids)

    cluster_count = len(centroids)
    print(f"Clustering complete. {cluster_count} clusters found.")
    print(f"Assignments → {assignments_path}")
    print(f"Centroids → {centroids_path}")
    logger.info(
        "Clustering complete: %d clusters -> (%s, %s)",
        cluster_count,
        assignments_path,
        centroids_path,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-dir", type=Path, default=Path("data/embeddings/fused"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clusters"))
    ap.add_argument("--config", type=Path, default=Path("configs/similarity.json"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/cluster_entities.log"))
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting clustering process")
    main(args.fused_dir, args.out_dir, args.config)
    logger.info("Clustering process complete")
