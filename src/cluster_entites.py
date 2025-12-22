import argparse
import json
import logging
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3


def main(fused_dir: Path, out_dir: Path, min_cluster_size, min_samples):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all fused embeddings
    entities = []
    embeddings = []
    for f in fused_dir.glob("*.npy"):
        eid = f.stem
        emb = np.load(f)
        embeddings.append(emb)
        entities.append(eid)

    embeddings = np.stack(embeddings)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean"
    )
    labels = clusterer.fit_predict(embeddings)

    assignments = pd.DataFrame({"entity_id": entities, "cluster_id": labels})
    assignments_path = out_dir / "assignments.parquet"
    assignments.to_parquet(assignments_path, index=False)

    # Save centroids
    centroids = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        centroids[cluster_id] = embeddings[mask].mean(axis=0).tolist()
    centroids_path = out_dir / "centroids.npy"
    np.save(centroids_path, centroids)

    print(
        f"Clustering complete. {len(set(labels)) - (1 if -1 in labels else 0)} clusters found."
    )
    print(f"Assignments → {assignments_path}")
    print(f"Centroids → {centroids_path}")
    logger.info(
        f"Clustering complete: {len(set(labels)) - (1 if -1 in labels else 0)} clusters found -> ({assignments_path}, {centroids_path})"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-dir", type=Path, default=Path("data/embeddings/fused"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clusters"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/cluster_entities.log"))
    ap.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting clustering process")

    main(args.fused_dir, args.out_dir, args.min_cluster_size, args.min_samples)

    logger.info("Clustering process complete")
