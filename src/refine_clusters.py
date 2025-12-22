import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MERGE_THRESHOLD = 0.85
DEFAULT_NOISE_THRESHOLD = 0.8


def main(
    cluster_dir: Path,
    out_dir: Path,
    merge_threshold: float,
    noise_threshold: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = cluster_dir / "assignments.parquet"
    centroids_path = cluster_dir / "centroids.npy"

    assignments = pd.read_parquet(assignments_path)
    centroids_dict = np.load(centroids_path, allow_pickle=True).item()

    cluster_ids = list(centroids_dict.keys())
    centroid_matrix = np.stack([centroids_dict[cid] for cid in cluster_ids])

    # Merge clusters with high similarity
    sim_matrix = cosine_similarity(centroid_matrix)
    merged_map = {}
    visited = set()
    new_cluster_id = 0
    for i, cid in tqdm(enumerate(cluster_ids), desc="Merging clusters", unit="cluster"):
        if cid in visited:
            continue
        merge_group = [cid]
        for j, other_cid in enumerate(cluster_ids):
            if i != j and sim_matrix[i, j] >= merge_threshold:
                merge_group.append(other_cid)
        for mc in merge_group:
            merged_map[mc] = new_cluster_id
            visited.add(mc)
        new_cluster_id += 1

    # Apply cluster merge
    assignments["cluster_id"] = assignments["cluster_id"].map(
        lambda x: merged_map.get(x, -1)
    )

    # Reassign noise points (-1) to nearest cluster if similarity >= noise_threshold
    noise_mask = assignments["cluster_id"] == -1
    if noise_mask.any():
        noise_entities = assignments[noise_mask]["entity_id"].tolist()
        noise_embeds = []
        for eid in noise_entities:
            emb = np.load(
                cluster_dir.parent.parent / "data/embeddings/fused" / f"{eid}.npy"
            )
            noise_embeds.append(emb)
        noise_embeds = np.stack(noise_embeds)

        final_centroids = []
        final_ids = []
        for cid in sorted(set(assignments["cluster_id"]) - {-1}):
            mask = assignments["cluster_id"] == cid
            ids = assignments[mask]["entity_id"].tolist()
            embs = [
                np.load(
                    cluster_dir.parent.parent / "data/embeddings/fused" / f"{eid}.npy"
                )
                for eid in ids
            ]
            centroid = np.mean(embs, axis=0)
            final_centroids.append(centroid)
            final_ids.append(cid)

        sims = cosine_similarity(noise_embeds, np.stack(final_centroids))
        new_labels = sims.argmax(axis=1)
        max_sims = sims.max(axis=1)

        for i, eid in enumerate(noise_entities):
            if max_sims[i] >= noise_threshold:
                assignments.loc[assignments["entity_id"] == eid, "cluster_id"] = (
                    final_ids[new_labels[i]]
                )

    assignments.to_parquet(out_dir / "assignments_refined.parquet", index=False)
    with open(out_dir / "merges.json", "w") as f:
        json.dump(merged_map, f)

    print("Cluster refinement complete")
    print(f"Refined assignments → {out_dir / 'assignments_refined.parquet'}")
    print(f"Merges → {out_dir / 'merges.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-dir", type=Path, default=Path("data/clusters"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clusters/refined"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/cluster_entities.log"))
    ap.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    ap.add_argument("--noise-threshold", type=float, default=DEFAULT_NOISE_THRESHOLD)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting cluster refinement process")

    main(args.cluster_dir, args.out_dir, args.merge_threshold, args.noise_threshold)

    logger.info("Cluster refinement process complete")
