import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def load_similarity_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": payload.get("metric", "cosine"),
        "merge_threshold": float(payload["merge_threshold"]),
        "noise_threshold": float(payload["noise_threshold"]),
        "small_cluster_threshold": float(payload["small_cluster_threshold"]),
        "small_cluster_max_size": int(payload["small_cluster_max_size"]),
        "small_to_large_threshold": float(payload["small_to_large_threshold"]),
        "large_cluster_max_size": int(payload["large_cluster_max_size"]),
        "large_cluster_distance_threshold": float(
            payload["large_cluster_distance_threshold"]
        ),
        "min_cluster_size": int(payload["min_cluster_size"]),
        "min_cluster_force_threshold": float(payload["min_cluster_force_threshold"]),
        "force_reassign_singletons": bool(payload["force_reassign_singletons"]),
        "force_reassign_noise": bool(payload["force_reassign_noise"]),
    }


def split_large_cluster(
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


def build_merge_map(
    cluster_ids: list[int],
    centroid_matrix: np.ndarray,
    threshold: float,
) -> dict[int, int]:
    adjacency: dict[int, set[int]] = {cid: set() for cid in cluster_ids}
    sim_matrix = cosine_similarity(centroid_matrix)
    n = len(cluster_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adjacency[cluster_ids[i]].add(cluster_ids[j])
                adjacency[cluster_ids[j]].add(cluster_ids[i])

    merged_map: dict[int, int] = {}
    new_cluster_id = 0
    for cid in cluster_ids:
        if cid in merged_map:
            continue
        stack = [cid]
        while stack:
            current = stack.pop()
            if current in merged_map:
                continue
            merged_map[current] = new_cluster_id
            stack.extend(adjacency[current])
        new_cluster_id += 1
    return merged_map


def main(
    cluster_dir: Path,
    out_dir: Path,
    similarity_cfg: dict[str, Any],
    merge_threshold: float | None,
    noise_threshold: float | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = cluster_dir / "assignments.parquet"
    centroids_path = cluster_dir / "centroids.npy"

    assignments = pd.read_parquet(assignments_path)
    centroids_dict = np.load(centroids_path, allow_pickle=True).item()

    similarity_cfg = load_similarity_config(similarity_cfg)
    merge_threshold = (
        merge_threshold
        if merge_threshold is not None
        else similarity_cfg["merge_threshold"]
    )
    noise_threshold = (
        noise_threshold
        if noise_threshold is not None
        else similarity_cfg["noise_threshold"]
    )

    cluster_ids = list(centroids_dict.keys())
    centroid_matrix = np.stack([centroids_dict[cid] for cid in cluster_ids])

    logger.info(
        "Merging clusters using %s metric (threshold %.3f) from %s",
        similarity_cfg["metric"],
        merge_threshold,
        "configs/config.json",
    )
    merged_map = build_merge_map(cluster_ids, centroid_matrix, merge_threshold)

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

    # Merge tiny clusters into nearest larger cluster if similarity is high enough.
    # This reduces fragmentation when embeddings are noisy.
    cluster_sizes = assignments["cluster_id"].value_counts().to_dict()
    small_cluster_ids = [
        cid
        for cid, size in cluster_sizes.items()
        if cid != -1 and size <= similarity_cfg["small_cluster_max_size"]
    ]

    if small_cluster_ids:
        final_centroids = {}
        for cid in sorted(set(assignments["cluster_id"]) - {-1}):
            ids = assignments[assignments["cluster_id"] == cid]["entity_id"].tolist()
            embs = [
                np.load(
                    cluster_dir.parent.parent / "data/embeddings/fused" / f"{eid}.npy"
                )
                for eid in ids
            ]
            final_centroids[cid] = np.mean(embs, axis=0)

        centroid_ids = list(final_centroids.keys())
        centroid_matrix = np.stack([final_centroids[cid] for cid in centroid_ids])
        sim_matrix = cosine_similarity(centroid_matrix)

        id_to_index = {cid: idx for idx, cid in enumerate(centroid_ids)}
        for cid in small_cluster_ids:
            if cid not in id_to_index:
                continue
            idx = id_to_index[cid]
            sims = sim_matrix[idx]
            sims[idx] = -1.0
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            target_cid = centroid_ids[best_idx]
            target_size = cluster_sizes.get(target_cid, 0)
            threshold = similarity_cfg["small_cluster_threshold"]
            if target_size >= similarity_cfg["large_cluster_max_size"]:
                threshold = max(threshold, similarity_cfg["small_to_large_threshold"])
            if best_sim >= threshold:
                assignments.loc[assignments["cluster_id"] == cid, "cluster_id"] = (
                    target_cid
                )

    # Force-merge clusters smaller than min_cluster_size to the closest cluster
    # if similarity is above min_cluster_force_threshold.
    cluster_sizes = assignments["cluster_id"].value_counts().to_dict()
    min_cluster_ids = [
        cid
        for cid, size in cluster_sizes.items()
        if cid != -1 and size < similarity_cfg["min_cluster_size"]
    ]

    if min_cluster_ids:
        final_centroids = {}
        for cid in sorted(set(assignments["cluster_id"]) - {-1}):
            ids = assignments[assignments["cluster_id"] == cid]["entity_id"].tolist()
            embs = [
                np.load(
                    cluster_dir.parent.parent / "data/embeddings/fused" / f"{eid}.npy"
                )
                for eid in ids
            ]
            final_centroids[cid] = np.mean(embs, axis=0)

        centroid_ids = list(final_centroids.keys())
        centroid_matrix = np.stack([final_centroids[cid] for cid in centroid_ids])
        sim_matrix = cosine_similarity(centroid_matrix)
        id_to_index = {cid: idx for idx, cid in enumerate(centroid_ids)}

        for cid in min_cluster_ids:
            if cid not in id_to_index:
                continue
            idx = id_to_index[cid]
            sims = sim_matrix[idx]
            sims[idx] = -1.0
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= similarity_cfg["min_cluster_force_threshold"]:
                target_cid = centroid_ids[best_idx]
                assignments.loc[assignments["cluster_id"] == cid, "cluster_id"] = (
                    target_cid
                )

    # Split overly large clusters with a stricter distance threshold.
    cluster_sizes = assignments["cluster_id"].value_counts().to_dict()
    large_cluster_ids = [
        cid
        for cid, size in cluster_sizes.items()
        if cid != -1 and size > similarity_cfg["large_cluster_max_size"]
    ]

    if large_cluster_ids:
        next_cluster_id = (
            max(cid for cid in set(assignments["cluster_id"]) if cid != -1) + 1
        )
        for cid in large_cluster_ids:
            ids = assignments[assignments["cluster_id"] == cid]["entity_id"].tolist()
            if len(ids) <= 1:
                continue
            embs = [
                np.load(
                    cluster_dir.parent.parent / "data/embeddings/fused" / f"{eid}.npy"
                )
                for eid in ids
            ]
            embs = np.stack(embs)
            sub_labels = split_large_cluster(
                embs,
                similarity_cfg["metric"],
                similarity_cfg["large_cluster_distance_threshold"],
            )
            if len(set(sub_labels)) <= 1:
                continue
            mapping = {}
            for sub_id in sorted(set(sub_labels)):
                mapping[sub_id] = next_cluster_id
                next_cluster_id += 1
            for eid, sub_id in zip(ids, sub_labels):
                assignments.loc[assignments["entity_id"] == eid, "cluster_id"] = (
                    mapping[sub_id]
                )

    # Force reassign any remaining singletons to the closest cluster.
    if similarity_cfg["force_reassign_singletons"]:
        cluster_sizes = assignments["cluster_id"].value_counts().to_dict()
        singleton_ids = [
            cid for cid, size in cluster_sizes.items() if cid != -1 and size == 1
        ]
        if singleton_ids:
            final_centroids = {}
            for cid in sorted(set(assignments["cluster_id"]) - {-1}):
                ids = assignments[assignments["cluster_id"] == cid][
                    "entity_id"
                ].tolist()
                embs = [
                    np.load(
                        cluster_dir.parent.parent
                        / "data/embeddings/fused"
                        / f"{eid}.npy"
                    )
                    for eid in ids
                ]
                final_centroids[cid] = np.mean(embs, axis=0)

            centroid_ids = list(final_centroids.keys())
            centroid_matrix = np.stack([final_centroids[cid] for cid in centroid_ids])
            sim_matrix = cosine_similarity(centroid_matrix)
            id_to_index = {cid: idx for idx, cid in enumerate(centroid_ids)}

            for cid in singleton_ids:
                if cid not in id_to_index:
                    continue
                idx = id_to_index[cid]
                sims = sim_matrix[idx]
                sims[idx] = -1.0
                best_idx = int(np.argmax(sims))
                target_cid = centroid_ids[best_idx]
                assignments.loc[assignments["cluster_id"] == cid, "cluster_id"] = (
                    target_cid
                )

    # Force reassign any remaining noise points (-1) to the closest cluster.
    if similarity_cfg["force_reassign_noise"]:
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

            final_centroids = {}
            for cid in sorted(set(assignments["cluster_id"]) - {-1}):
                ids = assignments[assignments["cluster_id"] == cid][
                    "entity_id"
                ].tolist()
                embs = [
                    np.load(
                        cluster_dir.parent.parent
                        / "data/embeddings/fused"
                        / f"{eid}.npy"
                    )
                    for eid in ids
                ]
                final_centroids[cid] = np.mean(embs, axis=0)

            centroid_ids = list(final_centroids.keys())
            centroid_matrix = np.stack([final_centroids[cid] for cid in centroid_ids])
            sims = cosine_similarity(noise_embeds, centroid_matrix)
            new_labels = sims.argmax(axis=1)

            for i, eid in enumerate(noise_entities):
                assignments.loc[assignments["entity_id"] == eid, "cluster_id"] = (
                    centroid_ids[new_labels[i]]
                )

    assignments.to_parquet(out_dir / "assignments_refined.parquet", index=False)
    merged_map_json = {int(k): int(v) for k, v in merged_map.items()}
    with open(out_dir / "merges.json", "w") as f:
        json.dump(merged_map_json, f)

    print("Cluster refinement complete")
    print(f"Refined assignments → {out_dir / 'assignments_refined.parquet'}")
    print(f"Merges → {out_dir / 'merges.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--cluster-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--merge-threshold", type=float, default=None)
    ap.add_argument("--noise-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "refine_clusters")

    cluster_dir = resolve_path(resolve(args.cluster_dir, section.get("cluster_dir")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    merge_threshold = resolve(args.merge_threshold, section.get("merge_threshold"))
    noise_threshold = resolve(args.noise_threshold, section.get("noise_threshold"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting cluster refinement process")

    main(
        cluster_dir,
        out_dir,
        section,
        merge_threshold,
        noise_threshold,
    )

    logger.info("Cluster refinement process complete")
