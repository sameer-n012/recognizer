import json
import logging
import math
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, send_file
from readerwriterlock import rwlock

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
TEMPLATE_DIR = Path("templates")
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_PORT = 5033

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder="static")


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open() as cfg_file:
        raw = json.load(cfg_file)

    resolved: Dict[str, Any] = dict(raw)
    resolved["assignments_path"] = PROJECT_ROOT / Path(raw["assignments_path"])
    resolved["file_index"] = PROJECT_ROOT / Path(raw["file_index"])
    resolved["entities_dir"] = PROJECT_ROOT / Path(raw["entities_dir"])
    resolved["embeddings"] = {
        key: PROJECT_ROOT / Path(value) for key, value in raw["embeddings"].items()
    }
    similarity = raw.get("similarity", {})
    resolved["similarity"] = {
        "metric": similarity.get("metric", "manhattan"),
        "transform": similarity.get("transform", "reciprocal"),
        "scale": float(similarity.get("scale", 1.0)),
        "epsilon": float(similarity.get("epsilon", 1e-6)),
    }
    return resolved


context = {
    "assignments": None,
    "file_index": None,
    "entities": {},
    "asset_lookup": {},
    "clusters_overview": [],
    "cluster_details": {},
    "similarity_metric": None,
    "config": load_config(),
    "lock": rwlock.RWLockFairD(),
}


def compute_distance(
    vec: np.ndarray, centroid: np.ndarray, metric: str, epsilon: float
) -> float:
    if centroid.shape != vec.shape:
        raise ValueError("Embedding mismatch between centroid and entity vector")
    if metric == "manhattan":
        return float(np.sum(np.abs(vec - centroid)))
    if metric == "euclidean":
        return float(np.linalg.norm(vec - centroid))
    if metric == "cosine":
        denom = np.linalg.norm(vec) * np.linalg.norm(centroid)
        if denom < epsilon:
            return 1.0
        similarity = float(np.dot(vec, centroid) / denom)
        return max(0.0, min(1.0, 1.0 - similarity))
    raise ValueError(f"Unsupported similarity metric: {metric}")


def distance_to_similarity(distance: float, similarity_cfg: Dict[str, Any]) -> float:
    transform = similarity_cfg["transform"]
    scale = similarity_cfg["scale"]
    epsilon = similarity_cfg["epsilon"]
    if transform == "reciprocal":
        denominator = max(1.0 + scale * distance, epsilon)
        return float(scale / denominator)
    if transform == "linear":
        return float(max(0.0, scale - distance))
    if transform == "exp":
        return float(math.exp(-scale * distance))
    raise ValueError(f"Unsupported similarity transform: {transform}")


def load_entities(entities_dir: Path) -> Dict[str, Dict[str, Any]]:
    loaded = {}
    for entity_file in sorted(entities_dir.glob("*.json")):
        entity = json.loads(entity_file.read_text())
        loaded[entity["entity_id"]] = entity
    return loaded


def build_embedding_index(
    entity_ids: Iterable[str], embedding_dirs: Dict[str, Path]
) -> Dict[str, Dict[str, np.ndarray]]:
    embedding_index: Dict[str, Dict[str, np.ndarray]] = {
        modality: {} for modality in embedding_dirs
    }
    for entity_id in entity_ids:
        for modality, emb_dir in embedding_dirs.items():
            emb_path = emb_dir / f"{entity_id}.npy"
            if emb_path.exists():
                embedding_index[modality][entity_id] = np.load(emb_path)
    return embedding_index


def compute_cluster_centroids(
    members: Dict[int, list[str]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
) -> Dict[int, Dict[str, np.ndarray]]:
    centroids: Dict[int, Dict[str, np.ndarray]] = {}
    for cluster_id, entity_ids in members.items():
        modality_centroids: Dict[str, np.ndarray] = {}
        for modality, modality_data in embeddings.items():
            vectors = [modality_data[eid] for eid in entity_ids if eid in modality_data]
            if not vectors:
                continue
            stacked = np.stack(vectors)
            modality_centroids[modality] = np.mean(stacked, axis=0)
        centroids[cluster_id] = modality_centroids
    return centroids


def build_cluster_views(
    assignments: pd.DataFrame,
    entities: Dict[str, Dict[str, Any]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    asset_lookup: Dict[str, str],
    similarity_cfg: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    cluster_members: Dict[int, list[str]] = defaultdict(list)
    for _, row in assignments.iterrows():
        cluster_id = int(row["cluster_id"])
        if cluster_id < 0:
            continue
        entity_id = row["entity_id"]
        if entity_id not in entities:
            continue
        cluster_members[cluster_id].append(entity_id)

    centroids = compute_cluster_centroids(cluster_members, embeddings)
    clusters_overview: list[Dict[str, Any]] = []
    cluster_details: Dict[int, Dict[str, Any]] = {}

    for cluster_id, member_ids in sorted(cluster_members.items()):
        asset_counter = Counter()
        asset_set = set()
        modality_accumulator: Dict[str, list[float]] = {
            modality: [] for modality in embeddings
        }
        items: list[Dict[str, Any]] = []

        for entity_id in member_ids:
            entity = entities[entity_id]
            asset_id = entity["asset_id"]
            asset_type = entity.get("type", "unknown")
            asset_counter[asset_type] += 1
            asset_set.add(asset_id)

            frame_count = len(entity.get("frames", []))
            face_total = sum(
                len(frame.get("faces", [])) for frame in entity.get("frames", [])
            )
            similarities: Dict[str, float | None] = {}
            for modality in embeddings:
                centroid = centroids.get(cluster_id, {}).get(modality)
                embedding = embeddings[modality].get(entity_id)
                if centroid is not None and embedding is not None:
                    distance = compute_distance(
                        embedding,
                        centroid,
                        similarity_cfg["metric"],
                        similarity_cfg["epsilon"],
                    )
                    similarity = distance_to_similarity(distance, similarity_cfg)
                    similarities[modality] = similarity
                    modality_accumulator[modality].append(similarity)
                else:
                    similarities[modality] = None

            asset_path = asset_lookup.get(asset_id)
            items.append(
                {
                    "entity_id": entity_id,
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "asset_path": asset_path,
                    "asset_basename": Path(asset_path).name if asset_path else asset_id,
                    "frame_count": frame_count,
                    "faces_total": face_total,
                    "has_face": bool(entity.get("has_face")),
                    "similarities": similarities,
                }
            )

        items.sort(
            key=lambda entry: entry["similarities"].get("fused") or 0.0, reverse=True
        )

        modality_summary: Dict[str, Dict[str, Any]] = {}
        for modality in embeddings:
            sims = modality_accumulator[modality]
            avg_similarity = float(sum(sims) / len(sims)) if sims else None
            modality_summary[modality] = {
                "average_similarity": avg_similarity,
                "coverage": len(sims),
            }

        summary = {
            "cluster_id": cluster_id,
            "size": len(member_ids),
            "unique_assets": len(asset_set),
            "asset_breakdown": dict(asset_counter),
            "modality_summary": modality_summary,
        }

        clusters_overview.append(
            {
                "cluster_id": cluster_id,
                "size": summary["size"],
                "unique_assets": summary["unique_assets"],
                "asset_breakdown": summary["asset_breakdown"],
                "modality_summary": summary["modality_summary"],
            }
        )
        cluster_details[cluster_id] = {"summary": summary, "items": items}

    return clusters_overview, cluster_details


def refresh_context():
    with context["lock"].gen_wlock():
        config = context["config"]
        assignments = pd.read_parquet(config["assignments_path"])
        file_index = pd.read_parquet(config["file_index"])
        context["assignments"] = assignments
        context["file_index"] = file_index
        context["asset_lookup"] = {
            row["asset_id"]: row["path"] for _, row in file_index.iterrows()
        }
        context["entities"] = load_entities(config["entities_dir"])
        embedding_index = build_embedding_index(
            context["entities"].keys(), config["embeddings"]
        )

        clusters_overview, cluster_details = build_cluster_views(
            assignments,
            context["entities"],
            embedding_index,
            context["asset_lookup"],
            config["similarity"],
        )

        context["clusters_overview"] = clusters_overview
        context["cluster_details"] = cluster_details
        context["similarity_metric"] = {
            "metric": config["similarity"]["metric"],
            "transform": config["similarity"]["transform"],
            "scale": config["similarity"]["scale"],
        }

        logger.info(
            "Context refreshed (%d clusters, %d entities)",
            len(clusters_overview),
            len(context["entities"]),
        )


refresh_context()


@app.route("/")
def index():
    logger.info("Index page requested")
    return render_template("index.html")


@app.route("/api/refresh")
def refresh():
    logger.info("Refresh requested")
    refresh_context()
    return "Refreshed", 200


@app.route("/api/clusters")
def get_clusters():
    with context["lock"].gen_rlock():
        clusters = context["clusters_overview"]
        metric_info = context["similarity_metric"]
    logger.info("Clusters overview requested (%d clusters)", len(clusters))
    return jsonify({"clusters": clusters, "similarity_metric": metric_info})


@app.route("/api/cluster/<int:cluster_id>")
def get_cluster(cluster_id: int):
    with context["lock"].gen_rlock():
        detail = context["cluster_details"].get(cluster_id)
    if detail is None:
        logger.warning("Requested unknown cluster %d", cluster_id)
        return jsonify({"error": "Cluster not found"}), 404
    logger.info("Cluster %d detail requested", cluster_id)
    return jsonify(detail)


@app.route("/media/<asset_id>")
def serve_media(asset_id: str):
    with context["lock"].gen_rlock():
        asset_lookup = context["asset_lookup"]
        path = asset_lookup.get(asset_id)
    if not path:
        logger.warning("Media not found for %s", asset_id)
        return "Not found", 404
    logger.info("Media requested: %s -> %s", asset_id, path)
    return send_file(path, conditional=True)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--log-file", type=Path, default=PROJECT_ROOT / "logs" / "server.log"
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting server")
    app.run(debug=True, host="127.0.0.1", port=args.port)
