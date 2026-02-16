import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from threading import Lock

import pandas as pd
from flask import Flask, jsonify, render_template, send_file
from readerwriterlock import rwlock

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_DIR = Path("templates")
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_PORT = 5033


app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder="static")

# Load once at startup
context = {
    "assignments": None,
    "file_index": None,
    "entities": {},
    "asset_lookup": {},
    "lock": rwlock.RWLockFairD(),
}


def refresh_context():
    with context["lock"].gen_wlock():
        context["assignments"] = pd.read_parquet(
            DATA_DIR / "clusters" / "refined" / "assignments_refined.parquet"
        )
        context["file_index"] = pd.read_parquet(CACHE_DIR / "file_index.parquet")

        for p in (DATA_DIR / "entities").glob("*.json"):
            e = json.loads(p.read_text())
            context["entities"][e["entity_id"]] = e

        context["asset_lookup"] = {
            row["asset_id"]: row["path"] for _, row in context["file_index"].iterrows()
        }


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
    clusters = {}
    with context["lock"].gen_rlock():
        for _, row in context["assignments"].iterrows():
            cid = int(row["cluster_id"])
            if cid == -1:
                continue
            clusters.setdefault(cid, []).append(row["entity_id"])

    logger.info(f"Clusters overview requested, total clusters: {len(clusters)}")
    return jsonify(
        {
            "clusters": [
                {"cluster_id": cid, "size": len(eids)}
                for cid, eids in sorted(clusters.items())
            ]
        }
    )


@app.route("/api/cluster/<int:cluster_id>")
def get_cluster(cluster_id):
    with context["lock"].gen_rlock():
        assignments = context["assignments"]
        entities = context["entities"]
        rows = assignments[assignments["cluster_id"] == cluster_id]
        items = []

        for _, row in rows.iterrows():
            eid = row["entity_id"]
            ent = entities[eid]
            asset_id = ent["asset_id"]
            asset_type = ent["type"]
            items.append(
                {
                    "entity_id": eid,
                    "asset_id": asset_id,
                    "path": asset_id,
                    "asset_type": asset_type,
                }
            )

    asset_ids = set()
    unique_items = []
    for item in items:
        if item["asset_id"] in asset_ids:
            continue
        asset_ids.add(item["asset_id"])
        unique_items.append(item)

    logger.info(f"Cluster {cluster_id} requested, size: {len(unique_items)}")
    return jsonify({"items": unique_items})


@app.route("/media/<asset_id>")
def serve_media(asset_id):
    path = None
    with context["lock"].gen_rlock():
        asset_lookup = context["asset_lookup"]
        path = asset_lookup.get(asset_id)
    logger.info(f"Media requested: {asset_id} -> {path}")
    if not path:
        return "Not found", 404
    print(path)
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
